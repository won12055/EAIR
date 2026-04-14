# hop.py
"""
EAIR: Evidence-Aware Iterative Reasoning for Multi-hop Question Answering

This module implements:
- ECDLogitsProcessor: Evidence-Conditioned Contrastive Decoding (Eq. 7 in paper)
- LLMClient: Model wrapper with standard and ECD generation
- Parsing helpers for hop decomposition and entity extraction
"""
import re
from typing import Any, Dict, List, Tuple, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)


MODEL_NAME_TO_PATH: Dict[str, str] = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
}


# =========================================================
# 1. Evidence-Conditioned Contrastive Decoding (ECD)
# =========================================================
class ECDLogitsProcessor(LogitsProcessor):
    """
    Evidence-Conditioned Contrastive Decoding (Section 3.4).

    Computes steered logits: (1 + alpha) * l_ev - alpha * l_no (Eq. 7)

    ECD is applied uniformly to all tokens including EOS.
    """
    def __init__(
        self,
        alpha: float,
        eos_token_id: Optional[int] = None,
    ):
        self.alpha = alpha
        self.eos_token_id = eos_token_id  # kept for interface compatibility

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # Expect batch size 2: [evidence, no_evidence]
        if scores.shape[0] != 2:
            return scores

        logits_ev = scores[0].float()
        logits_no = scores[1].float()

        # ECD formula: (1 + α) * l_ev - α * l_no
        ecd_logits = (1 + self.alpha) * logits_ev - self.alpha * logits_no

        return torch.stack([ecd_logits, ecd_logits]).to(scores.dtype)


# =========================================================
# 2. LLM Client
# =========================================================
class LLMClient:
    def __init__(self, model_name: str):
        self.model_name = model_name
        model_path = MODEL_NAME_TO_PATH.get(model_name, model_name)

        print(f"Loading Model: {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Standard generation (no-evidence prompt)."""
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        do_sample = temperature > 0.0

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=0.9 if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(
            outputs[0][input_ids.shape[-1]:],
            skip_special_tokens=True,
        ).strip()

    def generate_ecd(
        self,
        prompt_evidence: Union[str, List[Dict[str, str]]],
        prompt_no_evidence: Union[str, List[Dict[str, str]]],
        alpha: float = 0.3,
        max_tokens: int = 64,
        temperature: float = 0.0,
    ) -> str:
        """Evidence-Conditioned Contrastive Decoding (Section 3.4)."""
        if isinstance(prompt_evidence, str):
            msgs_ev = [{"role": "user", "content": prompt_evidence}]
        else:
            msgs_ev = prompt_evidence

        if isinstance(prompt_no_evidence, str):
            msgs_no = [{"role": "user", "content": prompt_no_evidence}]
        else:
            msgs_no = prompt_no_evidence

        text_ev = self.tokenizer.apply_chat_template(
            msgs_ev,
            tokenize=False,
            add_generation_prompt=True,
        )
        text_no = self.tokenizer.apply_chat_template(
            msgs_no,
            tokenize=False,
            add_generation_prompt=True,
        )

        batch_texts = [text_ev, text_no]
        inputs = self.tokenizer(
            batch_texts,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        logits_processor = LogitsProcessorList(
            [
                ECDLogitsProcessor(
                    alpha=alpha,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            ]
        )

        do_sample = temperature > 0.0

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=0.9 if do_sample else None,
                logits_processor=logits_processor,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# =========================================================
# 3. Parsing Helpers
# =========================================================
def parse_hop_plan(
    text: str,
    max_allowed: int,
    original_question: str = "",
) -> Tuple[int, List[str]]:
    """Parse entity-referential sub-question instructions from hop planner output."""
    m = re.search(r"HOP_COUNT\s*:\s*(\d+)", text)
    hop_count = int(m.group(1)) if m else 2
    hop_count = max(1, min(hop_count, max_allowed))

    descs: Dict[int, str] = {}
    for line in text.splitlines():
        m_hop = re.match(
            r"^(?:-|\*)?\s*Hop\s+(\d+)\s*:\s*(.+)",
            line.strip(),
            re.IGNORECASE,
        )
        if m_hop:
            descs[int(m_hop.group(1))] = m_hop.group(2).strip()

    return hop_count, [descs.get(t, f"Step {t}") for t in range(1, hop_count + 1)]


def parse_initial_entity(text: str) -> str:
    """Extract initial topic entity (o_0) from entity extraction output."""
    m = re.search(r"^\s*Q_ENTITY\s*:\s*(.*)\s*$", text, flags=re.MULTILINE)
    if not m:
        return ""
    ent = m.group(1).strip()
    ent = ent.strip().strip("\"").strip("'").strip()
    return ent


# =========================================================
# 4. Reference-Resolved Sub-question Builder
# =========================================================
_ANSWER_PAT = re.compile(r"\bANSWER_(\d+)\b")


def materialize_answer_placeholders(text: str, prev_hop_answers: Optional[List[str]]) -> str:
    """Replace ANSWER_k placeholders with actual hop answers."""
    if not prev_hop_answers:
        return text

    def _repl(m: re.Match) -> str:
        k = int(m.group(1))
        idx = k - 1
        if 0 <= idx < len(prev_hop_answers):
            return prev_hop_answers[idx].strip()
        return m.group(0)

    return _ANSWER_PAT.sub(_repl, text)


def build_resolved_subquestion(
    question: str,
    hop_idx: int,
    hop_desc: str,
    prev_hop_answers: Optional[List[str]] = None,
) -> str:
    """
    Build reference-resolved sub-question (sq_t') for retrieval.

    For t > 1: sq_t' = sq_t + Ctx(o_{t-1})
    """
    desc_clean = re.sub(r"[.,?!\s]+$", "", hop_desc.strip())
    desc_clean = materialize_answer_placeholders(desc_clean, prev_hop_answers)

    lower_desc = desc_clean.lower()
    is_dummy = len(lower_desc.split()) <= 2 and ("step" in lower_desc or "hop" in lower_desc)

    if is_dummy:
        if hop_idx == 1:
            return question.strip()
        if prev_hop_answers:
            last_ans = prev_hop_answers[-1].strip()
            last_ans = re.sub(r"[.,?!\s]+$", "", last_ans)
            return f"{question.strip()} (Context: {last_ans})"

    if hop_idx == 1:
        return desc_clean

    # For hop >= 2, append context suffix if not already present
    if prev_hop_answers:
        last_ans = prev_hop_answers[-1].strip()
        last_ans = re.sub(r"[.,?!\s]+$", "", last_ans)

        if last_ans and (last_ans.lower() in desc_clean.lower()):
            return desc_clean

        return f"{desc_clean} (Context: {last_ans})"

    return desc_clean


# =========================================================
# 5. Answer Extractor
# =========================================================
def extract_answer(text: str) -> str:
    """Extract single-entity answer from QA output."""
    if "A:" in text:
        text = text.split("A:", 1)[1]
    elif "Answer:" in text:
        text = text.split("Answer:", 1)[1]
    text = text.strip()
    if "\n" in text:
        text = text.split("\n")[0].strip()
    if text.endswith("."):
        text = text[:-1]
    return text
