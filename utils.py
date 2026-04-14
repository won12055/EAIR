# utils.py
import json
import re
import string
from typing import Any, Dict, List

import torch
from tqdm import tqdm


# ===================== JSON I/O ===================== #
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_mquake(path: str) -> List[Dict[str, Any]]:
    return load_json(path)


# ===================== Evidence Formatting ===================== #
def format_evidence_block(evidences: List[str]) -> str:
    return "\n".join([f"- {s}" for s in evidences])


# ===================== Answer Normalization ===================== #
def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# ===================== Lexical Helpers ===================== #
def normalize_entity_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[.,?!\"“”'’\s]+$", "", s)
    s = re.sub(r"(?:'s|’s)\b", "", s)
    return s.lower()


def entity_in_text(entity: str, text: str) -> bool:
    entity_norm = normalize_entity_text(entity)
    if not entity_norm:
        return False
    text_norm = text.lower()
    pattern = r"\b" + re.escape(entity_norm) + r"\b"
    return re.search(pattern, text_norm) is not None


# ===================== Retrieval Utilities ===================== #
def mean_pooling(
    token_embeddings: "torch.Tensor",
    mask: "torch.Tensor",
) -> "torch.Tensor":
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def get_sent_embeddings(
    sents: List[str],
    contriever,
    tok,
    BSZ: int = 32,
) -> "torch.Tensor":
    all_embs: List["torch.Tensor"] = []
    for i in tqdm(range(0, len(sents), BSZ), desc="Building sentence embeddings"):
        sent_batch = sents[i: i + BSZ]

        inputs = tok(
            sent_batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs["attention_mask"])

        all_embs.append(embeddings.cpu())

    all_embs_tensor = torch.vstack(all_embs)
    return all_embs_tensor


def retrieve_facts(
    query: str,
    fact_embs: "torch.Tensor",
    contriever,
    tok,
    k: int = 5,
    candidate_ids: List[int] | None = None,
):
    inputs = tok(
        [query],
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = contriever(**inputs)
        query_emb = mean_pooling(
            outputs[0],
            inputs["attention_mask"],
        ).cpu()

    if candidate_ids is not None:
        sub_embs = fact_embs[candidate_ids]
    else:
        sub_embs = fact_embs
        candidate_ids = list(range(fact_embs.size(0)))

    sim = (query_emb @ sub_embs.T)[0]

    if sim.numel() == 0:
        return [], []

    k_eff = min(k, sim.size(0))
    knn = sim.topk(k_eff, largest=True)

    local_indices = knn.indices.tolist()
    scores = knn.values.tolist()

    global_indices = [candidate_ids[i] for i in local_indices]
    return global_indices, scores
