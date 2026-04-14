# main.py
"""
EAIR: Evidence-Aware Iterative Reasoning for Multi-hop Question Answering

This script implements the full EAIR pipeline:
1. Entity-referential query decomposition (Decompose)
2. Entity-aware retrieval (Retrieve)
3. Evidence-conditioned contrastive decoding (Decode)
4. Reflection-based knowledge routing (Route)
"""
import argparse
from typing import Any, Dict, List
import json
import os
import copy

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer as HFTokenizer, set_seed

from utils import (
    load_json,
    get_sent_embeddings,
    retrieve_facts,
    load_mquake,
    normalize_answer,
    format_evidence_block,
    entity_in_text,
)
from hop import (
    LLMClient,
    build_resolved_subquestion,
    extract_answer,
    parse_hop_plan,
    parse_initial_entity,
)


def is_refusal_response(text: str) -> bool:
    """Detect reflection-based refusal responses (Section 3.5)."""
    word_count = len(text.split())
    low = text.lower()
    return (
        ("sorry" in low) or ("can't" in low) or ("cannot" in low) or ("can not" in low)
        or ("unable to" in low) or ("not able to" in low) or ("unfortunately" in low)
        or ("not enough" in low) or ("insufficient" in low) or ("no information" in low)
        or ("not available" in low) or ("none" in low) or ("n/a" in low) or ("unknown" in low)
        or word_count > 10
    )


def main():
    parser = argparse.ArgumentParser(description="EAIR: Evidence-Aware Iterative Reasoning")
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()

    cfg = load_json(args.config)

    # Configuration
    seed = cfg.get("seed")
    if seed is not None:
        set_seed(seed)

    dataset_name = cfg.get("dataset_name", "MQuAKE-CF-3k-v2")
    dataset_path = f"datasets/{dataset_name}.json"
    model_name = cfg.get("model_name", "llama3.1-8b")
    alpha = float(cfg.get("alpha", 0.3))
    k_evidence = int(cfg.get("k_evidence", 8))

    # Load prompts
    hop_plan_base = load_json("prompt/hop_plan.json")
    entity_extract_base = load_json("prompt/q_entity_extract.json")
    qa_prompt_spec = load_json("prompt/qa_prompt.json")

    qa_examples = qa_prompt_spec["examples"]
    qa_examples_block = "\n\n".join([f"Q: {ex['Q']}\nA: {ex['A']}" for ex in qa_examples])

    # Load dataset
    data = load_mquake(dataset_path)
    num_cases = len(data)

    # Output path
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name
    output_path = f"./output/EAIR_{model_short}_{dataset_name}"

    print(f"\n{'='*60}")
    print("EAIR: Evidence-Aware Iterative Reasoning")
    print(f"{'='*60}")
    print(f"  Model   : {model_name}")
    print(f"  Dataset : {dataset_name} ({num_cases} cases)")
    print(f"  Alpha   : {alpha}")
    print(f"  K_ev    : {k_evidence}")
    print(f"{'='*60}\n")

    # Initialize LLM and retriever
    llm = LLMClient(model_name=model_name)
    contriever = AutoModel.from_pretrained("facebook/contriever-msmarco").cuda()
    ret_tok = HFTokenizer.from_pretrained("facebook/contriever-msmarco")

    # Build global edit set (E)
    all_edits: List[str] = []
    for c in data:
        for r in c.get("requested_rewrite", []):
            subj = r["subject"]
            prompt_filled = r["prompt"].format(subj)

            if "target_new_str" in r:
                target_val = r["target_new_str"]
            elif "target_new" in r and isinstance(r["target_new"], dict):
                target_val = r["target_new"]["str"]
            else:
                target_val = r["target_new"]

            fact_str = f"{prompt_filled} {target_val}"
            if fact_str not in all_edits:
                all_edits.append(fact_str)

    edit_embs = get_sent_embeddings(all_edits, contriever, ret_tok)
    print(f"Built edit set: {len(all_edits)} facts.\n")

    # Metrics
    results: List[Dict[str, Any]] = []
    tot_cases = 0
    tot_questions = 0
    correct_questions = 0
    cases_lenient = 0  # At least one question correct
    cases_strict = 0   # All questions correct

    pbar = tqdm(total=num_cases, desc="EAIR", unit="case")

    for case in data:
        case_id = case.get("case_id", tot_cases)
        tot_cases += 1

        questions = case.get("questions", [])
        gold_answers = [case["new_answer"].lower()] + [a.lower() for a in case.get("new_answer_alias", [])]

        per_question_results: List[Dict[str, Any]] = []
        case_correct_cnt = 0

        for qi, q in enumerate(questions):
            # === Step 1: Entity-referential query decomposition (Section 3.1) ===
            # Extract initial entity o_0
            entity_msgs = copy.deepcopy(entity_extract_base)
            entity_msgs.append({"role": "user", "content": f"Question: {q}"})
            entity_raw = llm.generate(entity_msgs, max_tokens=32, temperature=0.0)
            initial_entity = parse_initial_entity(entity_raw)

            # Generate hop plan
            plan_msgs = copy.deepcopy(hop_plan_base)
            plan_msgs.append({"role": "user", "content": f"Question: {q}"})
            plan_raw = llm.generate(plan_msgs, max_tokens=256, temperature=0.0)
            hop_count, sub_questions = parse_hop_plan(plan_raw, max_allowed=10, original_question=q)

            result_data: Dict[str, Any] = {
                "question": q,
                "initial_entity": initial_entity,
                "hop_plan": sub_questions,
                "hop_results": [],
                "final_answer": None,
            }

            try:
                prev_answers: List[str] = []
                final_answer: str | None = None

                for t in range(1, hop_count + 1):
                    sq_t = sub_questions[t - 1]

                    # Build reference-resolved sub-question sq_t'
                    sq_t_resolved = build_resolved_subquestion(
                        question=q,
                        hop_idx=t,
                        hop_desc=sq_t,
                        prev_hop_answers=prev_answers,
                    )

                    # === Step 2: Entity-aware retrieval (Section 3.2) ===
                    # Determine topic entity for this hop
                    if t == 1:
                        topic_entity = initial_entity
                    else:
                        topic_entity = prev_answers[-1] if prev_answers else ""

                    # Entity exact matching
                    candidate_ids: List[int] | None = None
                    entity_matched = False

                    if topic_entity:
                        matched_ids = [
                            i for i, edit in enumerate(all_edits)
                            if entity_in_text(topic_entity, edit)
                        ]
                        if matched_ids:
                            candidate_ids = matched_ids
                            entity_matched = True

                    # Skip retrieval if hop 1 entity match fails
                    skip_retrieval = False
                    if t == 1 and initial_entity and not entity_matched:
                        skip_retrieval = True

                    # Dense retrieval
                    if skip_retrieval:
                        retrieved_ids, _ = [], []
                    else:
                        retrieved_ids, _ = retrieve_facts(
                            sq_t_resolved,
                            fact_embs=edit_embs,
                            contriever=contriever,
                            tok=ret_tok,
                            k=k_evidence,
                            candidate_ids=candidate_ids,
                        )

                    # Prepare prompts
                    system_no = qa_prompt_spec["template_no"].format(
                        examples_block=qa_examples_block,
                        question=sq_t_resolved,
                    )
                    msgs_no = [{"role": "system", "content": system_no}]

                    # === Step 3: Evidence-conditioned contrastive decoding (Section 3.4) ===
                    used_evidence = False
                    retrieved_edits: List[str] = []

                    if not retrieved_ids:
                        # No evidence: use parametric knowledge
                        ans_raw = llm.generate(msgs_no, max_tokens=64, temperature=0.0)
                    else:
                        used_evidence = True
                        retrieved_edits = [all_edits[i] for i in retrieved_ids[:k_evidence]]

                        evidence_block = format_evidence_block(retrieved_edits)
                        system_ev = qa_prompt_spec["template_ctx"].format(
                            evidence_block=evidence_block,
                            examples_block=qa_examples_block,
                            question=sq_t_resolved,
                        )
                        msgs_ev = [{"role": "system", "content": system_ev}]

                        # ECD generation
                        ans_raw = llm.generate_ecd(
                            msgs_ev,
                            msgs_no,
                            alpha=alpha,
                            max_tokens=64,
                            temperature=0.0,
                        )

                    answer = extract_answer(ans_raw)

                    # === Step 4: Reflection-based knowledge routing (Section 3.5) ===
                    if used_evidence and is_refusal_response(answer):
                        # Route to no-evidence generation
                        ans_raw = llm.generate(msgs_no, max_tokens=64, temperature=0.0)
                        answer = extract_answer(ans_raw)
                        used_evidence = False
                        retrieved_edits = []

                    prev_answers.append(answer)

                    result_data["hop_results"].append({
                        "hop": t,
                        "sub_question": sq_t_resolved,
                        "retrieved_edits": retrieved_edits,
                        "answer": answer,
                        "used_parametric": not used_evidence,
                    })

                    if t == hop_count:
                        final_answer = answer

                if not final_answer and result_data["hop_results"]:
                    final_answer = result_data["hop_results"][-1]["answer"]

                result_data["final_answer"] = final_answer

                # Evaluate
                pred_norm = normalize_answer(final_answer or "")
                is_correct = any(normalize_answer(ga) in pred_norm for ga in gold_answers)
                result_data["is_correct"] = is_correct

                if is_correct:
                    correct_questions += 1
                    case_correct_cnt += 1

                per_question_results.append(result_data)
                tot_questions += 1

            except Exception as e:
                print(f"Error in Case {case_id}: {e}")

        # Update case-level metrics
        if case_correct_cnt > 0:
            cases_lenient += 1
        if case_correct_cnt == len(questions) and len(questions) > 0:
            cases_strict += 1

        results.append({"case_id": case_id, "results": per_question_results})

        # Update progress bar
        q_acc = (correct_questions / tot_questions * 100) if tot_questions else 0
        lenient_acc = (cases_lenient / tot_cases * 100) if tot_cases else 0
        strict_acc = (cases_strict / tot_cases * 100) if tot_cases else 0

        pbar.set_postfix({
            "Q": f"{q_acc:.2f}%",
            "Lenient": f"{lenient_acc:.2f}%",
            "Strict": f"{strict_acc:.2f}%"
        })
        pbar.update(1)

    pbar.close()

    # Save results
    os.makedirs(output_path, exist_ok=True)
    with open(f"{output_path}/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Final report
    print(f"\n{'='*60}")
    print("EAIR Results")
    print(f"{'='*60}")
    print(f"  Total Cases     : {tot_cases}")
    print(f"  Total Questions : {tot_questions}")
    print(f"{'-'*60}")
    if tot_questions:
        print(f"  Question Accuracy      : {(correct_questions / tot_questions * 100):.2f}%")
    if tot_cases:
        print(f"  Lenient Case Accuracy  : {(cases_lenient / tot_cases * 100):.2f}%")
        print(f"  Strict Case Accuracy   : {(cases_strict / tot_cases * 100):.2f}%")
    print(f"{'='*60}")
    print(f"Results saved to: {output_path}/results.json\n")


if __name__ == "__main__":
    main()
