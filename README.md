# EAIR: Entity-aware Inference-Time Knowledge Routing for Multi-Hop Knowledge Editing

This repository contains the official implementation of **EAIR**, a framework for multi-hop question answering with knowledge editing.

## Method Overview

EAIR addresses hop-wise error propagation in multi-hop QA through explicit knowledge routing between parametric and retrieved knowledge:

1. **Entity-referential Query Decomposition** (`Decompose`): Decomposes multi-hop questions into sequential sub-questions with entity references
2. **Entity-aware Retrieval** (`Retrieve`): Retrieves relevant edits using entity exact matching + dense retrieval
3. **Evidence-Conditioned Contrastive Decoding** (`Decode`): Applies contrastive decoding: `(1 + α) * l_ev - α * l_no`
4. **Reflection-based Knowledge Routing** (`Route`): Falls back to parametric knowledge when evidence is irrelevant

## Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

## Quick Start

```bash
# Run with default config
uv run python main.py

# Run with custom config
uv run python main.py --config config.json
```

## Configuration

Edit `config.json` to customize experiments:

```json
{
  "model_name": "Qwen/Qwen2.5-7B-Instruct",
  "dataset_name": "MQuAKE-CF-3k-v2",
  "alpha": 0.3,
  "k_evidence": 8,
  "seed": 100
}
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_name` | HuggingFace model path or alias | `Qwen/Qwen2.5-7B-Instruct` |
| `dataset_name` | Dataset name (without `.json`) | `MQuAKE-CF-3k-v2` |
| `alpha` | ECD contrast strength (α in Eq. 7) | `0.3` |
| `k_evidence` | Number of edits to retrieve per hop | `8` |
| `seed` | Random seed for reproducibility | `100` |

### Supported Models

| Alias | HuggingFace Path |
|-------|------------------|
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` |
| `qwen2.5-14b` | `Qwen/Qwen2.5-14B-Instruct` |

### Supported Datasets

- `MQuAKE-CF-3k-v2` (main benchmark)
- `MQuAKE-CF3k` (Remastered)
- `MQuAKE-T`
- `MQuAKE-T-RE` (Remastered)

## Output

Results are saved to `./output/EAIR_{model}_{dataset}/results.json`

### Metrics

- **Question Accuracy**: Per-question accuracy
- **Lenient Case Accuracy**: At least one question correct per case
- **Strict Case Accuracy**: All questions correct per case

## Project Structure

```
EAIR-ACL2026/
├── main.py              # Main entry point
├── hop.py               # LLMClient, ECDLogitsProcessor, parsing helpers
├── utils.py             # Embeddings, retrieval, normalization utilities
├── config.json          # Experiment configuration
├── pyproject.toml       # Dependencies
├── uv.lock              # Lockfile for reproducible deps
├── prompt/
│   ├── hop_plan.json       # Hop decomposition prompt
│   ├── qa_prompt.json      # QA prompt templates
│   └── q_entity_extract.json  # Entity extraction prompt
└── datasets/
    ├── MQuAKE-CF-3k-v2.json
    ├── MQuAKE-CF3k.json
    ├── MQuAKE-T.json
    └── MQuAKE-T-RE.json
```

## Experimental Hardware

Our experiments were conducted on the following hardware:

- **7B/8B models** (Qwen2.5-7B, Llama3.1-8B): 1x NVIDIA A6000 48GB
- **14B models** (Qwen2.5-14B): 1x NVIDIA H200
