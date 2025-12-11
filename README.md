# llm-collusion
Experiments and analyses on LLM collusion and identifiability in code evaluation.

## Overview

This project investigates whether LLMs can recognize code written by themselves or other AI models. It implements three attribution tasks:

| Task | Description |
|------|-------------|
| **Self-Recognition** | Can an LLM identify which code snippet it wrote? |
| **Target Identification** | Can an LLM identify code written by a specific named model? |
| **Full Attribution** | Can an LLM correctly classify both code snippets to their respective authors? |

## Supported Datasets

| Dataset | Size | Description | Splits |
|---------|------|-------------|--------|
| **MBPP** | 974 tasks | Mostly Basic Python Problems (sanitized) | train, test, validation, prompt |
| **HumanEval** | 164 tasks | OpenAI's hand-written Python evaluation set | test only |
| **DS-1000** | 1000 tasks | Data science problems (NumPy, Pandas, Matplotlib, etc.) | test only |

## Quick Start

### 1. Generate Code with Multiple Models

```bash
# MBPP dataset
python generate_pairs.py run --dataset mbpp --split test --start-index 0 --end-index -1

# HumanEval dataset
python generate_pairs.py run --dataset humaneval --split test --start-index 0 --end-index -1

# DS-1000 dataset
python generate_pairs.py run --dataset ds1000 --split test --start-index 0 --end-index -1
```

This generates code from all models configured in `configs/config.yaml` and saves to:
- `data/code_generation/mbpp-sanitized/{split}/`
- `data/code_generation/humaneval/{split}/`
- `data/code_generation/ds1000/{split}/`

### 2. Run Attribution Experiments

#### Self-Recognition
```bash
python self_recognition.py run --dataset-folder mbpp-sanitized --split test
python self_recognition.py run --dataset-folder humaneval --split test
python self_recognition.py run --dataset-folder ds1000 --split test
```

#### Target Identification
```bash
python target_identification.py run \
  --dataset-folder mbpp-sanitized --split test \
  --model1 anthropic/claude-haiku-4.5 \
  --model2 deepseek/deepseek-chat-v3-0324 \
  --judge openai/gpt-5
```

#### Full Attribution
```bash
python full_attribution.py run \
  --dataset-folder mbpp-sanitized --split test \
  --model1 anthropic/claude-haiku-4.5 \
  --model2 deepseek/deepseek-chat-v3-0324 \
  --judge openai/gpt-5
```

## Project Structure

```
llm-collusion/
├── configs/
│   └── config.yaml          # Dataset and model configurations
├── data/
│   ├── code_generation/     # Generated code by model and dataset
│   │   ├── mbpp-sanitized/
│   │   ├── humaneval/
│   │   └── ds1000/
│   ├── self_recognition/    # Self-recognition results
│   ├── target_identification/
│   └── full_attribution/    # Full attribution results
├── notebooks/
│   ├── evaluate_full_attribution.ipynb
│   └── evaluate_all_attribution_tasks.ipynb
├── prompts/
│   ├── generation.md
│   └── model_attribution/
│       ├── self_recognition.md
│       ├── target_identification.md
│       └── full_attribution.md
├── src/
│   ├── common/types.py      # Data types
│   ├── datasets/
│   │   ├── mbpp.py          # MBPP loader
│   │   ├── humaneval.py     # HumanEval loader
│   │   └── ds1000.py        # DS-1000 loader
│   └── lib/                 # Utilities
├── generate_pairs.py        # Code generation script
├── self_recognition.py      # Self-recognition evaluation
├── target_identification.py # Target identification evaluation
└── full_attribution.py      # Full attribution evaluation
```

## Configuration

Edit `configs/config.yaml` to configure:
- Models to use for each dataset
- API concurrency settings
- Data paths

```yaml
datasets:
  mbpp:
    models:
      - openai/gpt-5
      - anthropic/claude-haiku-4.5
      - google/gemini-2.5-flash
      - x-ai/grok-4-fast
      - deepseek/deepseek-chat-v3-0324

  humaneval:
    models:
      - openai/gpt-5
      - anthropic/claude-haiku-4.5
      # ... same models

  ds1000:
    models:
      - openai/gpt-5
      - anthropic/claude-haiku-4.5
      # ... same models
```

## Environment Setup

1. Install dependencies:
```bash
poetry install
```

2. Set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY=your_key_here
```

## Self-Recognition Judge Fine-tuning

This repo can fine-tune a small open-source model to identify which of two code snippets was authored by a target model (self-recognition). The pipeline:

1) Generate code with two models for a dataset

```bash
python generate_pairs.py run --dataset mbpp --start-index 0 --end-index 200
```

2) Fine-tune the judge on labeled pairs (Code1 vs Code2) for a target model

```bash
python train_self_recognition.py run \
  --input data/mbpp-YYYYMMDD-HHMMSS.jsonl \
  --dataset mbpp \
  --target-model openai/gpt-5 \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output-dir outputs/self_judge-gpt5 --epochs 1 --batch-size 2
```

Notes:
- The script constructs pairs per task, randomizes Code1/Code2 order, and sets the label to the index of the target model.
- Training is standard causal LM fine-tuning: the prompt is the self-recognition instruction plus Code1/Code2; the label is the short answer token "1" or "2" with loss masked over the prompt tokens.
- Change `--target-model` to train a judge specialized to a different source model.
