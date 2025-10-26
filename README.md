# llm-collusion
Experiments and analyses on LLM collusion and identifiability in code evaluation.

## Self-Recognition Judge Fine-tuning

This repo can fine-tune a small open-source model to identify which of two code snippets was authored by a target model (self-recognition). The pipeline:

1) Generate code with two models for MBPP

```bash
python generate_pairs.py run --dataset mbpp --start-index 0 --end-index 200 \
  --config configs/config.yaml
```

This writes a JSONL under `data/` containing records with fields like `benchmark`, `task_id`, `prompt`, `model_name`, `generated_code`.

2) Fine-tune the judge on labeled pairs (Code1 vs Code2) for a target model

```bash
python train_self_recognition.py run \
  --input data/mbpp-YYYYMMDD-HHMMSS.jsonl \
  --dataset mbpp \
  --target-model openai/gpt-5 \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output-dir outputs/self_judge-gpt5 --epochs 1 --batch-size 2
```

Notes
- The script constructs pairs per task, randomizes Code1/Code2 order, and sets the label to the index of the target model.
- Training is standard causal LM fine-tuning: the prompt is the self-recognition instruction plus Code1/Code2; the label is the short answer token "1" or "2" with loss masked over the prompt tokens.
- Change `--target-model` to train a judge specialized to a different source model.
