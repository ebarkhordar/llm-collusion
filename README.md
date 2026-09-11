# llm-collusion

Code and data for **"Style, Not Self: Surface Cues Explain Zero-Shot Code Attribution by Large Language Models."**

The paper asks whether an LLM can recognise code it wrote itself. Across three attribution
paradigms the answer is essentially no: what looks like self-recognition is driven by surface
style cues and response bias rather than any sense of authorship. This repository contains the
generation and evaluation pipeline, the raw judge outputs, and the scripts that rebuild every
table and figure in the paper from that raw data.

## Tasks

| Task | Question | Script |
|------|----------|--------|
| **Self-recognition (pairwise, 1a)** | Shown two snippets, can a model pick its own? | `experiments/self_recognition.py` |
| **Self-recognition (individual, 1b)** | Shown one snippet, can a model say "I wrote this"? | `experiments/self_recognition_single.py` |
| **Target identification (2)** | Can a model spot code by a specific *named* model? | `experiments/target_identification.py` |
| **Self-preference (3)** | Blind on authorship, does a model prefer its own code? | `experiments/self_preference.py` |
| **Full attribution** | Can a model assign both snippets to their authors? | `experiments/full_attribution.py` |

Full attribution is kept here for completeness but is **excluded from the paper**: every run used
a pre-fix prompt that aligned model-name order with solution order. See `result.md`.

## Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| **MBPP** (sanitized) | 974 tasks | Mostly Basic Python Problems — the primary dataset |
| **HumanEval** | 164 tasks | Hand-written Python evaluation set |
| **DS-1000** | 1000 tasks | Data-science problems (NumPy, Pandas, Matplotlib) |

## Setup

```bash
poetry install
export OPENROUTER_API_KEY=your_key_here   # or put it in .env
```

All model calls go through OpenRouter. Every script under `experiments/` resolves its paths from
the repository root, so it can be run from any working directory.

## Reproducing the paper

The raw judge outputs are committed under `data/`, so the tables and figures rebuild offline with
no API calls:

```bash
# LaTeX tables + summary.json, and the IPP figure -> build/
python analysis/make_tables.py
python analysis/make_figures.py

# or write straight into the paper repo
PAPER_DIR=~/LatexProjects/llm-collusion-paper python analysis/make_tables.py
```

`result.md` is the Markdown summary printed by `make_tables.py`, with a hand-written preamble
recording which runs are included and why.

## Running the experiments

### 1. Generate code

```bash
python experiments/generate_pairs.py --dataset mbpp --split test --start-index 0 --end-index -1
```

Writes to `data/code_generation/<dataset>/<split>/<model>.jsonl`, using the models in
`configs/config.yaml`.

### 2. Attribution tasks

```bash
python experiments/self_recognition.py --dataset-folder mbpp-sanitized --split test

python experiments/self_recognition_single.py --dataset-folder mbpp-sanitized --split test

python experiments/target_identification.py \
  --dataset-folder mbpp-sanitized --split test \
  --model1 anthropic/claude-haiku-4.5 \
  --model2 deepseek/deepseek-chat-v3-0324 \
  --judge openai/gpt-5
```

`self_recognition.py` writes `<evaluator>.jsonl` and **overwrites** it — never run two batches
concurrently against the same output folder.

### 3. Supporting scripts

```bash
# Non-LLM heuristic lower bounds (comment ratio, identifier length, ...)
python experiments/baselines.py --dataset-folder mbpp-sanitized --split test

# Redaction & Paraphrasing: strip comments/docstrings, rename locals, normalize formatting
python experiments/obfuscate.py run --dataset-folder mbpp-sanitized --split test

# pass@1 via unit tests
python experiments/run_tests.py --input data/code_generation_normalized/mbpp-sanitized/test

# carry the normalized-code judgments over to the corrected normalizer (re-judges changed items only)
python experiments/rejudge_normalized.py --dry-run
```

The DS-1000 results in `data/tests/ds1000/` were produced with pandas 2.3.3. The lockfile now pins
pandas 3.0.1, under which about ten Pandas problems per model fail for API reasons, so pin pandas
2.3.3 to reproduce them.

## Layout

```
llm-collusion/
├── configs/config.yaml        # models, concurrency, data paths
├── prompts/                   # generation + attribution prompt templates
├── src/
│   ├── common/types.py        # shared record types
│   ├── datasets/              # MBPP / HumanEval / DS-1000 loaders
│   ├── generation/            # per-dataset generation + code extraction
│   └── lib/                   # OpenRouter client, config, JSONL, prompt rendering
├── experiments/               # the five tasks + baselines, obfuscation, unit tests
│   └── _paths.py              # repo-root anchoring shared by the entrypoints
├── analysis/
│   ├── make_tables.py         # rebuilds every paper table from data/
│   └── make_figures.py        # rebuilds the IPP figure
├── data/                      # generated code and raw judge outputs (committed)
└── result.md                  # Markdown summary of the current numbers
```

### About `data/`

`code_generation/` holds the original model output. GPT-5 returned no code on 45 tasks (it spent its
2,000-token budget on reasoning); the analysis counts these as test failures and drops them from
every attribution task. `code_generation_normalized/` is the output of the corrected normalizer, and
the judgments on it live in `*/mbpp-sanitized-normalized/`. `code_generation_obfuscated/` and
`*/mbpp-sanitized-obfuscated/` are the September reruns on an earlier normalizer version that left
lambda parameters and nested function names unrenamed; `rejudge_normalized.py` re-judged the items
whose code changed. A few directories named by timestamp (e.g. `full_attribution/20260319-152100/`) are early
exploratory runs, superseded by the dataset-named directories and not read by `analysis/`.

## License

MIT — see [LICENSE](LICENSE).
