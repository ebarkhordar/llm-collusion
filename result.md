# Complete Experiment Results (Fixed Prompt Only)

> Random baseline = 50%. All on MBPP-sanitized (257 tasks) unless noted.
> Full attribution results excluded entirely — all used old prompt.

---

## 1. Self-Recognition (Pair) — "Is this YOUR code?"

| Model | MBPP |
|-------|:----:|
| **GPT-5** | **75.9%** |
| Gemini-2.5-Flash | 52.9% |
| Grok-4-Fast | 44.7% |
| Claude-Haiku | 35.8% |
| DeepSeek-V3 | 33.9% |

## 2. Self-Recognition (Single/IPP) — "Is this code mine? Yes/No"

| Model | MBPP | HumanEval | DS-1000 |
|-------|:----:|:---------:|:-------:|
| **Claude-Haiku** | **65.8%** | **66.9%** | **64.2%** |
| GPT-5 | 59.9% | 60.2% | 62.4% |
| DeepSeek-V3 | 59.0% | 59.1% | 55.3% |
| Grok-4-Fast | 43.8% | 39.6% | 48.0% |
| Gemini-2.5-Flash | 42.4% | 38.4% | 48.0% |

---

## 3. Target Identification (Fixed Prompt)

### Claude-Haiku vs DeepSeek-V3 — EASY (heuristic=89%)

| Judge | Target | Acc |
|-------|--------|:---:|
| **Codex** | **DeepSeek** | **91.8%** |
| Codex | Claude | 89.9% |
| Grok-Code | Claude | 80.9% |
| Grok-Code | DeepSeek | 69.3% |

### Gemini-2.5-Flash vs GPT-5 — MEDIUM (heuristic=57%)

| Judge | Target | Acc |
|-------|--------|:---:|
| **Codex** | **Gemini** | **77.8%** ✅ +21pp |
| Grok-Code | Gemini | 44.0% |

### Claude-Haiku vs GPT-5 — MEDIUM (heuristic=63%)

| Judge | Target | Acc |
|-------|--------|:---:|
| **Codex** | **Claude** | **69.3%** ✅ +6pp |

### Codestral vs GPT-5.3-Codex — MEDIUM (heuristic=57%)

| Judge | Target | Acc |
|-------|--------|:---:|
| GPT-5 | Codestral | 56.4% |

### Codestral vs Grok-4-Fast — MEDIUM (heuristic=61%)

| Judge | Target | Acc |
|-------|--------|:---:|
| Codex | Codestral | 49.0% |

### DeepSeek-V3.2 vs MiMo — MEDIUM (heuristic=56%)

| Judge | Target | Acc |
|-------|--------|:---:|
| GPT-5 | DeepSeek-v3.2 | 51.8% |

### Qwen3-Coder vs MiMo — HARD (heuristic=53%)

| Judge | Target | Acc |
|-------|--------|:---:|
| GPT-5 | Qwen | 40.5% |
| GPT-5 | MiMo | 41.6% |
| Codex | Qwen | 41.2% |
| Codex | MiMo | 42.8% |
| Grok-Code | Qwen | 39.7% |

### Claude-Opus vs Gemini-3.1-Flash-Lite — EASY (heuristic=98%)

| Judge | Target | Acc |
|-------|--------|:---:|
| Codex | Claude-Opus | 7.8% ❌ |
| Codex | Gemini-3.1 | 31.1% ❌ |

---

## 4. Code Generation Coverage

| Dataset | Models | Tasks |
|---------|:------:|:-----:|
| **MBPP** | **13** | **257** |
| HumanEval | 5 | 164 |
| DS-1000 | 5 | 150 |

MBPP models: claude-haiku, claude-opus, deepseek-v3, deepseek-v3.2, gemini-2.5-flash, gemini-3.1-flash-lite, llama-4-maverick, codestral, gpt-5.3-codex, gpt-5, qwen3-coder, grok-4-fast, mimo-v2-pro

---

## Summary

| Finding | Evidence |
|---------|----------|
| **Only 1 strong cross-heuristic result** | Codex→Gemini vs GPT-5 = 77.8% (heuristic 57%) |
| **Easy pairs ≈ heuristic** | Claude vs DeepSeek: judge 92% ≈ heuristic 89% |
| **Same-ecosystem = chance** | Qwen vs MiMo ≈ 40% across all judges |
| **Judge bias** | Codex→Opus=7.8%, severe model preference |
| **Best judge** | GPT-5.3-Codex (most consistent) |
| **Self-rec leader** | GPT-5 pair=75.9%, Claude IPP=65.8% |
