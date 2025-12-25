# Test Statistics Report

## Overview

This report summarizes the performance of different LLM models on code generation tests from the `data/tests/` directory. A total of **2,855 test results** were analyzed across **3 benchmarks** and **5 models**.

### Models Evaluated
- GPT-5 (OpenAI)
- Grok-4-Fast (X.AI)
- DeepSeek-Chat-v3-0324 (DeepSeek)
- Claude-Haiku-4.5 (Anthropic)
- Gemini-2.5-Flash (Google)

### Benchmarks
- **HumanEval**: Function-level code generation benchmark (164 tasks)
- **MBPP** (Mostly Basic Python Programming): Python programming problems (257 tasks)
- **DS1000**: Data science code generation (150 tasks)

---

## Overall Model Performance Ranking

| Rank | Model | Total Tasks | Tasks Passed | Task Pass Rate | Test Pass Rate |
|------|-------|------------|--------------|----------------|----------------|
| 1 | **gpt-5** | 571 | 439 | **76.88%** | 77.66% |
| 2 | grok-4-fast | 571 | 415 | 72.68% | 74.50% |
| 3 | deepseek-chat-v3-0324 | 571 | 406 | 71.10% | 75.23% |
| 4 | claude-haiku-4.5 | 571 | 385 | 67.43% | 71.53% |
| 5 | gemini-2.5-flash | 571 | 385 | 67.43% | 71.89% |

### Key Findings
- **GPT-5** leads across all metrics with a 76.88% overall task pass rate
- There's a ~10% gap between the top performer (GPT-5) and the lowest performers
- Claude-Haiku-4.5 and Gemini-2.5-Flash are tied at 67.43% task pass rate

---

## Performance by Benchmark

### HumanEval (Function-level Code Generation)

| Model | Tasks Passed | Task Pass Rate |
|-------|--------------|----------------|
| gpt-5 | 157/164 | **95.73%** |
| grok-4-fast | 154/164 | 93.90% |
| claude-haiku-4.5 | 152/164 | 92.68% |
| deepseek-chat-v3-0324 | 147/164 | 89.63% |
| gemini-2.5-flash | 133/164 | 81.10% |

**Analysis**: All models perform relatively well on HumanEval, with most achieving >89% pass rate. GPT-5 leads with 95.73%, while Gemini-2.5-Flash trails at 81.10%.

### MBPP (Mostly Basic Python Programming)

| Model | Tasks Passed | Task Pass Rate |
|-------|--------------|----------------|
| gpt-5 | 191/257 | **74.32%** |
| deepseek-chat-v3-0324 | 187/257 | 72.76% |
| gemini-2.5-flash | 182/257 | 70.82% |
| claude-haiku-4.5 | 179/257 | 69.65% |
| grok-4-fast | 179/257 | 69.65% |

**Analysis**: Performance is more uniform on MBPP (~70-74% range). DeepSeek shows strength here, placing second despite being third overall.

### DS1000 (Data Science)

| Model | Tasks Passed | Task Pass Rate |
|-------|--------------|----------------|
| gpt-5 | 91/150 | **60.67%** |
| grok-4-fast | 82/150 | 54.67% |
| deepseek-chat-v3-0324 | 72/150 | 48.00% |
| gemini-2.5-flash | 70/150 | 46.67% |
| claude-haiku-4.5 | 54/150 | 36.00% |

**Analysis**: DS1000 appears to be the most challenging benchmark. Claude-Haiku-4.5 shows a significant drop (36%), suggesting difficulty with data science-specific coding patterns. GPT-5 maintains its lead at 60.67%.

---

## Error Analysis

Out of 2,855 tests, **825 tasks failed** (28.9% failure rate).

### Error Type Distribution

| Error Type | Count | Percentage |
|------------|-------|------------|
| AssertionError | 552 | 66.9% |
| TypeError | 100 | 12.1% |
| Other | 113 | 13.7% |
| NameError | 23 | 2.8% |
| SyntaxError | 20 | 2.4% |
| SetupError | 14 | 1.7% |
| TimeoutError | 3 | 0.4% |

### Error Breakdown by Model

| Model | AssertionError | TypeError | NameError | SyntaxError | SetupError | TimeoutError | Other |
|-------|---------------|-----------|-----------|-------------|------------|--------------|-------|
| claude-haiku-4.5 | 138 | 20 | 1 | 7 | 0 | 1 | 19 |
| deepseek-chat-v3-0324 | 118 | 22 | 2 | 0 | 0 | 0 | 23 |
| gemini-2.5-flash | 118 | 22 | 3 | 9 | 14 | 1 | 19 |
| gpt-5 | 71 | 12 | 17 | 4 | 0 | 0 | 28 |
| grok-4-fast | 107 | 24 | 0 | 0 | 0 | 1 | 24 |

### Key Error Insights
- **AssertionErrors** dominate (67%), indicating logic/correctness issues rather than syntax problems
- **GPT-5** has the fewest total errors (132), but notably more NameErrors (17) - suggesting occasional use of undefined variables
- **Gemini-2.5-Flash** has unique SetupErrors (14), likely from code that fails to compile/run
- **TimeoutErrors** are rare (<1%), indicating most models produce efficient code

---

## Conclusions

1. **GPT-5 is the top performer** across all benchmarks, with consistent performance in both simple (HumanEval) and complex (DS1000) tasks.

2. **DS1000 is the most challenging benchmark**, with pass rates 20-35% lower than HumanEval across all models.

3. **Model rankings shift by benchmark**:
   - HumanEval: GPT-5 > Grok-4 > Claude > DeepSeek > Gemini
   - MBPP: GPT-5 > DeepSeek > Gemini > Claude = Grok-4
   - DS1000: GPT-5 > Grok-4 > DeepSeek > Gemini > Claude

4. **Error patterns reveal model tendencies**:
   - Most errors are logical (AssertionErrors) not syntactic
   - GPT-5's NameErrors suggest occasional scope/variable issues
   - Gemini's SetupErrors indicate compilation challenges

5. **Claude-Haiku-4.5** underperforms on DS1000 specifically (36% vs 67% overall), indicating potential weakness in data science code generation.
