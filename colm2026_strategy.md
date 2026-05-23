# Strategic Direction for COLM 2026

## Where You Stand

Your paper has a strong **infrastructure** already written: abstract, introduction, related work, datasets, experimental setup, code generation results, and error analysis. The abstract promises four things:

1. LLMs can self-recognize their own code ✅ *You have data, but it's nuanced*
2. Self-preference bias (higher scores to own code) ❌ *Not yet tested*
3. R&P obfuscation framework ❌ *Not yet built or evaluated*
4. Identifiability–evaluability trade-off ❌ *Not yet measured*

Your experimental results tell a **more interesting and publishable story** than the one the abstract currently promises. Here's why, and what to do about it.

---

## The Core Problem: Your Results Are Nuanced, Not Clean

The literature sets up two opposing camps:
- **"Models can self-recognize"** (Panickssery et al. 2024: GPT-4 at 73.5%; CoSur reaches ~99%)
- **"Models cannot self-recognize"** (Davidson et al. 2024; Bai et al. 2025: near chance, attribution bias toward frontier families)

Your results land in the **messy middle**, which is actually the most valuable space for a paper:

| Finding | Why It's Interesting |
|---------|---------------------|
| GPT-5 self-recognizes at 75.9% (pair), but 0% true positive in IPP | The "success" is paradigm-dependent and driven by response bias, not genuine recognition |
| Full attribution reaches 88% when model names are given | Models can match code to *names* but can't recognize their *own* code—this is **stereotyping, not identity** |
| Claude/GPT never say "yes" in IPP; Grok/Gemini always say "yes" | This is a **response calibration** problem masquerading as recognition |
| DeepSeek-V3 is the only balanced model, yet only ~60% accurate | Balance ≠ ability; even the "best calibrated" model barely beats chance |

> [!IMPORTANT]
> **The key insight your paper should deliver:** LLMs don't recognize their own code—they recognize *brand associations* and *stereotypes* about which models write which kind of code. This is closer to **social cognition bias** than to **self-awareness**.

---

## Recommended Paper Direction

### Option A: "Code Attribution ≠ Self-Recognition" (Recommended)

**Thesis:** When LLMs appear to attribute code correctly, they're relying on stereotypes about model families rather than genuine stylistic self-recognition. This distinction matters for AI safety because stereotype-based attribution is fragile and exploitable.

**Why this works for COLM 2026:**
- COLM values **mechanistic understanding** and **careful evaluation methodology**
- This finding directly contradicts/refines multiple recent papers (Panickssery 2024, Bai 2025)
- The three-paradigm comparison (pair, IPP, full attribution) is **methodologically novel** for code
- The "no" bias / "yes" bias decomposition is a **new finding** nobody has reported

**Structure:**
1. **RQ1** (Self-Recognition Pair): Only 1/5 models above chance → weak evidence
2. **RQ2** (Self-Recognition IPP): Above-chance accuracy is entirely driven by response bias, not true recognition
3. **RQ3** (Full Attribution): When given model names, attribution works → models know stereotypes, not themselves
4. **Key analysis**: Decompose accuracy into TPR/TNR to show bias is the driver
5. **Obfuscation** (R&P): Show that simple transformations break even the partial attribution signal

### Option B: Keep the "Collusion" Framing

**Risk:** Your results don't actually show collusion (self-preference bias during evaluation). You'd need to run evaluation-bias experiments showing models rate their own code higher. This is a significant additional experiment.

---

## What You Need to Do (Priority Order)

### 🔴 Critical (Must Have for Submission)

1. **Fill in the Results section** with attribution task results (you have all the data)
   - Present the three-paradigm results with statistical tests
   - Add the TPR/TNR decomposition table (your "Own→Yes" / "Other→No" analysis)
   - This is the core contribution—currently the Results section stops after code generation

2. **Run more full attribution experiments** to fill gaps:
   - HumanEval and DS-1000 only have Grok-Code-Fast as judge—need at least 2-3 more judges
   - Need symmetric experiments: if GPT-5 judges Claude vs DeepSeek, also run Claude judging GPT-5 vs DeepSeek
   - Run more model pair combinations for full attribution

3. **Reframe the abstract and introduction** to match actual findings:
   - De-emphasize "collusion" if you don't have self-preference bias data
   - Emphasize the paradigm-dependent nature of self-recognition
   - Highlight the "stereotyping vs. identity" distinction

### 🟡 Important (Strongly Recommended)

4. **Implement and evaluate R&P (obfuscation)**:
   - You already have [obfuscate.py](file:///Users/ehsan/CodingProjects/llm-collusion/obfuscate.py) in the repo—run it on generated code
   - Re-run attribution experiments on obfuscated code
   - Show that obfuscation collapses attribution to chance (the literature strongly predicts this)

5. **Add the "self as judge" bias experiment**:
   - Have each model score pairs of code (its own vs. another model's) for quality
   - Compare scores when code is its own vs. not
   - This would validate the "collusion" narrative in your abstract

6. **Add pair self-recognition for HumanEval and DS-1000** (currently only MBPP)

### 🟢 Nice to Have

7. **Cross-model attribution matrix**: Have each model try to identify every other model's code
8. **Obfuscation robustness curves**: Vary obfuscation intensity, plot attribution accuracy decay
9. **Positional bias analysis**: You compute it but don't report it—include in appendix

---

## How Your Results Compare to the Literature

| Paper | Their Finding | Your Counterpoint |
|-------|--------------|-------------------|
| Panickssery 2024 (73.5% GPT-4 self-rec) | Self-recognition is real | Your IPP shows it's response bias, not recognition (0% TPR for GPT-5/Claude) |
| Davidson 2024 (no consistent self-rec) | Models pick "best" answer, not own | Your full attribution data agrees: models use stereotypes when given names |
| Bai 2025 (94% predictions → GPT/Claude) | Attribution bias toward frontier families | Your full attribution shows the same pattern—GPT-5 is best at attribution |
| Bisztray 2025 (95% code stylometry) | Trained classifiers work well | Your zero-shot LLM-based attribution reaches ~88% (full attribution)—competitive without training |
| Obfuscation papers (AUROC → 0.5) | Transformations break provenance | You should test this with R&P → show it also breaks zero-shot LLM attribution |

> [!TIP]
> Your **unique contribution** vs. all prior work: you study **code** self-recognition across **multiple experimental paradigms** (pair, IPP, full attribution) and show that the paradigm itself determines the result. No prior paper does this comparison on code.

---

## Suggested Timeline

| Week | Task |
|------|------|
| **Week 1** | Fill in Results section with existing data; run additional full attribution experiments (more judges, more datasets) |
| **Week 2** | Implement R&P obfuscation pipeline and re-run attribution on obfuscated code |
| **Week 3** | Run self-preference bias experiment (scoring own code higher); add pair self-rec for HumanEval/DS-1000 |
| **Week 4** | Rewrite abstract/intro to match findings; write Discussion/Analysis section |
| **Week 5** | Polish figures, write conclusion, proofread |

---

## Bottom Line

You have **enough data right now** to write a compelling Results section showing that LLM self-recognition in code is paradigm-dependent and driven by bias rather than genuine identity recognition. The R&P framework and self-preference bias experiments would strengthen the paper significantly but aren't strictly necessary if you reframe the contribution. The strongest version of this paper is the "stereotyping vs. self-awareness" story with R&P as the intervention that breaks even the stereotype-based signal.
