#!/usr/bin/env python3
"""Statistical analysis of LLM self-recognition and full attribution results.

Computes:
- Binomial test p-values vs 50% baseline for each experiment
- 95% Wilson confidence intervals
- Positional bias analysis (how often Code1 vs Code2 is chosen)
- Summary table ready for the paper
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ── Wilson Score Interval ──────────────────────────────────────────────────


def wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> Tuple[float, float]:
    """Compute Wilson score 95% confidence interval."""
    if n_total == 0:
        return (0.0, 0.0)
    p_hat = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


# ── Binomial test (exact, two-sided vs p0=0.5) ────────────────────────────


def binomial_test_p(n_success: int, n_total: int, p0: float = 0.5) -> float:
    """Two-sided exact binomial test p-value vs null hypothesis p0."""
    from math import comb, log

    if n_total == 0:
        return 1.0

    # PMF of Binomial(n_total, p0)
    def pmf(k: int) -> float:
        return comb(n_total, k) * p0**k * (1 - p0) ** (n_total - k)

    observed_pmf = pmf(n_success)
    # Sum probabilities of all outcomes as extreme or more extreme
    p_value = sum(pmf(k) for k in range(n_total + 1) if pmf(k) <= observed_pmf + 1e-15)
    return min(p_value, 1.0)


# ── Load results ──────────────────────────────────────────────────────────


@dataclass
class ExperimentResult:
    name: str
    experiment_type: str  # "self_recognition" or "full_attribution"
    n_total: int
    n_correct: int
    n_unparsed: int  # responses that couldn't be parsed
    accuracy: float
    p_value: float
    ci_low: float
    ci_high: float
    pos_bias_code1: float  # fraction of times code 1 was chosen
    pos_bias_code2: float  # fraction of times code 2 was chosen


def load_self_recognition(path: Path) -> Optional[ExperimentResult]:
    """Load and analyze a self-recognition result file."""
    records = [json.loads(line) for line in path.read_text().strip().split("\n") if line.strip()]
    if not records:
        return None

    evaluator = records[0].get("evaluator_model", "unknown")
    n_total = 0
    n_correct = 0
    n_unparsed = 0
    code1_chosen = 0
    code2_chosen = 0

    for r in records:
        predicted = r.get("predicted_candidate")
        gold = r.get("gold_candidate")
        is_correct = r.get("is_correct")

        if predicted is None:
            n_unparsed += 1
            continue

        n_total += 1
        if predicted == 1:
            code1_chosen += 1
        elif predicted == 2:
            code2_chosen += 1

        if is_correct is True:
            n_correct += 1

    if n_total == 0:
        return None

    accuracy = n_correct / n_total
    p_val = binomial_test_p(n_correct, n_total, 0.5)
    ci_lo, ci_hi = wilson_ci(n_correct, n_total)
    total_choices = code1_chosen + code2_chosen
    pos1 = code1_chosen / total_choices if total_choices > 0 else 0.5
    pos2 = code2_chosen / total_choices if total_choices > 0 else 0.5

    short_name = evaluator.split("/")[-1] if "/" in evaluator else evaluator
    return ExperimentResult(
        name=f"Self-Rec: {short_name}",
        experiment_type="self_recognition",
        n_total=n_total,
        n_correct=n_correct,
        n_unparsed=n_unparsed,
        accuracy=accuracy,
        p_value=p_val,
        ci_low=ci_lo,
        ci_high=ci_hi,
        pos_bias_code1=pos1,
        pos_bias_code2=pos2,
    )


def load_full_attribution(path: Path) -> Optional[ExperimentResult]:
    """Load and analyze a full attribution result file."""
    records = [json.loads(line) for line in path.read_text().strip().split("\n") if line.strip()]
    if not records:
        return None

    judge = records[0].get("judge_model", "unknown")
    model1 = records[0].get("model1", "unknown")
    model2 = records[0].get("model2", "unknown")
    n_total = 0
    n_correct = 0
    n_unparsed = 0
    code1_assigned_to_model1 = 0  # how often Solution A is attributed to model1

    for r in records:
        predicted = r.get("predicted_attribution")
        is_correct = r.get("is_correct")

        if predicted is None or not isinstance(predicted, dict):
            n_unparsed += 1
            continue

        n_total += 1

        # Check positional bias: was Solution A (or legacy Code1) attributed to model1?
        a_attr = predicted.get("A", "") or predicted.get("Code1", "")
        if a_attr == r.get("model1"):
            code1_assigned_to_model1 += 1

        if is_correct is True:
            n_correct += 1

    if n_total == 0:
        return None

    accuracy = n_correct / n_total
    p_val = binomial_test_p(n_correct, n_total, 0.5)
    ci_lo, ci_hi = wilson_ci(n_correct, n_total)
    pos1 = code1_assigned_to_model1 / n_total if n_total > 0 else 0.5
    pos2 = 1.0 - pos1

    short_judge = judge.split("/")[-1] if "/" in judge else judge
    short_m1 = model1.split("/")[-1] if "/" in model1 else model1
    short_m2 = model2.split("/")[-1] if "/" in model2 else model2
    return ExperimentResult(
        name=f"Attr: {short_judge} → {short_m1} vs {short_m2}",
        experiment_type="full_attribution",
        n_total=n_total,
        n_correct=n_correct,
        n_unparsed=n_unparsed,
        accuracy=accuracy,
        p_value=p_val,
        ci_low=ci_lo,
        ci_high=ci_hi,
        pos_bias_code1=pos1,
        pos_bias_code2=pos2,
    )


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    data_dir = Path("data")
    results: List[ExperimentResult] = []

    # Load self-recognition results
    sr_dir = data_dir / "self_recognition"
    if sr_dir.exists():
        for jsonl_file in sorted(sr_dir.rglob("*.jsonl")):
            r = load_self_recognition(jsonl_file)
            if r:
                results.append(r)

    # Load full attribution results
    fa_dir = data_dir / "full_attribution"
    if fa_dir.exists():
        for jsonl_file in sorted(fa_dir.rglob("*.jsonl")):
            r = load_full_attribution(jsonl_file)
            if r:
                results.append(r)

    if not results:
        print("No results found.")
        return

    # Print summary table
    print("\n" + "=" * 100)
    print("STATISTICAL ANALYSIS OF LLM ATTRIBUTION EXPERIMENTS")
    print("=" * 100)

    # Self-recognition table
    sr_results = [r for r in results if r.experiment_type == "self_recognition"]
    if sr_results:
        print("\n── SELF-RECOGNITION (Task 1) ─────────────────────────────────────────────")
        print(f"{'Model':<30} {'N':>5} {'Correct':>8} {'Acc':>7} {'95% CI':>16} {'p-value':>12} {'Sig':>5} {'Pos1%':>7} {'Pos2%':>7}")
        print("-" * 100)
        for r in sorted(sr_results, key=lambda x: x.accuracy, reverse=True):
            sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else "ns"
            ci_str = f"[{r.ci_low:.3f}, {r.ci_high:.3f}]"
            print(f"{r.name:<30} {r.n_total:>5} {r.n_correct:>8} {r.accuracy:>7.3f} {ci_str:>16} {r.p_value:>12.6f} {sig:>5} {r.pos_bias_code1:>6.1%} {r.pos_bias_code2:>6.1%}")

    # Full attribution table
    fa_results = [r for r in results if r.experiment_type == "full_attribution"]
    if fa_results:
        print("\n── FULL ATTRIBUTION (Task 3) ─────────────────────────────────────────────")
        print(f"{'Experiment':<50} {'N':>5} {'Correct':>8} {'Acc':>7} {'95% CI':>16} {'p-value':>12} {'Sig':>5}")
        print("-" * 100)
        for r in sorted(fa_results, key=lambda x: x.accuracy, reverse=True):
            sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else "ns"
            ci_str = f"[{r.ci_low:.3f}, {r.ci_high:.3f}]"
            print(f"{r.name:<50} {r.n_total:>5} {r.n_correct:>8} {r.accuracy:>7.3f} {ci_str:>16} {r.p_value:>12.6f} {sig:>5}")

    # Summary statistics
    print("\n── SUMMARY ──────────────────────────────────────────────────────────────")
    sig_sr = [r for r in sr_results if r.p_value < 0.05 and r.accuracy > 0.5]
    sig_fa = [r for r in fa_results if r.p_value < 0.05 and r.accuracy > 0.5]
    print(f"Self-recognition experiments above chance (p<0.05): {len(sig_sr)}/{len(sr_results)}")
    print(f"Full attribution experiments above chance (p<0.05): {len(sig_fa)}/{len(fa_results)}")

    # Significance legend
    print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("Baseline: 50% (random guessing)\n")


if __name__ == "__main__":
    main()
