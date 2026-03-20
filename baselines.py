"""Baseline evaluation for target identification.

Computes non-LLM baselines to establish lower bounds for comparison:
  1. Random (50%)
  2. Always-A / Always-B
  3. Longer code heuristic (pick the longer code as target)
  4. Shorter code heuristic (pick the shorter code as target)
  5. Docstring heuristic (pick code with/without docstrings)
  6. Code complexity (more lines = target)

Usage:
    python baselines.py --dataset-folder mbpp-sanitized --split test \
        --model1 anthropic/claude-haiku-4.5 --model2 deepseek/deepseek-chat-v3-0324 \
        --target deepseek/deepseek-chat-v3-0324
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import typer
from rich.console import Console
from rich.table import Table  # type: ignore[import]

from src.lib import load_config, read_jsonl
from src.common.types import Pair

app = typer.Typer(add_completion=False)
console = Console()


# ---------------------------------------------------------------------------
# Pair building (same as target_identification.py)
# ---------------------------------------------------------------------------

def iter_records(source_path: Path) -> Iterator[Dict[str, Any]]:
    if source_path.is_dir():
        for f in sorted(source_path.glob("*.jsonl")):
            for rec in read_jsonl(f):
                yield rec
    else:
        yield from read_jsonl(source_path)


def build_pairs(records: Iterator[Dict[str, Any]], model1: str, model2: str, seed: int = 42) -> List[Pair]:
    random.seed(seed)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = (str(r.get("benchmark")), str(r.get("task_id")))
        grouped[key].append(r)

    pairs: List[Pair] = []
    for (benchmark, task_id), items in grouped.items():
        m1_item = m2_item = None
        for item in items:
            model_name = str(item.get("model_name"))
            if model_name == model1:
                m1_item = item
            elif model_name == model2:
                m2_item = item
        if m1_item is None or m2_item is None:
            continue

        # Randomize position (same logic as target_identification.py)
        if random.random() < 0.5:
            code1, code2 = m1_item, m2_item
            m1, m2 = model1, model2
        else:
            code1, code2 = m2_item, m1_item
            m1, m2 = model2, model1

        pairs.append(Pair(
            benchmark=benchmark, task_id=str(task_id),
            task_prompt=str(code1.get("prompt", "")),
            code1=str(code1.get("generated_code", "")),
            code2=str(code2.get("generated_code", "")),
            model1=m1, model2=m2,
        ))
    return pairs


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def has_docstring(code: str) -> bool:
    """Check if code contains a docstring (triple-quoted string)."""
    return bool(re.search(r'""".*?"""', code, re.DOTALL) or re.search(r"'''.*?'''", code, re.DOTALL))


def count_comments(code: str) -> int:
    """Count single-line comments."""
    return sum(1 for line in code.split("\n") if line.strip().startswith("#"))


def count_lines(code: str) -> int:
    """Count non-empty lines."""
    return sum(1 for line in code.split("\n") if line.strip())


def avg_line_length(code: str) -> float:
    """Average length of non-empty lines."""
    lines = [line for line in code.split("\n") if line.strip()]
    return sum(len(l) for l in lines) / max(len(lines), 1)


def has_type_hints(code: str) -> bool:
    """Check for type hint patterns."""
    return bool(re.search(r":\s*(int|str|float|bool|List|Dict|Optional|Tuple|Any)\b", code) or
                re.search(r"->\s*(int|str|float|bool|List|Dict|Optional|Tuple|Any|None)\b", code))


# ---------------------------------------------------------------------------
# Baseline strategies
# ---------------------------------------------------------------------------

def baseline_random(pairs: List[Pair], target_model: str, seed: int = 99) -> List[str]:
    """Random baseline: pick A or B with 50/50 probability."""
    rng = random.Random(seed)
    return [rng.choice(["1", "2"]) for _ in pairs]


def baseline_always_a(pairs: List[Pair], target_model: str) -> List[str]:
    """Always pick Solution A."""
    return ["1"] * len(pairs)


def baseline_always_b(pairs: List[Pair], target_model: str) -> List[str]:
    """Always pick Solution B."""
    return ["2"] * len(pairs)


def baseline_longer_code(pairs: List[Pair], target_model: str) -> List[str]:
    """Pick the longer code as the target."""
    return ["1" if len(p.code1) >= len(p.code2) else "2" for p in pairs]


def baseline_shorter_code(pairs: List[Pair], target_model: str) -> List[str]:
    """Pick the shorter code as the target."""
    return ["1" if len(p.code1) <= len(p.code2) else "2" for p in pairs]


def baseline_more_lines(pairs: List[Pair], target_model: str) -> List[str]:
    """Pick the code with more lines."""
    return ["1" if count_lines(p.code1) >= count_lines(p.code2) else "2" for p in pairs]


def baseline_has_docstring(pairs: List[Pair], target_model: str) -> List[str]:
    """Pick the code that has a docstring (or A if both/neither)."""
    results = []
    for p in pairs:
        d1, d2 = has_docstring(p.code1), has_docstring(p.code2)
        if d1 and not d2:
            results.append("1")
        elif d2 and not d1:
            results.append("2")
        else:
            results.append("1")  # default when tied
    return results


def baseline_no_docstring(pairs: List[Pair], target_model: str) -> List[str]:
    """Pick the code that does NOT have a docstring."""
    results = []
    for p in pairs:
        d1, d2 = has_docstring(p.code1), has_docstring(p.code2)
        if not d1 and d2:
            results.append("1")
        elif not d2 and d1:
            results.append("2")
        else:
            results.append("1")
    return results


def baseline_more_comments(pairs: List[Pair], target_model: str) -> List[str]:
    """Pick the code with more comments."""
    return ["1" if count_comments(p.code1) >= count_comments(p.code2) else "2" for p in pairs]


def baseline_has_type_hints(pairs: List[Pair], target_model: str) -> List[str]:
    """Pick the code that has type hints."""
    results = []
    for p in pairs:
        t1, t2 = has_type_hints(p.code1), has_type_hints(p.code2)
        if t1 and not t2:
            results.append("1")
        elif t2 and not t1:
            results.append("2")
        else:
            results.append("1")
    return results


def baseline_longer_avg_line(pairs: List[Pair], target_model: str) -> List[str]:
    """Pick code with longer average line length."""
    return ["1" if avg_line_length(p.code1) >= avg_line_length(p.code2) else "2" for p in pairs]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

ALL_BASELINES = {
    "Random (50%)": baseline_random,
    "Always A": baseline_always_a,
    "Always B": baseline_always_b,
    "Longer code": baseline_longer_code,
    "Shorter code": baseline_shorter_code,
    "More lines": baseline_more_lines,
    "Has docstring": baseline_has_docstring,
    "No docstring": baseline_no_docstring,
    "More comments": baseline_more_comments,
    "Has type hints": baseline_has_type_hints,
    "Longer avg line": baseline_longer_avg_line,
}


def evaluate_baselines(
    pairs: List[Pair], target_model: str
) -> List[Tuple[str, int, int, float]]:
    """Run all baselines and return (name, correct, total, accuracy)."""
    # Compute gold labels
    gold = []
    for p in pairs:
        if p.model1 == target_model:
            gold.append("1")
        elif p.model2 == target_model:
            gold.append("2")
        else:
            gold.append("?")

    results = []
    for name, baseline_fn in ALL_BASELINES.items():
        preds = baseline_fn(pairs, target_model)
        correct = sum(1 for g, p in zip(gold, preds) if g == p)
        total = len(pairs)
        acc = correct / total if total > 0 else 0.0
        results.append((name, correct, total, acc))

    # Sort by accuracy descending
    results.sort(key=lambda x: -x[3])
    return results


def print_feature_distribution(pairs: List[Pair], target_model: str) -> None:
    """Print feature distributions to understand baseline performance."""
    console.print("\n[bold]Feature Distribution Analysis[/]")

    target_codes = []
    other_codes = []
    for p in pairs:
        if p.model1 == target_model:
            target_codes.append(p.code1)
            other_codes.append(p.code2)
        else:
            target_codes.append(p.code2)
            other_codes.append(p.code1)

    features = [
        ("Avg code length (chars)", lambda c: len(c)),
        ("Avg lines", lambda c: count_lines(c)),
        ("Has docstring (%)", lambda c: int(has_docstring(c))),
        ("Avg comments", lambda c: count_comments(c)),
        ("Has type hints (%)", lambda c: int(has_type_hints(c))),
        ("Avg line length", lambda c: avg_line_length(c)),
    ]

    table = Table(title="Feature Comparison")
    table.add_column("Feature", style="cyan")
    table.add_column(f"Target ({target_model.split('/')[-1]})", justify="right")
    table.add_column("Other model", justify="right")
    table.add_column("Diff", justify="right")

    for fname, fn in features:
        t_vals = [fn(c) for c in target_codes]
        o_vals = [fn(c) for c in other_codes]
        t_avg = sum(t_vals) / len(t_vals) if t_vals else 0
        o_avg = sum(o_vals) / len(o_vals) if o_vals else 0

        if "%" in fname:
            table.add_row(fname, f"{t_avg:.0%}", f"{o_avg:.0%}", f"{t_avg - o_avg:+.0%}")
        else:
            table.add_row(fname, f"{t_avg:.1f}", f"{o_avg:.1f}", f"{t_avg - o_avg:+.1f}")

    console.print(table)


@app.command()
def run(
    dataset_folder: str = typer.Option(..., "--dataset-folder", help="Dataset folder (e.g., mbpp-sanitized)"),
    split: str = typer.Option("test", "--split", help="Dataset split"),
    model1: str = typer.Option(..., "--model1", help="First model"),
    model2: str = typer.Option(..., "--model2", help="Second model"),
    target: str = typer.Option(..., "--target", help="Target model (must be model1 or model2)"),
    seed: int = typer.Option(42, help="Random seed"),
    show_features: bool = typer.Option(True, "--features/--no-features", help="Show feature distributions"),
):
    """Run baseline evaluations for target identification."""
    config_path = Path("configs/config.yaml")
    cfg = load_config(config_path)
    data_dir = Path(cfg.get("paths", {}).get("data_dir", "data"))
    source_path = data_dir / "code_generation" / dataset_folder / split

    if not source_path.exists():
        raise typer.BadParameter(f"Path not found: {source_path}")
    if target not in (model1, model2):
        raise typer.BadParameter(f"Target '{target}' must be model1 or model2")

    console.print(f"[blue]Loading pairs from {source_path}[/]")
    records = iter_records(source_path)
    pairs = build_pairs(records, model1, model2, seed=seed)
    console.print(f"Built {len(pairs)} pairs: {model1.split('/')[-1]} vs {model2.split('/')[-1]}")
    console.print(f"Target: {target}")

    # Show feature distributions
    if show_features:
        print_feature_distribution(pairs, target)

    # Run baselines
    results = evaluate_baselines(pairs, target)

    # Display results
    table = Table(title="\nBaseline Results")
    table.add_column("Baseline", style="cyan")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Accuracy", justify="right", style="bold")

    for name, correct, total, acc in results:
        style = "green" if acc > 0.6 else ("yellow" if acc > 0.5 else "red")
        table.add_row(name, str(correct), str(total), f"[{style}]{acc:.1%}[/{style}]")

    console.print(table)

    # Print best non-random baseline
    non_random = [(n, c, t, a) for n, c, t, a in results if "Random" not in n and "Always" not in n]
    if non_random:
        best = non_random[0]
        console.print(f"\n[bold]Best heuristic baseline:[/] {best[0]} = {best[3]:.1%}")
        console.print(f"[dim]LLM judges must beat this to demonstrate genuine style detection.[/]")


if __name__ == "__main__":
    app()
