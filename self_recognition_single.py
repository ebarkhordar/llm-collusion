"""Single-code self-recognition (IPP – Individual Presentation Paradigm).

For each evaluator model, present individual code snippets and ask:
  "Did you generate this code?" → yes / no

Prompt: prompts/model_attribution/self_recognition.md
"""
from __future__ import annotations

import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from tqdm import tqdm

from src.lib import read_jsonl, render_prompt, OpenRouterClient, write_jsonl_line
from src.lib import load_config

app = typer.Typer(add_completion=False)
console = Console()

PROMPT_PATH = Path("prompts/model_attribution/self_recognition.md")


@dataclass
class SingleRecJob:
    """One evaluation job: show `code` to `evaluator` and ask if it wrote it."""
    benchmark: str
    task_id: str
    task_prompt: str
    code: str
    code_model: str       # model that actually wrote the code
    evaluator: str        # model being asked
    expected: str         # "yes" if code_model == evaluator, else "no"


@dataclass
class SingleRecResult:
    benchmark: str
    task_id: str
    code_model: str
    evaluator: str
    expected: str         # "yes" or "no"
    predicted: Optional[str]
    is_correct: Optional[bool]
    raw_response: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_yes_no(text: str) -> Optional[str]:
    """Parse a yes/no response."""
    s = (text or "").strip().lower()
    for word in s.split():
        w = word.strip(".,!?\"'")
        if w == "yes":
            return "yes"
        if w == "no":
            return "no"
    # Fallback: check first character
    if s and s[0] == "y":
        return "yes"
    if s and s[0] == "n":
        return "no"
    return None


def build_jobs(
    source_path: Path,
    evaluator: str,
    other_models: List[str],
    dataset_filter: Optional[str] = None,
    max_other: int = 2,
) -> List[SingleRecJob]:
    """Build evaluation jobs.

    For the evaluator model:
      - Show its OWN code → expected "yes"
      - Show code from `other_models` → expected "no"
    """
    # Read evaluator's own code
    eval_safe = evaluator.replace("/", "-").replace(":", "-")
    eval_file = source_path / f"{eval_safe}.jsonl"
    if not eval_file.exists():
        console.print(f"[yellow]No code file for evaluator {evaluator}: {eval_file}[/]")
        return []

    own_records = list(read_jsonl(eval_file))
    jobs: List[SingleRecJob] = []

    # "Yes" jobs — evaluator's own code
    for rec in own_records:
        if dataset_filter and str(rec.get("benchmark", "")).strip().lower() != dataset_filter:
            continue
        jobs.append(SingleRecJob(
            benchmark=str(rec.get("benchmark", "")),
            task_id=str(rec.get("task_id", "")),
            task_prompt=str(rec.get("prompt", "")),
            code=str(rec.get("generated_code", "")),
            code_model=evaluator,
            evaluator=evaluator,
            expected="yes",
        ))

    # "No" jobs — code from other models
    selected_others = other_models[:max_other]
    for other_model in selected_others:
        other_safe = other_model.replace("/", "-").replace(":", "-")
        other_file = source_path / f"{other_safe}.jsonl"
        if not other_file.exists():
            continue
        other_records = list(read_jsonl(other_file))
        for rec in other_records:
            if dataset_filter and str(rec.get("benchmark", "")).strip().lower() != dataset_filter:
                continue
            jobs.append(SingleRecJob(
                benchmark=str(rec.get("benchmark", "")),
                task_id=str(rec.get("task_id", "")),
                task_prompt=str(rec.get("prompt", "")),
                code=str(rec.get("generated_code", "")),
                code_model=other_model,
                evaluator=evaluator,
                expected="no",
            ))

    random.shuffle(jobs)
    return jobs


def judge_one(
    job: SingleRecJob,
    client: OpenRouterClient,
    temperature: float,
) -> SingleRecResult:
    """Send one code snippet to the evaluator and get yes/no."""
    rendered = render_prompt(PROMPT_PATH, prompt=job.task_prompt, code=job.code)
    messages = [{"role": "user", "content": str(rendered.get("user", "")).strip()}]

    resp = client.generate_code(model=job.evaluator, messages=messages, temperature=temperature)
    resp_text = str(resp) if resp else ""
    predicted = parse_yes_no(resp_text)
    is_correct = (predicted == job.expected) if predicted else None

    return SingleRecResult(
        benchmark=job.benchmark,
        task_id=job.task_id,
        code_model=job.code_model,
        evaluator=job.evaluator,
        expected=job.expected,
        predicted=predicted,
        is_correct=is_correct,
        raw_response=resp_text,
    )


@app.command()
def run(
    dataset_folder: str = typer.Option(..., "--dataset-folder", help="Dataset folder (e.g., mbpp-sanitized)"),
    split: str = typer.Option(..., "--split", help="Split (e.g., test)"),
    evaluator: str = typer.Option(..., "--evaluator", help="Model to evaluate (e.g., openai/gpt-5)"),
    max_other: int = typer.Option(2, "--max-other", help="Max number of other models to test 'no' against"),
    concurrency: int = typer.Option(8, "--concurrency", help="Concurrent requests"),
    temperature: float = typer.Option(0.0, help="Temperature"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Run single-code self-recognition (IPP paradigm)."""
    random.seed(seed)

    config_path = Path("configs/config.yaml")
    cfg = load_config(config_path)
    data_dir = Path(cfg.get("paths", {}).get("data_dir", "data"))
    source_path = data_dir / "code_generation" / dataset_folder / split

    if not source_path.exists():
        raise typer.BadParameter(f"Input path not found: {source_path}")

    # Discover all models from files
    all_models = []
    for f in sorted(source_path.glob("*.jsonl")):
        # Convert filename back to model ID
        # e.g., openai-gpt-5.jsonl → need to read file to get actual model name
        recs = list(read_jsonl(f))
        if recs:
            model_name = str(recs[0].get("model_name", ""))
            if model_name and model_name not in all_models:
                all_models.append(model_name)

    other_models = [m for m in all_models if m != evaluator]
    console.print(f"[blue]Evaluator:[/] {evaluator}")
    console.print(f"[blue]Other models (max {max_other}):[/] {other_models[:max_other]}")
    console.print(f"[blue]Source:[/] {source_path}")

    jobs = build_jobs(source_path, evaluator, other_models, max_other=max_other)
    if not jobs:
        console.print("[yellow]No jobs to run.[/]")
        return

    yes_count = sum(1 for j in jobs if j.expected == "yes")
    no_count = sum(1 for j in jobs if j.expected == "no")
    console.print(f"\nTotal jobs: {len(jobs)}  (own={yes_count}, other={no_count})")

    # Run
    client = OpenRouterClient()
    results: List[SingleRecResult] = []

    # Output directory
    out_dir = data_dir / "self_recognition_single" / dataset_folder / split
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_safe = evaluator.replace("/", "-").replace(":", "-")
    results_path = out_dir / f"{eval_safe}.jsonl"

    # Clear previous results
    if results_path.exists():
        results_path.unlink()

    correct = 0
    processed = 0
    tp = fp = tn = fn = 0  # confusion matrix

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(judge_one, job, client, temperature): job for job in jobs}
        with tqdm(total=len(jobs), desc="Judging", unit="req") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    write_jsonl_line(results_path, result.to_dict())
                    processed += 1

                    if result.is_correct is not None:
                        if result.is_correct:
                            correct += 1
                        # Confusion matrix
                        if result.expected == "yes" and result.predicted == "yes":
                            tp += 1
                        elif result.expected == "no" and result.predicted == "yes":
                            fp += 1
                        elif result.expected == "no" and result.predicted == "no":
                            tn += 1
                        elif result.expected == "yes" and result.predicted == "no":
                            fn += 1
                except Exception as e:
                    console.print(f"[red]Error: {e}[/]")
                pbar.update(1)

    if processed == 0:
        console.print("[yellow]No results.[/]")
        return

    acc = correct / processed
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    console.print(f"\n[green]Done.[/] Processed: {processed}, Correct: {correct}, Accuracy: {acc:.3f}")
    console.print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    console.print(f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
    console.print(f"\n[green]Results saved:[/] {results_path}")

    # Save metadata
    prompt_hash = hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()[:12]
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "task": "self_recognition_single",
        "paradigm": "IPP",
        "prompt_file": str(PROMPT_PATH),
        "prompt_sha256": prompt_hash,
        "evaluator": evaluator,
        "other_models": other_models[:max_other],
        "temperature": temperature,
        "seed": seed,
        "input_path": str(source_path),
        "dataset_folder": dataset_folder,
        "split": split,
        "total_jobs": len(jobs),
        "yes_jobs": yes_count,
        "no_jobs": no_count,
        "processed": processed,
        "correct": correct,
        "accuracy": round(acc, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "results_file": str(results_path),
    }
    meta_path = results_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2))
    console.print(f"  -> {meta_path.name}")


if __name__ == "__main__":
    app()
