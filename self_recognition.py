from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterator

import typer
from rich.console import Console
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.lib import read_jsonl, render_prompt, OpenRouterClient, write_jsonl_line
from src.common.types import Pair, SelfRecognitionResult
from src.lib import load_config


app = typer.Typer(add_completion=False)
console = Console()



def normalize_dataset_name(name: Optional[str]) -> str:
    """Normalize dataset names to allow flexible filtering.

    Treat related variants like "mbpp" and "mbpp-sanitized" as the same family
    for the purpose of filtering, while preserving original names elsewhere.
    """
    s = str(name or "").strip().lower()
    if s.startswith("mbpp"):
        return "mbpp"
    return s


def iter_records(path: Path, dataset_filter: Optional[str]) -> Iterator[Dict[str, Any]]:
    norm_filter = normalize_dataset_name(dataset_filter) if dataset_filter else None
    for rec in read_jsonl(path):
        if norm_filter and normalize_dataset_name(rec.get("benchmark")) != norm_filter:
            continue
        yield rec


def build_pairs(records: Iterator[Dict[str, Any]], cfg: dict) -> List[Pair]:
    # group by (benchmark, task_id)
    from collections import defaultdict

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = (str(r.get("benchmark")), str(r.get("task_id")))
        grouped[key].append(r)

    pairs: List[Pair] = []
    for (benchmark, task_id), items in grouped.items():
        # Prefer order of models as configured for this dataset if present
        ds_cfg = cfg.get("datasets", {}).get(normalize_dataset_name(benchmark), {})
        preferred_order: List[str] = list(ds_cfg.get("models", []))

        # dedupe by model, keep first occurrence
        seen_models: set[str] = set()
        unique_items: List[Dict[str, Any]] = []
        for it in items:
            m = str(it.get("model_id"))
            if m in seen_models:
                continue
            seen_models.add(m)
            unique_items.append(it)

        # Sort unique_items by preferred order when available
        if preferred_order:
            idx = {m: i for i, m in enumerate(preferred_order)}
            unique_items.sort(key=lambda it: idx.get(str(it.get("model_id")), 1_000_000))

        if len(unique_items) < 2:
            continue

        a, b = unique_items[0], unique_items[1]
        pairs.append(
            Pair(
                benchmark=benchmark,
                task_id=str(task_id),
                task_prompt=str(a.get("prompt", "")),
                code1=str(a.get("generated_code", "")),
                code2=str(b.get("generated_code", "")),
                model1=str(a.get("model_id", "")),
                model2=str(b.get("model_id", "")),
            )
        )

    return pairs


def build_messages(task: str, code1: str, code2: str, prompt_path: Path) -> List[Dict[str, str]]:
    rendered = render_prompt(prompt_path, task=task, code1=code1, code2=code2)
    return [
        {"role": "system", "content": rendered["system"]},
        {"role": "user", "content": rendered["user"]},
    ]


def parse_choice(text: str) -> Optional[int]:
    s = (text or "").strip()
    for ch in s:
        if ch == "1":
            return 1
        if ch == "2":
            return 2
    return None


def execute(
    config_path: Path,
    input_path: Optional[Path],
    dataset: Optional[str],
    judge_model_override: Optional[str],
    concurrency_override: Optional[int],
    temperature: float,
) -> None:
    cfg = load_config(config_path)

    paths = cfg.get("paths", {})
    default_input = paths.get("output_path")
    source_path = Path(input_path or default_input or "").resolve()
    if not source_path.exists():
        raise typer.BadParameter(f"Input JSONL not found: {source_path}")

    # Client
    client = OpenRouterClient(api_key=cfg.get("api", {}).get("openrouter_api_key") or None)
    prompt_path = Path("prompts/self_recognition.yaml")

    # Concurrency
    max_workers = int(concurrency_override or int(cfg.get("api", {}).get("concurrency", 4)))

    # Build pairs
    records = iter_records(source_path, dataset)
    pairs = build_pairs(records, cfg)
    if not pairs:
        console.print("[yellow]No pairs found to evaluate.[/]")
        return

    console.print(f"Evaluating self-recognition on {len(pairs)} pairs")

    # Resolve judge model per dataset: override > dataset's first model
    def judge_for_dataset(ds: str) -> Optional[str]:
        if judge_model_override:
            return judge_model_override
        norm_ds = normalize_dataset_name(ds)
        models = list(cfg.get("datasets", {}).get(norm_ds, {}).get("models", []))
        return models[0] if models else None

    # Submit all valid jobs
    # Precompute judge model per dataset once
    dataset_names = {p.benchmark for p in pairs}
    ds_to_judge: Dict[str, Optional[str]] = {ds: judge_for_dataset(ds) for ds in dataset_names}

    jobs: List[Tuple[int, Pair, str]] = []
    for idx, pair in enumerate(pairs):
        judge_model = ds_to_judge.get(pair.benchmark) or None
        if not judge_model:
            continue
        # Only valid if judge is one of the two models in the pair
        if judge_model not in (pair.model1, pair.model2):
            continue
        jobs.append((idx, pair, judge_model))

    if not jobs:
        console.print("[yellow]No valid pairs where judge model is one of the two.[/]")
        return

    total = len(jobs)
    correct = 0
    processed = 0
    skipped = len(pairs) - total

    # Determine results output path
    paths = cfg.get("paths", {})
    data_dir = Path(paths.get("data_dir", "data"))
    results_dir = data_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    # filename includes dataset and timestamp
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    ds_tag = normalize_dataset_name(dataset) if dataset else "all"
    results_path = results_dir / f"self_recognition-{ds_tag}-{ts}.jsonl"

    def submit_job(job: Tuple[int, Pair, str]) -> Tuple[int, str, Optional[int], Optional[int]]:
        _, p, judge_model = job
        messages = build_messages(p.task_prompt, p.code1, p.code2, prompt_path)
        resp = client.generate_code(prompt="", model=judge_model, messages=messages, temperature=temperature)
        choice = parse_choice(resp)
        # ground truth index
        truth: Optional[int] = 1 if judge_model == p.model1 else (2 if judge_model == p.model2 else None)
        return (job[0], resp, choice, truth)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(submit_job, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Judging", unit="req"):
            try:
                job_idx, resp_text, choice, truth = fut.result()
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]Judge request failed[/]: {e}")
                continue

            processed += 1
            if choice is None or truth is None:
                # Still write a result row with missing values
                _, pair, judge_model = jobs[job_idx]
                result = SelfRecognitionResult(
                    benchmark=pair.benchmark,
                    task_id=pair.task_id,
                    evaluator_model=judge_model,
                    candidate_1_model=pair.model1,
                    candidate_2_model=pair.model2,
                    predicted_candidate=choice,
                    gold_candidate=truth,
                    is_correct=None,
                    evaluator_response=resp_text,
                )
                write_jsonl_line(results_path, result.to_dict())
                continue
            is_correct = choice == truth
            if is_correct:
                correct += 1
            # Persist result row
            _, pair, judge_model = jobs[job_idx]
            result = SelfRecognitionResult(
                benchmark=pair.benchmark,
                task_id=pair.task_id,
                evaluator_model=judge_model,
                candidate_1_model=pair.model1,
                candidate_2_model=pair.model2,
                predicted_candidate=choice,
                gold_candidate=truth,
                is_correct=is_correct,
                evaluator_response=resp_text,
            )
            write_jsonl_line(results_path, result.to_dict())

    if processed == 0:
        console.print("[yellow]No judge responses processed.[/]")
        return

    acc = correct / processed
    console.print(
        f"[green]Done.[/] Valid pairs: {total}, Processed: {processed}, Skipped: {skipped}, "
        f"Correct: {correct}, Accuracy: {acc:.3f}"
    )
    console.print(f"[green]Saved results[/] -> {results_path}")


@app.command()
def run(
    config_path: Path = typer.Option(Path("configs/config.yaml"), help="Path to config YAML"),
    input_path: Optional[Path] = typer.Option(None, "--input", "-i", help="Path to generations JSONL"),
    dataset: Optional[str] = typer.Option(None, help="Filter to a dataset name (optional)"),
    judge_model: Optional[str] = typer.Option(None, help="Override judge model ID"),
    concurrency: Optional[int] = typer.Option(None, help="Override concurrency for judge requests"),
    temperature: float = typer.Option(0.0, help="Temperature for judge model"),
):
    execute(
        config_path=config_path,
        input_path=input_path,
        dataset=dataset,
        judge_model_override=judge_model,
        concurrency_override=concurrency,
        temperature=temperature,
    )


if __name__ == "__main__":
    app()


