from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterator

import typer
from rich.console import Console
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

from src.lib import read_jsonl, render_prompt, OpenRouterClient, write_jsonl_line
from src.common.types import Pair, SelfRecognitionResult
from src.lib import load_config


app = typer.Typer(add_completion=False)
console = Console()

def iter_records(source_path: Path, dataset_filter: Optional[str]) -> Iterator[Dict[str, Any]]:
    """Read records from either a single JSONL file or all JSONL files in a directory."""
    norm_filter = (str(dataset_filter).strip().lower()) if dataset_filter else None
    
    # If source_path is a directory, read all JSONL files from it
    if source_path.is_dir():
        jsonl_files = sorted(source_path.glob("*.jsonl"))
        if not jsonl_files:
            console.print(f"[yellow]No JSONL files found in directory: {source_path}[/]")
            return
        for jsonl_file in jsonl_files:
            for rec in read_jsonl(jsonl_file):
                rec_ds = str(rec.get("benchmark", "")).strip().lower()
                if norm_filter and rec_ds != norm_filter:
                    continue
                yield rec
    else:
        # Single file mode (backward compatibility)
        for rec in read_jsonl(source_path):
            rec_ds = str(rec.get("benchmark", "")).strip().lower()
            if norm_filter and rec_ds != norm_filter:
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
        ds_cfg = cfg.get("datasets", {}).get(str(benchmark).strip().lower(), {})
        preferred_order: List[str] = list(ds_cfg.get("models", []))

        # dedupe by model, keep first occurrence
        seen_models: set[str] = set()
        unique_items: List[Dict[str, Any]] = []
        for it in items:
            m = str(it.get("model_name"))
            if m in seen_models:
                continue
            seen_models.add(m)
            unique_items.append(it)

        # Sort unique_items by preferred order when available
        if preferred_order:
            idx = {m: i for i, m in enumerate(preferred_order)}
            unique_items.sort(key=lambda it: idx.get(str(it.get("model_name")), 1_000_000))

        if len(unique_items) < 2:
            continue

        a, b = unique_items[0], unique_items[1]
        # Randomize order to avoid systematically placing the preferred model first,
        # which would otherwise make gold_candidate always 1 when the judge is models[0]
        if random.random() < 0.5:
            a, b = b, a
        pairs.append(
            Pair(
                benchmark=benchmark,
                task_id=str(task_id),
                task_prompt=str(a.get("prompt", "")),
                code1=str(a.get("generated_code", "")),
                code2=str(b.get("generated_code", "")),
                model1=str(a.get("model_name", "")),
                model2=str(b.get("model_name", "")),
            )
        )

    return pairs


def build_messages(prompt: str, code1: str, code2: str, model1: str, model2: str, prompt_path: Path) -> List[Dict[str, str]]:
    rendered = render_prompt(prompt_path, prompt=prompt, code1=code1, code2=code2, model1=model1, model2=model2)
    return [
        {"role": "user", "content": str(rendered.get("user", "")).strip()},
    ]


def parse_choice(text: str) -> Optional[int]:
    s = (text or "").strip()
    for ch in s:
        if ch == "1":
            return 1
        if ch == "2":
            return 2
    return None


def extract_dataset_and_split(input_path: Path, data_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract dataset name and split from input path.
    E.g., data/code_generation/mbpp-sanitized/train -> ('mbpp-sanitized', 'train')
    """
    try:
        # Make path relative to data_dir/code_generation if possible
        code_generation_dir = data_dir / "code_generation"
        if input_path.is_relative_to(code_generation_dir):
            rel_path = input_path.relative_to(code_generation_dir)
            parts = rel_path.parts
            if len(parts) >= 2:
                return parts[0], parts[1]  # dataset, split
            elif len(parts) == 1:
                return parts[0], None  # dataset only
    except (ValueError, IndexError):
        pass
    return None, None


def execute(
    input_path: Optional[Path],
    dataset_folder: Optional[str],
    split: Optional[str],
    dataset: Optional[str],
    judge_model_override: Optional[str],
    concurrency_override: Optional[int],
    temperature: float,
) -> None:
    config_path = Path("configs/config.yaml")
    cfg = load_config(config_path)

    paths = cfg.get("paths", {})
    data_dir = Path(paths.get("data_dir", "data"))
    code_generation_dir = data_dir / "code_generation"
    
    # Build input path from dataset_folder and split if provided
    if dataset_folder and split:
        source_path = code_generation_dir / dataset_folder / split
        if not source_path.exists():
            raise typer.BadParameter(f"Input path not found: {source_path}")
        console.print(f"[blue]Using input path: {source_path}[/]")
    elif input_path is not None:
        source_path = Path(input_path).resolve()
        if not source_path.exists():
            raise typer.BadParameter(f"Input path not found: {source_path}")
    else:
        # If no input_path specified, look for latest timestamp directory in code_generation
        if code_generation_dir.exists():
            # Find all timestamp directories
            timestamp_dirs = [d for d in code_generation_dir.iterdir() if d.is_dir() and d.name]
            if timestamp_dirs:
                # Sort by name (timestamp) and use latest
                timestamp_dirs.sort(reverse=True)
                source_path = timestamp_dirs[0]
                console.print(f"[blue]Using latest code generation directory: {source_path}[/]")
            else:
                raise typer.BadParameter(f"No timestamp directories found in {code_generation_dir}")
        else:
            raise typer.BadParameter(f"Code generation directory not found: {code_generation_dir}")

    # Client
    client = OpenRouterClient(api_key=cfg.get("api", {}).get("openrouter_api_key") or None)
    prompt_path = Path("prompts/self_recognition.md")

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
        key = str(ds).strip().lower()
        models = list(cfg.get("datasets", {}).get(key, {}).get("models", []))
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

    # Print evaluator model(s) that will be used
    unique_judges = sorted({jm for _, _, jm in jobs})
    if unique_judges:
        if len(unique_judges) == 1:
            console.print(f"Evaluator model: {unique_judges[0]}")
        else:
            console.print(f"Evaluator models: {', '.join(unique_judges)}")

    if not jobs:
        console.print("[yellow]No valid pairs where judge model is one of the two.[/]")
        return

    total = len(jobs)
    correct = 0
    processed = 0
    skipped = len(pairs) - total

    # Determine results output path based on input structure
    paths = cfg.get("paths", {})
    data_dir = Path(paths.get("data_dir", "data"))
    self_recognition_dir = data_dir / "self_recognition"
    
    # Extract dataset and split from input path for mirrored output structure
    extracted_dataset, extracted_split = extract_dataset_and_split(source_path, data_dir)
    
    # Get evaluator model name for filename (use first unique judge model)
    evaluator_name = None
    if unique_judges:
        # Convert model name to filename-safe format (e.g., "anthropic/claude-haiku-4.5" -> "anthropic-claude-haiku-4.5")
        evaluator_name = unique_judges[0].replace("/", "-").replace(":", "-")
    
    # Build output directory to mirror input structure
    if extracted_dataset and extracted_split:
        results_subdir = self_recognition_dir / extracted_dataset / extracted_split
        results_subdir.mkdir(parents=True, exist_ok=True)
        if evaluator_name:
            results_path = results_subdir / f"{evaluator_name}.jsonl"
        else:
            results_path = results_subdir / f"self_recognition-{extracted_dataset}-{extracted_split}.jsonl"
        console.print(f"[blue]Results will be saved to: {results_subdir}[/]")
    elif extracted_dataset:
        results_subdir = self_recognition_dir / extracted_dataset
        results_subdir.mkdir(parents=True, exist_ok=True)
        ds_tag = (str(dataset).strip().lower()) if dataset else extracted_dataset
        if evaluator_name:
            results_path = results_subdir / f"{evaluator_name}.jsonl"
        else:
            results_path = results_subdir / f"self_recognition-{ds_tag}.jsonl"
    else:
        # Fallback to timestamp-based structure
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_subdir = self_recognition_dir / ts
        results_subdir.mkdir(parents=True, exist_ok=True)
        ds_tag = (str(dataset).strip().lower()) if dataset else "all"
        if evaluator_name:
            results_path = results_subdir / f"{evaluator_name}.jsonl"
        else:
            results_path = results_subdir / f"self_recognition-{ds_tag}.jsonl"

    def submit_job(job: Tuple[int, Pair, str]) -> Tuple[int, str, Optional[int], Optional[int]]:
        _, p, judge_model = job
        messages = build_messages(p.task_prompt, p.code1, p.code2, p.model1, p.model2, prompt_path)
        resp = client.generate_code(model=judge_model, messages=messages, temperature=temperature)
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
    console.print(f"\n[green]Results saved in:[/] {results_subdir}")
    console.print(f"  -> {results_path.name}")


@app.command()
def run(
    dataset_folder: Optional[str] = typer.Option(None, "--dataset-folder", help="Dataset folder name (e.g., mbpp-sanitized)"),
    split: Optional[str] = typer.Option(None, "--split", help="Dataset split (e.g., test, train, validation)"),
    evaluator: Optional[str] = typer.Option(None, "--evaluator", help="Evaluator model ID (e.g., anthropic/claude-haiku-4.5)"),
    input_path: Optional[Path] = typer.Option(None, "--input", "-i", help="Path to folder containing JSONL files (e.g., data/code_generation/mbpp-sanitized/train)"),
    dataset: Optional[str] = typer.Option(None, help="Filter to a dataset name (optional)"),
    concurrency: Optional[int] = typer.Option(None, help="Override concurrency for judge requests"),
    temperature: float = typer.Option(0.0, help="Temperature for judge model"),
):
    execute(
        input_path=input_path,
        dataset_folder=dataset_folder,
        split=split,
        dataset=dataset,
        judge_model_override=evaluator,
        concurrency_override=concurrency,
        temperature=temperature,
    )


if __name__ == "__main__":
    app()


