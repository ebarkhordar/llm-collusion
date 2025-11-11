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


def build_pairs(records: Iterator[Dict[str, Any]], target_model: str, other_model: Optional[str] = None) -> List[Pair]:
    """
    Build pairs where one code is from target_model and the other is from a different model.
    """
    from collections import defaultdict

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = (str(r.get("benchmark")), str(r.get("task_id")))
        grouped[key].append(r)

    pairs: List[Pair] = []
    for (benchmark, task_id), items in grouped.items():
        # Find target model's code
        target_item = None
        other_items = []
        
        for item in items:
            if str(item.get("model_name")) == target_model:
                target_item = item
            else:
                other_items.append(item)
        
        # If user specified a particular other model, filter to it
        if other_model is not None:
            other_items = [it for it in other_items if str(it.get("model_name")) == other_model]

        # Skip if we don't have target model's code or no other models remaining
        if target_item is None or not other_items:
            continue
        
        # Create a pair with target model and first other model
        other_item = other_items[0]
        
        # Randomize order to avoid positional bias
        if random.random() < 0.5:
            code1, code2 = target_item, other_item
            model1, model2 = target_model, str(other_item.get("model_name"))
        else:
            code1, code2 = other_item, target_item
            model1, model2 = str(other_item.get("model_name")), target_model
        
        pairs.append(
            Pair(
                benchmark=benchmark,
                task_id=str(task_id),
                task_prompt=str(code1.get("prompt", "")),
                code1=str(code1.get("generated_code", "")),
                code2=str(code2.get("generated_code", "")),
                model1=model1,
                model2=model2,
            )
        )

    return pairs


def build_messages(prompt: str, code1: str, code2: str, model1: str, model2: str, target_model: str, prompt_path: Path) -> List[Dict[str, str]]:
    rendered = render_prompt(prompt_path, prompt=prompt, code1=code1, code2=code2, model1=model1, model2=model2, target_model=target_model)
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
    E.g., data/results/mbpp-sanitized/train -> ('mbpp-sanitized', 'train')
    """
    try:
        # Make path relative to data_dir/results if possible
        results_dir = data_dir / "results"
        if input_path.is_relative_to(results_dir):
            rel_path = input_path.relative_to(results_dir)
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
    judge_model: str,
    target_model: str,
    other_model: Optional[str],
    concurrency_override: Optional[int],
    temperature: float,
) -> None:
    config_path = Path("configs/config.yaml")
    cfg = load_config(config_path)

    paths = cfg.get("paths", {})
    data_dir = Path(paths.get("data_dir", "data"))
    results_dir = data_dir / "results"
    
    # Build input path from dataset_folder and split if provided
    if dataset_folder and split:
        source_path = results_dir / dataset_folder / split
        if not source_path.exists():
            raise typer.BadParameter(f"Input path not found: {source_path}")
        console.print(f"[blue]Using input path: {source_path}[/]")
    elif input_path is not None:
        source_path = Path(input_path).resolve()
        if not source_path.exists():
            raise typer.BadParameter(f"Input path not found: {source_path}")
    else:
        raise typer.BadParameter("Either --dataset-folder and --split, or --input must be provided")

    # Client
    client = OpenRouterClient(api_key=cfg.get("api", {}).get("openrouter_api_key") or None)
    prompt_path = Path("prompts/cross_model_detection.md")

    # Concurrency
    max_workers = int(concurrency_override or int(cfg.get("api", {}).get("concurrency", 4)))

    # Build pairs
    console.print(f"[blue]Building pairs with target model: {target_model}[/]")
    if other_model:
        if other_model == target_model:
            raise typer.BadParameter("--other must be different from --target")
        console.print(f"[blue]Restricting other model to: {other_model}[/]")
    records = iter_records(source_path, dataset)
    pairs = build_pairs(records, target_model, other_model)
    if not pairs:
        console.print("[yellow]No pairs found to evaluate.[/]")
        return

    console.print(f"Evaluating cross-model detection on {len(pairs)} pairs")
    console.print(f"Judge model: {judge_model}")
    console.print(f"Target model: {target_model}")
    if other_model:
        console.print(f"Other model: {other_model}")

    total = len(pairs)
    correct = 0
    processed = 0

    # Determine results output path
    cross_detection_dir = data_dir / "cross_model_detection"
    
    # Extract dataset and split from input path for mirrored output structure
    extracted_dataset, extracted_split = extract_dataset_and_split(source_path, data_dir)
    
    # Get evaluator and target model names for filename
    judge_name = judge_model.replace("/", "-").replace(":", "-")
    target_name = target_model.replace("/", "-").replace(":", "-")
    
    # Build output directory to mirror input structure
    if extracted_dataset and extracted_split:
        results_subdir = cross_detection_dir / extracted_dataset / extracted_split
        results_subdir.mkdir(parents=True, exist_ok=True)
        results_path = results_subdir / f"judge-{judge_name}_target-{target_name}.jsonl"
        console.print(f"[blue]Results will be saved to: {results_path}[/]")
    elif extracted_dataset:
        results_subdir = cross_detection_dir / extracted_dataset
        results_subdir.mkdir(parents=True, exist_ok=True)
        results_path = results_subdir / f"judge-{judge_name}_target-{target_name}.jsonl"
    else:
        # Fallback to timestamp-based structure
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_subdir = cross_detection_dir / ts
        results_subdir.mkdir(parents=True, exist_ok=True)
        results_path = results_subdir / f"judge-{judge_name}_target-{target_name}.jsonl"

    def submit_job(idx: int, pair: Pair) -> Tuple[int, str, Optional[int], Optional[int]]:
        messages = build_messages(pair.task_prompt, pair.code1, pair.code2, pair.model1, pair.model2, target_model, prompt_path)
        resp = client.generate_code(model=judge_model, messages=messages, temperature=temperature)
        choice = parse_choice(resp)
        # ground truth: which position is the target model in?
        truth: Optional[int] = 1 if target_model == pair.model1 else (2 if target_model == pair.model2 else None)
        return (idx, resp, choice, truth)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(submit_job, idx, pair) for idx, pair in enumerate(pairs)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Judging", unit="req"):
            try:
                job_idx, resp_text, choice, truth = fut.result()
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]Judge request failed[/]: {e}")
                continue

            processed += 1
            pair = pairs[job_idx]
            
            if choice is None or truth is None:
                # Still write a result row with missing values
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
        f"[green]Done.[/] Total pairs: {total}, Processed: {processed}, "
        f"Correct: {correct}, Accuracy: {acc:.3f}"
    )
    console.print(f"\n[green]Results saved in:[/] {results_path}")


@app.command()
def run(
    dataset_folder: Optional[str] = typer.Option(None, "--dataset-folder", help="Dataset folder name (e.g., mbpp-sanitized)"),
    split: Optional[str] = typer.Option(None, "--split", help="Dataset split (e.g., test, train, validation)"),
    judge: str = typer.Option(..., "--judge", help="Judge model ID (e.g., openai/gpt-5)"),
    target: str = typer.Option(..., "--target", help="Target model to identify (e.g., anthropic/claude-haiku-4.5)"),
    other: Optional[str] = typer.Option(None, "--other", help="Specific other model to pair with target (optional)"),
    input_path: Optional[Path] = typer.Option(None, "--input", "-i", help="Path to folder containing JSONL files"),
    dataset: Optional[str] = typer.Option(None, help="Filter to a dataset name (optional)"),
    concurrency: Optional[int] = typer.Option(None, help="Override concurrency for judge requests"),
    temperature: float = typer.Option(0.0, help="Temperature for judge model"),
):
    """
    Cross-model detection: Have a judge model identify code written by a specific target model.
    
    Example:
        python cross_model_detection.py --dataset-folder mbpp-sanitized --split test \\
               --judge openai/gpt-5 --target anthropic/claude-haiku-4.5
    """
    execute(
        input_path=input_path,
        dataset_folder=dataset_folder,
        split=split,
        dataset=dataset,
        judge_model=judge,
        target_model=target,
        other_model=other,
        concurrency_override=concurrency,
        temperature=temperature,
    )


if __name__ == "__main__":
    app()

