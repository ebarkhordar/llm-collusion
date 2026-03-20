"""Target identification: Judge identifies which code was written by a specific target model.

Task 2 in our attribution paradigm hierarchy:
  - Given two code snippets from model1 and model2
  - A judge is asked: "Which code was written by {{ target_model }}?"
  - Judge responds with '1' or '2'

Prompt: prompts/model_attribution/target_identification.md
"""
from __future__ import annotations

import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import typer
from rich.console import Console
from tqdm import tqdm

from src.lib import read_jsonl, render_prompt, OpenRouterClient, write_jsonl_line
from src.common.types import Pair, CrossModelDetectionResult
from src.lib import load_config

app = typer.Typer(add_completion=False)
console = Console()

PROMPT_PATH = Path("prompts/model_attribution/target_identification.md")


def iter_records(source_path: Path, dataset_filter: Optional[str]) -> Iterator[Dict[str, Any]]:
    """Read records from either a single JSONL file or all JSONL files in a directory."""
    norm_filter = (str(dataset_filter).strip().lower()) if dataset_filter else None

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
        for rec in read_jsonl(source_path):
            rec_ds = str(rec.get("benchmark", "")).strip().lower()
            if norm_filter and rec_ds != norm_filter:
                continue
            yield rec


def build_pairs(records: Iterator[Dict[str, Any]], model1: str, model2: str) -> List[Pair]:
    """Build pairs between two specific models."""
    from collections import defaultdict

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = (str(r.get("benchmark")), str(r.get("task_id")))
        grouped[key].append(r)

    pairs: List[Pair] = []

    for (benchmark, task_id), items in grouped.items():
        model1_item = None
        model2_item = None

        for item in items:
            model_name = str(item.get("model_name"))
            if model_name == model1:
                model1_item = item
            elif model_name == model2:
                model2_item = item

        if model1_item is None or model2_item is None:
            continue

        # Randomize order to avoid positional bias
        if random.random() < 0.5:
            code1, code2 = model1_item, model2_item
            m1, m2 = model1, model2
        else:
            code1, code2 = model2_item, model1_item
            m1, m2 = model2, model1

        pairs.append(
            Pair(
                benchmark=benchmark,
                task_id=str(task_id),
                task_prompt=str(code1.get("prompt", "")),
                code1=str(code1.get("generated_code", "")),
                code2=str(code2.get("generated_code", "")),
                model1=m1,
                model2=m2,
            )
        )

    return pairs


def build_messages(
    prompt: str, code1: str, code2: str,
    model1: str, model2: str, target_model: str,
) -> List[Dict[str, str]]:
    """Build messages for the judge."""
    rendered = render_prompt(
        PROMPT_PATH,
        prompt=prompt, code1=code1, code2=code2,
        model1=model1, model2=model2, target_model=target_model,
    )
    return [
        {"role": "user", "content": str(rendered.get("user", "")).strip()},
    ]


def parse_choice(text: str) -> Optional[int]:
    """Parse 'A'/'B' (or '1'/'2') from response and return 1 or 2."""
    s = (text or "").strip().upper()
    for ch in s:
        if ch in ("A", "1"):
            return 1
        if ch in ("B", "2"):
            return 2
    return None


def execute(
    input_path: Optional[Path],
    dataset_folder: Optional[str],
    split: Optional[str],
    dataset: Optional[str],
    judge_model: str,
    model1: str,
    model2: str,
    target_model: str,
    concurrency_override: Optional[int],
    temperature: float,
    seed: int = 42,
) -> None:
    random.seed(seed)
    config_path = Path("configs/config.yaml")
    cfg = load_config(config_path)

    paths = cfg.get("paths", {})
    data_dir = Path(paths.get("data_dir", "data"))
    code_generation_dir = data_dir / "code_generation"

    # Resolve input path
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
        raise typer.BadParameter("Either --dataset-folder and --split, or --input must be provided")

    # Validate target model
    if target_model not in (model1, model2):
        raise typer.BadParameter(f"Target model '{target_model}' must be one of model1 '{model1}' or model2 '{model2}'")

    # Client
    client = OpenRouterClient()

    # Concurrency
    max_workers = int(concurrency_override or int(cfg.get("api", {}).get("concurrency", 4)))

    # Build pairs
    console.print(f"[blue]Building pairs between {model1} and {model2}[/]")
    records = iter_records(source_path, dataset)
    pairs = build_pairs(records, model1, model2)
    if not pairs:
        console.print("[yellow]No pairs found to evaluate.[/]")
        return

    console.print(f"Evaluating target identification on {len(pairs)} pairs")
    console.print(f"Judge model: {judge_model}")
    console.print(f"Target model: {target_model}")
    console.print(f"Task: Which code was written by {target_model}?")

    total = len(pairs)
    correct = 0
    processed = 0

    # Extract dataset and split for output path
    extracted_dataset = dataset_folder
    extracted_split = split
    if not extracted_dataset:
        try:
            rel = source_path.relative_to(code_generation_dir)
            parts = rel.parts
            if len(parts) >= 2:
                extracted_dataset, extracted_split = parts[0], parts[1]
            elif len(parts) == 1:
                extracted_dataset = parts[0]
        except (ValueError, IndexError):
            pass

    # Output path
    target_id_dir = data_dir / "target_identification"
    judge_name = judge_model.replace("/", "-").replace(":", "-")
    target_name = target_model.replace("/", "-").replace(":", "-")
    m1_name = model1.replace("/", "-").replace(":", "-")
    m2_name = model2.replace("/", "-").replace(":", "-")
    filename = f"judge-{judge_name}_target-{target_name}_pair-{m1_name}_vs_{m2_name}.jsonl"

    if extracted_dataset and extracted_split:
        results_subdir = target_id_dir / extracted_dataset / extracted_split
    elif extracted_dataset:
        results_subdir = target_id_dir / extracted_dataset
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_subdir = target_id_dir / ts

    results_subdir.mkdir(parents=True, exist_ok=True)
    results_path = results_subdir / filename
    console.print(f"[blue]Results will be saved to: {results_path}[/]")

    def submit_job(idx: int, pair: Pair) -> Tuple[int, str, Optional[str]]:
        messages = build_messages(
            pair.task_prompt, pair.code1, pair.code2,
            pair.model1, pair.model2, target_model,
        )
        resp = client.generate_code(model=judge_model, messages=messages, temperature=temperature)
        choice = parse_choice(resp)
        return (idx, resp, str(choice) if choice is not None else None)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(submit_job, idx, pair) for idx, pair in enumerate(pairs)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Judging", unit="req"):
            try:
                job_idx, resp_text, predicted_code_id = fut.result()
            except Exception as e:
                console.print(f"[red]Judge request failed[/]: {e}")
                continue

            processed += 1
            pair = pairs[job_idx]

            # Determine gold answer: which code position has the target model?
            if pair.model1 == target_model:
                gold_code_id = "1"
            elif pair.model2 == target_model:
                gold_code_id = "2"
            else:
                gold_code_id = None

            # Check correctness
            is_correct: Optional[bool] = None
            if predicted_code_id is not None and gold_code_id is not None:
                is_correct = predicted_code_id == gold_code_id
                if is_correct:
                    correct += 1

            result = CrossModelDetectionResult(
                benchmark=pair.benchmark,
                task_id=pair.task_id,
                judge_model=judge_model,
                target_model=target_model,
                candidate_1_model=pair.model1,
                candidate_2_model=pair.model2,
                gold_target_code_id=gold_code_id,
                predicted_target_code_id=predicted_code_id,
                is_correct=is_correct,
                judge_response=resp_text,
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

    # Save experiment metadata
    prompt_hash = hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()[:12]
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "task": "target_identification",
        "prompt_file": str(PROMPT_PATH),
        "prompt_sha256": prompt_hash,
        "judge_model": judge_model,
        "target_model": target_model,
        "model1": model1,
        "model2": model2,
        "temperature": temperature,
        "seed": seed,
        "input_path": str(source_path),
        "total_pairs": total,
        "processed": processed,
        "correct": correct,
        "accuracy": round(acc, 4),
        "results_file": str(results_path),
    }
    meta_path = results_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2))
    console.print(f"  -> {meta_path.name}")


@app.command()
def run(
    dataset_folder: Optional[str] = typer.Option(None, "--dataset-folder", help="Dataset folder name (e.g., mbpp-sanitized, humaneval, ds1000)"),
    split: Optional[str] = typer.Option(None, "--split", help="Dataset split (e.g., test)"),
    judge: str = typer.Option(..., "--judge", help="Judge model ID (e.g., openai/gpt-5)"),
    model1: str = typer.Option(..., "--model1", help="First model in the pair"),
    model2: str = typer.Option(..., "--model2", help="Second model in the pair"),
    target: str = typer.Option(..., "--target", help="Target model to identify (must be model1 or model2)"),
    input_path: Optional[Path] = typer.Option(None, "--input", "-i", help="Path to folder containing JSONL files"),
    dataset: Optional[str] = typer.Option(None, help="Filter to a dataset name (optional)"),
    concurrency: Optional[int] = typer.Option(None, help="Override concurrency"),
    temperature: float = typer.Option(0.0, help="Temperature for judge model"),
    seed: int = typer.Option(42, help="Random seed for position randomization"),
):
    """
    Target identification: have a judge identify which code was written by a specific target model.

    Examples:
        # GPT-5 identifies DeepSeek code among Claude vs DeepSeek
        python target_identification.py run --dataset-folder mbpp-sanitized --split test \\
               --judge openai/gpt-5 \\
               --model1 anthropic/claude-haiku-4.5 --model2 deepseek/deepseek-chat-v3-0324 \\
               --target deepseek/deepseek-chat-v3-0324

        # Claude identifies GPT-5 code among GPT-5 vs Grok
        python target_identification.py run --dataset-folder humaneval --split test \\
               --judge anthropic/claude-haiku-4.5 \\
               --model1 openai/gpt-5 --model2 x-ai/grok-4-fast \\
               --target openai/gpt-5
    """
    execute(
        input_path=input_path,
        dataset_folder=dataset_folder,
        split=split,
        dataset=dataset,
        judge_model=judge,
        model1=model1,
        model2=model2,
        target_model=target,
        concurrency_override=concurrency,
        temperature=temperature,
        seed=seed,
    )


if __name__ == "__main__":
    app()
