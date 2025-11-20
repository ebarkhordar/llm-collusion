from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterator
from datetime import datetime

import typer
from rich.console import Console
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

from src.lib import read_jsonl, render_prompt, OpenRouterClient, write_jsonl_line
from src.common.types import Pair, CrossModelDetectionResult
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


def build_pairs(records: Iterator[Dict[str, Any]], model1: str, model2: str) -> List[Pair]:
    """
    Build pairs where one code is from model1 and the other is from model2.
    """
    from collections import defaultdict

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = (str(r.get("benchmark")), str(r.get("task_id")))
        grouped[key].append(r)

    pairs: List[Pair] = []
    for (benchmark, task_id), items in grouped.items():
        # Find both models' code
        model1_item = None
        model2_item = None
        
        for item in items:
            model_name = str(item.get("model_name"))
            if model_name == model1:
                model1_item = item
            elif model_name == model2:
                model2_item = item

        # Skip if we don't have both models' code
        if model1_item is None or model2_item is None:
            continue
        
        # Randomize order to avoid positional bias
        if random.random() < 0.5:
            code1, code2 = model1_item, model2_item
            pair_model1, pair_model2 = model1, model2
        else:
            code1, code2 = model2_item, model1_item
            pair_model1, pair_model2 = model2, model1
        
        pairs.append(
            Pair(
                benchmark=benchmark,
                task_id=str(task_id),
                task_prompt=str(code1.get("prompt", "")),
                code1=str(code1.get("generated_code", "")),
                code2=str(code2.get("generated_code", "")),
                model1=pair_model1,
                model2=pair_model2,
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
    judge_model: str,
    model1: str,
    model2: str,
    target_model: str,
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
        raise typer.BadParameter("Either --dataset-folder and --split, or --input must be provided")

    # Client
    client = OpenRouterClient(api_key=cfg.get("api", {}).get("openrouter_api_key") or None)
    prompt_path = Path("prompts/model_attribution/target_identification.md")

    # Concurrency
    max_workers = int(concurrency_override or int(cfg.get("api", {}).get("concurrency", 4)))

    # Validate that target is one of model1 or model2
    if target_model != model1 and target_model != model2:
        raise typer.BadParameter(f"--target must be one of --model1 ({model1}) or --model2 ({model2})")

    # Build pairs
    console.print(f"[blue]Building pairs: {model1} vs {model2}[/]")
    console.print(f"[blue]Target model: {target_model}[/]")
    records = iter_records(source_path, dataset)
    pairs = build_pairs(records, model1, model2)
    if not pairs:
        console.print("[yellow]No pairs found to evaluate.[/]")
        return

    console.print(f"Evaluating target identification on {len(pairs)} pairs")
    console.print(f"Judge model: {judge_model}")
    console.print(f"Model 1: {model1}")
    console.print(f"Model 2: {model2}")
    console.print(f"Target model: {target_model}")

    total = len(pairs)
    correct = 0
    processed = 0

    # Determine results output path
    target_identification_dir = data_dir / "target_identification"
    
    # Extract dataset and split from input path for mirrored output structure
    extracted_dataset, extracted_split = extract_dataset_and_split(source_path, data_dir)
    
    # Get evaluator and target model names for filename
    judge_name = judge_model.replace("/", "-").replace(":", "-")
    target_name = target_model.replace("/", "-").replace(":", "-")
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Build output directory to mirror input structure
    if extracted_dataset and extracted_split:
        results_subdir = target_identification_dir / extracted_dataset / extracted_split
        results_subdir.mkdir(parents=True, exist_ok=True)
        results_path = results_subdir / f"judge-{judge_name}_target-{target_name}-{timestamp}.jsonl"
        console.print(f"[blue]Results will be saved to: {results_path}[/]")
    elif extracted_dataset:
        results_subdir = target_identification_dir / extracted_dataset
        results_subdir.mkdir(parents=True, exist_ok=True)
        results_path = results_subdir / f"judge-{judge_name}_target-{target_name}-{timestamp}.jsonl"
    else:
        # Fallback to timestamp-based structure
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_subdir = target_identification_dir / ts
        results_subdir.mkdir(parents=True, exist_ok=True)
        results_path = results_subdir / f"judge-{judge_name}_target-{target_name}-{timestamp}.jsonl"

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
                result = CrossModelDetectionResult(
                    benchmark=pair.benchmark,
                    task_id=pair.task_id,
                    judge_model=judge_model,
                    target_model=target_model,
                    candidate_1_model=pair.model1,
                    candidate_2_model=pair.model2,
                    predicted_candidate=choice,
                    gold_candidate=truth,
                    is_correct=None,
                    judge_response=resp_text,
                )
                write_jsonl_line(results_path, result.to_dict())
                continue
            
            is_correct = choice == truth
            if is_correct:
                correct += 1
            
            # Persist result row
            result = CrossModelDetectionResult(
                benchmark=pair.benchmark,
                task_id=pair.task_id,
                judge_model=judge_model,
                target_model=target_model,
                candidate_1_model=pair.model1,
                candidate_2_model=pair.model2,
                predicted_candidate=choice,
                gold_candidate=truth,
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


@app.command()
def interactive():
    """
    Interactive mode: Prompts for all required parameters.
    """
    console.print("[cyan]=== Target Identification - Interactive Mode ===[/]\n")
    
    # Prompt for input method
    console.print("[yellow]Input method:[/]")
    console.print("1. Use dataset folder and split")
    console.print("2. Use custom input path")
    input_method = typer.prompt("Choose option (1 or 2)", default="1")
    
    dataset_folder = None
    split = None
    input_path = None
    
    if input_method == "1":
        dataset_folder = typer.prompt("Dataset folder name", default="mbpp-sanitized")
        split = typer.prompt("Split (test/train/validation)", default="test")
    else:
        input_path_str = typer.prompt("Input path (folder containing JSONL files)")
        input_path = Path(input_path_str)
    
    dataset = typer.prompt("Dataset filter (optional, press Enter to skip)", default="", show_default=False)
    dataset = dataset.strip() if dataset else None
    
    # Prompt for models
    console.print("\n[yellow]Model configuration:[/]")
    model1 = typer.prompt("Model 1 ID", default="anthropic/claude-haiku-4.5")
    model2 = typer.prompt("Model 2 ID", default="deepseek/deepseek-chat-v3-0324")
    
    # Prompt for target (must be one of model1 or model2)
    while True:
        target = typer.prompt("Target model to identify (must be model1 or model2)", default=model1)
        if target == model1 or target == model2:
            break
        console.print(f"[red]Error: Target must be either '{model1}' or '{model2}'[/]")
    
    judge = typer.prompt("Judge model ID", default="openai/gpt-5")
    
    # Optional parameters
    console.print("\n[yellow]Optional parameters:[/]")
    concurrency_str = typer.prompt("Concurrency (press Enter for default)", default="", show_default=False)
    concurrency = int(concurrency_str) if concurrency_str.strip() else None
    
    temperature_str = typer.prompt("Temperature (press Enter for 0.0)", default="0.0")
    temperature = float(temperature_str)
    
    # Confirm and execute
    console.print("\n[cyan]Configuration:[/]")
    if dataset_folder and split:
        console.print(f"  Input: {dataset_folder}/{split}")
    else:
        console.print(f"  Input: {input_path}")
    console.print(f"  Model 1: {model1}")
    console.print(f"  Model 2: {model2}")
    console.print(f"  Target: {target}")
    console.print(f"  Judge: {judge}")
    if concurrency:
        console.print(f"  Concurrency: {concurrency}")
    console.print(f"  Temperature: {temperature}")
    
    confirm = typer.confirm("\nProceed with this configuration?", default=True)
    if not confirm:
        console.print("[yellow]Cancelled.[/]")
        raise typer.Abort()
    
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
    )


@app.command()
def run(
    dataset_folder: Optional[str] = typer.Option(None, "--dataset-folder", help="Dataset folder name (e.g., mbpp-sanitized)"),
    split: Optional[str] = typer.Option(None, "--split", help="Dataset split (e.g., test, train, validation)"),
    model1: str = typer.Option(..., "--model1", help="First model ID (e.g., anthropic/claude-haiku-4.5)"),
    model2: str = typer.Option(..., "--model2", help="Second model ID (e.g., deepseek/deepseek-chat-v3-0324)"),
    target: str = typer.Option(..., "--target", help="Target model to identify (must be one of model1 or model2)"),
    judge: str = typer.Option(..., "--judge", help="Judge model ID (e.g., openai/gpt-5)"),
    input_path: Optional[Path] = typer.Option(None, "--input", "-i", help="Path to folder containing JSONL files"),
    dataset: Optional[str] = typer.Option(None, help="Filter to a dataset name (optional)"),
    concurrency: Optional[int] = typer.Option(None, help="Override concurrency for judge requests"),
    temperature: float = typer.Option(0.0, help="Temperature for judge model"),
):
    """
    Target identification: Have a judge model identify code written by a specific target model.
    
    Example:
        python target_identification.py --dataset-folder mbpp-sanitized --split test \\
               --model1 anthropic/claude-haiku-4.5 --model2 deepseek/deepseek-chat-v3-0324 \\
               --target anthropic/claude-haiku-4.5 --judge openai/gpt-5
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
    )


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    Target identification: Have a judge model identify code written by a specific target model.
    
    Run without arguments for interactive mode, or use 'run' command with arguments.
    """
    if ctx.invoked_subcommand is None:
        # No subcommand provided, run interactive mode
        ctx.invoke(interactive)


if __name__ == "__main__":
    app()

