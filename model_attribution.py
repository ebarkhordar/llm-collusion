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


def build_pairs(records: Iterator[Dict[str, Any]], model1: str, model2: str) -> List[Pair]:
    """
    Build pairs between two specific models only.
    """
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
        
        # Skip if we don't have both models
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


def build_messages(prompt: str, code1: str, code2: str, model1: str, model2: str, prompt_path: Path) -> List[Dict[str, str]]:
    """Build messages for the judge to classify which code belongs to which model."""
    rendered = render_prompt(prompt_path, prompt=prompt, code1=code1, code2=code2, model1=model1, model2=model2)
    return [
        {"role": "user", "content": str(rendered.get("user", "")).strip()},
    ]


def parse_attribution(text: str, model1: str, model2: str) -> Optional[int]:
    """
    Parse the model attribution response.
    Expected format (JSON):
        {
          "Code1": "model_name",
          "Code2": "model_name"
        }
    
    Returns:
        1 if Code1 is attributed to model1
        2 if Code2 is attributed to model1
        None if parsing fails
    """
    import json
    import re
    
    if not text:
        return None
    
    # Try to extract JSON from the response (handle markdown code blocks)
    text = text.strip()
    
    # Remove markdown code block markers if present
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON object directly
        json_match = re.search(r'\{.*?\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = text
    
    try:
        # Parse JSON
        data = json.loads(json_str)
        code1_model = data.get("Code1", "")
        code2_model = data.get("Code2", "")
        
        # Check if Code1 is attributed to model1
        if code1_model and model1 in code1_model:
            return 1
        elif code2_model and model1 in code2_model:
            return 2
    except json.JSONDecodeError:
        pass
    
    # Fallback: try to parse non-JSON format (Code1: model_name)
    lines = text.split('\n')
    code1_model = None
    code2_model = None
    
    for line in lines:
        line = line.strip()
        if line.lower().startswith('code1'):
            # Extract model name after "Code1:" or "Code1"
            if ':' in line:
                model_part = line.split(':', 1)[1].strip().strip('"').strip("'")
                code1_model = model_part
        elif line.lower().startswith('code2'):
            # Extract model name after "Code2:" or "Code2"
            if ':' in line:
                model_part = line.split(':', 1)[1].strip().strip('"').strip("'")
                code2_model = model_part
    
    # Check if Code1 is attributed to model1
    if code1_model and model1 in code1_model:
        return 1
    elif code2_model and model1 in code2_model:
        return 2
    
    # Last fallback: try to find model names anywhere in the response
    text_lower = text.lower()
    model1_simplified = model1.lower().split('/')[-1]  # e.g., "gpt-5" from "openai/gpt-5"
    model2_simplified = model2.lower().split('/')[-1]  # e.g., "claude-haiku-4.5" from "anthropic/claude-haiku-4.5"
    
    if 'code1' in text_lower and model1_simplified in text_lower:
        code1_pos = text_lower.index('code1')
        model1_pos = text_lower.index(model1_simplified)
        if code1_pos < model1_pos < code1_pos + 100:  # Within 100 chars
            return 1
    if 'code2' in text_lower and model1_simplified in text_lower:
        code2_pos = text_lower.index('code2')
        model1_pos = text_lower.index(model1_simplified)
        if code2_pos < model1_pos < code2_pos + 100:  # Within 100 chars
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
    model1: str,
    model2: str,
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
    prompt_path = Path("prompts/model_attribution.md")

    # Concurrency
    max_workers = int(concurrency_override or int(cfg.get("api", {}).get("concurrency", 4)))

    # Build pairs
    console.print(f"[blue]Building pairs between {model1} and {model2}[/]")
    records = iter_records(source_path, dataset)
    pairs = build_pairs(records, model1, model2)
    if not pairs:
        console.print("[yellow]No pairs found to evaluate.[/]")
        return

    console.print(f"Evaluating model attribution on {len(pairs)} pairs")
    console.print(f"Judge model: {judge_model}")
    console.print(f"Task: Identify which code is from {model1} vs {model2}")

    total = len(pairs)
    correct = 0
    processed = 0

    # Determine results output path
    attribution_dir = data_dir / "model_attribution"
    
    # Extract dataset and split from input path for mirrored output structure
    extracted_dataset, extracted_split = extract_dataset_and_split(source_path, data_dir)
    
    # Get model names for filename
    judge_name = judge_model.replace("/", "-").replace(":", "-")
    m1_name = model1.replace("/", "-").replace(":", "-")
    m2_name = model2.replace("/", "-").replace(":", "-")
    filename = f"judge-{judge_name}_classify-{m1_name}_vs_{m2_name}.jsonl"
    
    # Build output directory to mirror input structure
    if extracted_dataset and extracted_split:
        results_subdir = attribution_dir / extracted_dataset / extracted_split
        results_subdir.mkdir(parents=True, exist_ok=True)
        results_path = results_subdir / filename
        console.print(f"[blue]Results will be saved to: {results_path}[/]")
    elif extracted_dataset:
        results_subdir = attribution_dir / extracted_dataset
        results_subdir.mkdir(parents=True, exist_ok=True)
        results_path = results_subdir / filename
    else:
        # Fallback to timestamp-based structure
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_subdir = attribution_dir / ts
        results_subdir.mkdir(parents=True, exist_ok=True)
        results_path = results_subdir / filename

    def submit_job(idx: int, pair: Pair) -> Tuple[int, str, Optional[int], Optional[int]]:
        messages = build_messages(pair.task_prompt, pair.code1, pair.code2, model1, model2, prompt_path)
        resp = client.generate_code(model=judge_model, messages=messages, temperature=temperature)
        choice = parse_attribution(resp, model1, model2)
        # ground truth: which position is model1 in?
        truth: Optional[int] = 1 if model1 == pair.model1 else (2 if model1 == pair.model2 else None)
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
    judge: str = typer.Option(..., "--judge", help="Judge model ID (e.g., openai/gpt-5, anthropic/claude-haiku-4.5)"),
    model1: str = typer.Option(..., "--model1", help="First model to classify (e.g., openai/gpt-5)"),
    model2: str = typer.Option(..., "--model2", help="Second model to classify (e.g., anthropic/claude-haiku-4.5)"),
    input_path: Optional[Path] = typer.Option(None, "--input", "-i", help="Path to folder containing JSONL files"),
    dataset: Optional[str] = typer.Option(None, help="Filter to a dataset name (optional)"),
    concurrency: Optional[int] = typer.Option(None, help="Override concurrency for judge requests"),
    temperature: float = typer.Option(0.0, help="Temperature for judge model"),
):
    """
    Model attribution: Have a judge classify which code belongs to which model.
    
    The judge is asked: "Which code is from model1 and which is from model2?"
    This is a clearer classification task than asking about a specific target model.
    
    Examples:
        # GPT-5 judge classifying GPT-5 vs Claude code
        python model_attribution.py --dataset-folder mbpp-sanitized --split test \\
               --judge openai/gpt-5 \\
               --model1 openai/gpt-5 --model2 anthropic/claude-haiku-4.5
        
        # Third-party judge (Gemini) classifying GPT-5 vs Claude code
        python model_attribution.py --dataset-folder mbpp-sanitized --split test \\
               --judge google/gemini-2.5-flash \\
               --model1 openai/gpt-5 --model2 anthropic/claude-haiku-4.5
    """
    execute(
        input_path=input_path,
        dataset_folder=dataset_folder,
        split=split,
        dataset=dataset,
        judge_model=judge,
        model1=model1,
        model2=model2,
        concurrency_override=concurrency,
        temperature=temperature,
    )


if __name__ == "__main__":
    app()

