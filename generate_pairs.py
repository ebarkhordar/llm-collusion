from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import typer
from rich.console import Console
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.lib import write_jsonl_line, OpenRouterClient, load_config
from src.datasets import load_mbpp, load_humaneval, load_ds1000
from src.common.types import TaskExample, GenerationRecord
from src.generation import get_generator, extract_code_from_response

app = typer.Typer(add_completion=False)
console = Console()


def get_dataset_registry() -> Dict[str, Callable[[int, int, str], List[TaskExample]]]:
    """Return a mapping from dataset key to its loader.

    New datasets can be registered here by adding an entry mapping a dataset
    identifier (CLI value) to a loader function that returns TaskExample items.
    
    Available datasets:
        - mbpp: Mostly Basic Python Problems (974 tasks, sanitized version)
        - humaneval: OpenAI HumanEval (164 tasks, test split only)
        - ds1000: DS-1000 Data Science benchmark (1000 tasks, test split only)
    """
    return {
        "mbpp": load_mbpp,
        "humaneval": load_humaneval,
        "ds1000": load_ds1000,
    }


def load_tasks(dataset: str, start_index: int, end_index: int, split: str = "test") -> List[TaskExample]:
    registry = get_dataset_registry()
    key = dataset.lower()
    if key not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise typer.BadParameter(f"Unknown dataset '{dataset}'. Available: {available}")
    loader = registry[key]
    return loader(start_index=start_index, end_index=end_index, split=split)


def execute(dataset: str, start_index: int, end_index: int, split: str = "test") -> None:
    config_path = Path("configs/config.yaml")
    cfg = load_config(config_path)

    # Get the dataset-specific generator
    generator = get_generator(dataset)()

    # Read models from per-dataset config
    ds_cfg = cfg.get("datasets", {}).get(generator.get_dataset_key(), {})
    all_models: List[str] = list(ds_cfg.get("models", []))
    if len(all_models) < 1:
        raise typer.BadParameter(
            f"Config for dataset '{dataset}' must specify at least one model."
        )

    paths = cfg.get("paths", {})
    data_dir = Path(paths.get("data_dir", "data"))

    # Filter out models that already have output files
    models: List[str] = []
    for model in all_models:
        output_path = generator.compute_output_path(data_dir, model, split)
        if output_path.exists():
            console.print(f"[yellow]Skipping {model}[/] - output file already exists: {output_path}")
        else:
            models.append(model)
    
    if not models:
        console.print("[green]All models already have output files. Nothing to generate.[/]")
        return

    console.print(f"[green]Generating code for models:[/] {models}")

    client = OpenRouterClient(api_key=cfg.get("api", {}).get("openrouter_api_key") or None)

    tasks = load_tasks(dataset, start_index, end_index, split)
    console.print(f"Generating code for {len(tasks)} tasks from {split} split")

    # Determine concurrency level
    max_workers = int(cfg.get("api", {}).get("concurrency", 4))
    
    # Create output paths for each model
    model_outputs = {
        model: generator.compute_output_path(data_dir, model, split)
        for model in models
    }

    def submit_job(task: TaskExample, model: str) -> Tuple[TaskExample, str, str]:
        messages = generator.build_messages(task)
        response = client.generate_code(
            model=model,
            messages=messages,
            temperature=0.0,
            json_mode=True,
        )
        # Extract code from JSON response
        code = extract_code_from_response(response)
        return task, model, code

    jobs: List[Tuple[TaskExample, str]] = [
        (task, model)
        for task in tasks
        for model in models
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(submit_job, task, model) for task, model in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating", unit="req"):
            try:
                task, model_name, code = future.result()
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]Generation failed[/]: {e}")
                continue
            record = generator.make_record(task, model_name, code)
            # Write to the appropriate model-specific file
            write_jsonl_line(model_outputs[model_name], record.to_dict())

    # Print all output files and the results directory
    results_dir = model_outputs[models[0]].parent
    console.print(f"\n[green]Results saved in:[/] {results_dir}")
    for model, output_path in model_outputs.items():
        console.print(f"  {model} -> {output_path.name}")


@app.command()
def run(
    dataset: str = typer.Option("mbpp", help="Dataset name (e.g., mbpp, humaneval, ds1000)"),
    start_index: int = typer.Option(0, help="Start index (inclusive)"),
    end_index: int = typer.Option(10, help="End index (exclusive)"),
    split: str = typer.Option("test", help="Dataset split (train, test, validation, prompt)"),
):
    execute(dataset=dataset, start_index=start_index, end_index=end_index, split=split)


if __name__ == "__main__":
    app()
