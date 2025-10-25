from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import typer
from rich.console import Console
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.lib import write_jsonl_line, render_prompt, OpenRouterClient
from src.datasets.mbpp import load_mbpp
from src.common.types import TaskExample, GenerationRecord
from src.lib import load_config

app = typer.Typer(add_completion=False)
console = Console()


def get_dataset_registry() -> Dict[str, Callable[[int, int], List[TaskExample]]]:
    """Return a mapping from dataset key to its loader.

    New datasets can be registered here by adding an entry mapping a dataset
    identifier (CLI value) to a loader function that returns TaskExample items.
    """
    return {
        "mbpp": load_mbpp,
    }


def load_tasks(dataset: str, start_index: int, end_index: int) -> List[TaskExample]:
    registry = get_dataset_registry()
    key = dataset.lower()
    if key not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise typer.BadParameter(f"Unknown dataset '{dataset}'. Available: {available}")
    loader = registry[key]
    return loader(start_index=start_index, end_index=end_index)


def compute_output_path(base_dir: Path, source: str) -> Path:
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = base_dir / f"{source}-{ts}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def build_messages(prompt: str, gen_prompt_path: Path) -> List[Dict[str, str]]:
    rendered = render_prompt(gen_prompt_path, prompt=prompt)
    return [
        {"role": "system", "content": rendered["system"]},
        {"role": "user", "content": rendered["user"]},
    ]


def make_record(task: TaskExample, model_name: str, code: str) -> GenerationRecord:
    return GenerationRecord(task=task, model_id=model_name, generated_code=code)


def execute(config_path: Path, dataset: str, start_index: int, end_index: int) -> None:
    cfg = load_config(config_path)

    # Read models from per-dataset config
    source = dataset
    ds_cfg = cfg.get("datasets", {}).get(source, {})
    models: List[str] = list(ds_cfg.get("models", []))
    if len(models) < 2:
        raise typer.BadParameter(
            f"Config for dataset '{source}' must specify at least two models."
        )

    paths = cfg.get("paths", {})
    data_dir = Path(paths.get("data_dir", "data"))
    out = compute_output_path(data_dir, source)

    client = OpenRouterClient(api_key=cfg.get("api", {}).get("openrouter_api_key") or None)
    gen_prompt_path = Path("prompts/generation.yaml")

    tasks = load_tasks(source, start_index, end_index)
    console.print(f"Generating code for {len(tasks)} tasks using models: {models[:2]}")

    # Determine concurrency level
    max_workers = int(cfg.get("api", {}).get("concurrency", 4))

    def submit_job(task: TaskExample, model: str) -> Tuple[TaskExample, str, str]:
        messages = build_messages(task.prompt, gen_prompt_path)
        code = client.generate_code(prompt="", model=model, messages=messages)
        return task, model, code

    jobs: List[Tuple[TaskExample, str]] = [
        (task, model)
        for task in tasks
        for model in models[:2]
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(submit_job, task, model) for task, model in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating", unit="req"):
            try:
                task, model_name, code = future.result()
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]Generation failed[/]: {e}")
                continue
            record = make_record(task, model_name, code)
            write_jsonl_line(out, record.to_dict())

    console.print(f"[green]Saved[/] -> {out}")


@app.command()
def run(
    config_path: Path = typer.Option(Path("configs/config.yaml"), help="Path to config YAML"),
    dataset: str = typer.Option("mbpp", help="Dataset name (e.g., mbpp)"),
    start_index: int = typer.Option(0, help="Start index (inclusive)"),
    end_index: int = typer.Option(10, help="End index (exclusive)"),
):
    execute(config_path=config_path, dataset=dataset, start_index=start_index, end_index=end_index)


if __name__ == "__main__":
    app()
