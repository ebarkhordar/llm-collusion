from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict, Tuple

import typer
import yaml
from rich.console import Console
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.io import write_jsonl_line
from src.utils.prompts import render_prompt
from src.utils.openai_client import OpenRouterClient
from src.datasets.mbpp import load_mbpp

app = typer.Typer(add_completion=False)
console = Console()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_tasks(source: str, n: int) -> List[Dict[str, str]]:
    # Only MBPP for now
    if source.lower() != "mbpp":
        source = "mbpp"
    examples = load_mbpp(limit=n)
    items: List[Dict[str, str]] = []
    for ex in examples:
        items.append(
            {
                "dataset_name": ex.dataset_name,
                "dataset_task_id": ex.dataset_task_id,
                "prompt": ex.prompt,
                "test_list": ex.test_list,
                "challenge_test_list": ex.challenge_test_list,
                "test_setup_code": ex.test_setup_code,
                "reference_solution": ex.reference_solution,
            }
        )
    return items


def execute(config_path: Path, num_tasks: int | None = None, output_path: Path | None = None, concurrency: int | None = None) -> None:
    cfg = load_config(config_path)

    models: List[str] = list(cfg.get("models", []))
    if len(models) < 2:
        raise typer.BadParameter("Config must specify at least two models.")

    bench = cfg.get("benchmark", {})
    source = bench.get("source", "mbpp")
    n = num_tasks or int(bench.get("num_tasks", 10))

    paths = cfg.get("paths", {})
    default_base = Path(paths.get("output_path", "data/code_pairs.jsonl"))
    if output_path is not None:
        out = output_path
    else:
        # Attach timestamp to default filename: data/code_pairs-YYYYMMDD-HHMMSS.jsonl
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = default_base.stem  # e.g., code_pairs
        suffix = default_base.suffix or ".jsonl"
        out = default_base.with_name(f"{stem}-{ts}{suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)

    client = OpenRouterClient(api_key=cfg.get("api", {}).get("openrouter_api_key") or None)
    gen_prompt_path = Path("prompts/generation.yaml")

    tasks = get_tasks(source, n)
    console.print(f"Generating code for {len(tasks)} tasks using models: {models[:2]}")

    # Determine concurrency level
    cfg_concurrency = int(cfg.get("api", {}).get("concurrency", 4))
    max_workers = concurrency or cfg_concurrency

    def submit_job(task: Dict[str, str], model: str) -> Tuple[Dict[str, str], str, str]:
        rendered = render_prompt(gen_prompt_path, prompt=task["prompt"])
        messages = [
            {"role": "system", "content": rendered["system"]},
            {"role": "user", "content": rendered["user"]},
        ]
        code = client.generate_code(prompt="", model=model, messages=messages)
        return task, model, code

    jobs: List[Tuple[Dict[str, str], str]] = []
    for task in tasks:
        for model in models[:2]:
            jobs.append((task, model))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(submit_job, task, model) for task, model in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating", unit="req"):
            try:
                task, model_name, code = future.result()
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]Generation failed[/]: {e}")
                continue
            record = {
                "dataset_name": task["dataset_name"],
                "dataset_task_id": task["dataset_task_id"],
                "prompt": task["prompt"],
                "model_name": model_name,
                "generated_code": code,
                "test_list": task.get("test_list"),
                "challenge_test_list": task.get("challenge_test_list"),
                "test_setup_code": task.get("test_setup_code"),
                "reference_solution": task.get("reference_solution"),
            }
            write_jsonl_line(out, record)

    console.print(f"[green]Saved[/] -> {out}")


@app.command()
def run(
    config_path: Path = typer.Option(Path("configs/config.yaml"), help="Path to config YAML"),
    num_tasks: int | None = typer.Option(None, help="Override number of tasks"),
    output_path: Path | None = typer.Option(None, help="Override output JSONL path"),
    concurrency: int | None = typer.Option(None, help="Max concurrent requests"),
):
    execute(config_path=config_path, num_tasks=num_tasks, output_path=output_path, concurrency=concurrency)


def main() -> None:
    execute(config_path=Path("configs/config.yaml"))


if __name__ == "__main__":
    app()
