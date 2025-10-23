from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict

import typer
import yaml
from rich.console import Console
from tqdm import tqdm

from utils.io import write_jsonl_line
from utils.openai_client import OpenRouterClient

app = typer.Typer(add_completion=False)
console = Console()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_tasks(source: str, n: int) -> List[Dict[str, str]]:
    # Try to load from datasets; fall back to simple stubs
    try:
        from datasets import load_dataset  # type: ignore

        if source.lower() == "mbpp":
            ds = load_dataset("mbpp", split="test")
            items = []
            for i, ex in enumerate(ds):
                prompt = ex.get("prompt") or ex.get("text") or ex.get("task_description") or "Write a Python function."
                items.append({"task_id": f"mbpp_{i}", "prompt": str(prompt)})
                if len(items) >= n:
                    break
            return items
        else:
            # humaneval
            ds = load_dataset("openai_humaneval", split="test")
            items = []
            for i, ex in enumerate(ds):
                prompt = ex.get("prompt") or ex.get("task_id") or "Complete the function as specified."
                items.append({"task_id": f"humaneval_{i}", "prompt": str(prompt)})
                if len(items) >= n:
                    break
            return items
    except Exception:
        # Minimal fallback
        fallback = [
            {"task_id": "stub_1", "prompt": "Write a function add(a, b) that returns a + b."},
            {"task_id": "stub_2", "prompt": "Write a function is_palindrome(s) that returns True if s is a palindrome."},
            {"task_id": "stub_3", "prompt": "Write a function factorial(n) that returns n!."},
        ]
        return fallback[:n]


@app.command()
def run(
    config_path: Path = typer.Option(Path("configs/config.yaml"), help="Path to config YAML"),
    num_tasks: int | None = typer.Option(None, help="Override number of tasks"),
    output_path: Path | None = typer.Option(None, help="Override output JSONL path"),
):
    cfg = load_config(config_path)

    models: List[str] = list(cfg.get("models", []))
    if len(models) < 2:
        raise typer.BadParameter("Config must specify at least two models.")

    bench = cfg.get("benchmark", {})
    source = bench.get("source", "mbpp")
    n = num_tasks or int(bench.get("num_tasks", 5))

    paths = cfg.get("paths", {})
    out = output_path or Path(paths.get("output_path", "data/code_pairs.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)

    client = OpenRouterClient(api_key=cfg.get("api", {}).get("openrouter_api_key") or None)

    tasks = get_tasks(source, n)
    console.print(f"Generating code for {len(tasks)} tasks using models: {models[:2]}")

    for task in tqdm(tasks, desc="Generating", unit="task"):
        for model in models[:2]:
            code = client.generate_code(task["prompt"], model=model)
            record = {
                "task_id": task["task_id"],
                "prompt": task["prompt"],
                "model": model,
                "code": code,
            }
            write_jsonl_line(out, record)

    console.print(f"[green]Saved[/] -> {out}")


if __name__ == "__main__":
    app()
