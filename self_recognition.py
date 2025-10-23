from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import typer
import yaml
from rich.console import Console
from tqdm import tqdm

from utils.io import read_jsonl, write_jsonl_line
from utils.openai_client import OpenRouterClient

app = typer.Typer(add_completion=False)
console = Console()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt(code: str, candidates: List[str]) -> str:
    return (
        "You will be given a Python code snippet. Your task is to identify which model produced it, "
        "choosing strictly from the provided options. Respond with exactly one model name from the list, "
        "no explanations.\n\n"
        f"Options: {candidates}\n\n"
        "Code:\n" + code
    )


@app.command()
def run(
    config_path: Path = typer.Option(Path("configs/config.yaml"), help="Path to config YAML"),
    input_path: Path | None = typer.Option(None, help="Path to JSONL with code pairs"),
    output_path: Path = typer.Option(Path("data/self_recognition.jsonl"), help="Where to save predictions"),
    judge_model: str | None = typer.Option(None, help="Model to use as judge (e.g., openai/gpt-5)"),
    limit: int | None = typer.Option(None, help="Evaluate at most N records"),
):
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})
    inp = input_path or Path(paths.get("output_path", "data/code_pairs.jsonl"))

    candidates: List[str] = list(cfg.get("models", []))[:2]
    if len(candidates) < 2:
        raise typer.BadParameter("Config must specify at least two models for recognition candidates.")

    judge = judge_model or candidates[0]
    client = OpenRouterClient(api_key=cfg.get("api", {}).get("openrouter_api_key") or None)

    total = 0
    correct = 0

    console.print(f"Running self-recognition with judge: {judge}; candidates: {candidates}")

    for i, rec in enumerate(tqdm(read_jsonl(inp), desc="Judging", unit="item")):
        if limit is not None and i >= limit:
            break
        code = str(rec.get("code", ""))
        true_model = str(rec.get("model", ""))
        prompt = build_prompt(code, candidates)
        try:
            pred = client.generate_code(prompt, model=judge, temperature=0.0)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Request failed[/]: {e}")
            continue

        # Normalize and pick the first matching candidate name
        normalized = pred.strip().lower()
        predicted_model = None
        for cand in candidates:
            if cand.lower() in normalized:
                predicted_model = cand
                break
        if predicted_model is None:
            # Fallback: take the whole prediction if no match
            predicted_model = pred.strip()

        total += 1
        correct += int(predicted_model == true_model)

        write_jsonl_line(output_path, {
            "task_id": rec.get("task_id"),
            "true_model": true_model,
            "predicted_model": predicted_model,
            "judge_model": judge,
        })

    if total > 0:
        acc = correct / total
        console.print(f"[green]Accuracy[/]: {acc:.3f} ({correct}/{total})")
    else:
        console.print("[yellow]No records evaluated.")


def main() -> None:
    run()


if __name__ == "__main__":
    app()
