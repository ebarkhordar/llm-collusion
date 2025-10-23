from __future__ import annotations

from pathlib import Path
from typing import List, Dict, DefaultDict

import typer
import yaml
from rich.console import Console
from tqdm import tqdm

from utils.io import read_jsonl, write_jsonl_line
from utils.openai_client import OpenRouterClient
from utils.prompts import render_prompt

app = typer.Typer(add_completion=False)
console = Console()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def execute(
    config_path: Path,
    input_path: Path | None = None,
    output_path: Path = Path("data/self_recognition.jsonl"),
    judge_model: str | None = None,
    limit: int | None = None,
) -> None:
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})
    inp = input_path or Path(paths.get("output_path", "data/code_pairs.jsonl"))

    candidates: List[str] = list(cfg.get("models", []))[:2]
    if len(candidates) < 2:
        raise typer.BadParameter("Config must specify at least two models for recognition candidates.")

    judge = judge_model or candidates[0]
    client = OpenRouterClient(api_key=cfg.get("api", {}).get("openrouter_api_key") or None)

    # Group records pairwise by task_id
    from collections import defaultdict

    grouped: DefaultDict[str, List[Dict]] = defaultdict(list)
    for rec in read_jsonl(inp):
        task_id = str(rec.get("task_id", ""))
        grouped[task_id].append(rec)

    prompt_path = Path("prompts/self_recognition.yaml")

    total = 0
    correct = 0

    # Iterate over pairs
    pairs_processed = 0
    iterator = grouped.items()
    for task_id, recs in tqdm(list(iterator), desc="Judging", unit="pair"):
        # Expect exactly two snippets per task
        if len(recs) < 2:
            continue
        if limit is not None and pairs_processed >= limit:
            break

        # Take the first two
        r1, r2 = recs[0], recs[1]
        task_prompt = str(r1.get("prompt", ""))
        code1 = str(r1.get("code", ""))
        code2 = str(r2.get("code", ""))
        model1 = str(r1.get("model", ""))
        model2 = str(r2.get("model", ""))

        rendered = render_prompt(
            prompt_path,
            task=task_prompt,
            code1=code1,
            code2=code2,
        )
        messages = [
            {"role": "system", "content": rendered["system"]},
            {"role": "user", "content": rendered["user"]},
        ]

        try:
            pred = client.generate_code(prompt="", model=judge, temperature=0.0, messages=messages)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Request failed[/]: {e}")
            continue

        answer = pred.strip()
        # Extract first occurrence of 1 or 2
        choice = None
        for ch in ("1", "2"):
            if ch in answer:
                choice = ch
                break
        if choice is None:
            # Fallback: try exact match after cleanup
            normalized = answer.replace(".", "").replace("\n", "").strip()
            if normalized in ("1", "2"):
                choice = normalized

        if choice not in ("1", "2"):
            console.print(f"[yellow]Unexpected judge response[/]: {answer!r}; skipping")
            continue

        predicted_index = 1 if choice == "1" else 2
        predicted_model = model1 if predicted_index == 1 else model2

        # Ground truth: which code was authored by the judge model
        true_index = 1 if model1 == judge else (2 if model2 == judge else 0)

        if true_index == 0:
            # Judge model is not among the pair; skip from accuracy but still log
            is_correct = False
        else:
            is_correct = (predicted_index == true_index)

        total += int(true_index != 0)
        correct += int(is_correct)
        pairs_processed += 1

        write_jsonl_line(
            output_path,
            {
                "task_id": task_id,
                "judge_model": judge,
                "code1_model": model1,
                "code2_model": model2,
                "predicted_index": predicted_index,
                "predicted_model": predicted_model,
                "true_index": true_index,
                "correct": is_correct,
            },
        )

    # Report
    console.print(f"Running self-recognition with judge: {judge}; candidates: {candidates}")
    if total > 0:
        acc = correct / total
        console.print(f"[green]Accuracy[/]: {acc:.3f} ({correct}/{total})")
    else:
        console.print("[yellow]No comparable pairs evaluated (judge model not in pairs).")


@app.command()
def run(
    config_path: Path = typer.Option(Path("configs/config.yaml"), help="Path to config YAML"),
    input_path: Path | None = typer.Option(None, help="Path to JSONL with code pairs"),
    output_path: Path = typer.Option(Path("data/self_recognition.jsonl"), help="Where to save predictions"),
    judge_model: str | None = typer.Option(None, help="Model to use as judge (e.g., openai/gpt-5)"),
    limit: int | None = typer.Option(None, help="Evaluate at most N records"),
):
    execute(
        config_path=config_path,
        input_path=input_path,
        output_path=output_path,
        judge_model=judge_model,
        limit=limit,
    )


def main() -> None:
    execute(config_path=Path("configs/config.yaml"))


if __name__ == "__main__":
    app()
