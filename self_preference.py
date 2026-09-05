"""Self-preference: does an evaluator prefer its own solution when judging quality blind?

For a pair of models (model1, model2) and a judge (any model), every MBPP task where both
models produced code is shown to the judge in random order with the prompt in
prompts/model_attribution/self_preference.md. No author information is given.

Output: data/self_preference/<dataset_folder>/<split>/judge-<judge>_pair-<m1>_vs_<m2>.jsonl
Each record stores the judge's choice, which model was chosen, and each side's unit-test result
(from data/tests/) so that preference can be conditioned on actual correctness.

Usage:
  python self_preference.py --dataset-folder mbpp-sanitized --split test \
      --judge anthropic/claude-haiku-4.5 \
      --model1 anthropic/claude-haiku-4.5 --model2 deepseek/deepseek-chat-v3-0324
"""
from __future__ import annotations

import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import typer
from rich.console import Console
from tqdm import tqdm

from src.lib import OpenRouterClient, read_jsonl, render_prompt, write_jsonl_line

app = typer.Typer(add_completion=False)
console = Console()
PROMPT_PATH = Path("prompts/model_attribution/self_preference.md")


def safe(m: str) -> str:
    return m.replace("/", "-").replace(":", "-")


def load_codes(folder: Path, model: str) -> Dict[str, dict]:
    p = folder / f"{safe(model)}.jsonl"
    return {str(r["task_id"]): r for r in read_jsonl(p)} if p.exists() else {}


def load_tests(dataset_folder: str, model: str) -> Dict[str, Optional[bool]]:
    d = "mbpp-sanitized-obfuscated" if dataset_folder.endswith("obfuscated") else f"{dataset_folder}/test"
    p = Path("data/tests") / d / f"tests-{safe(model)}.jsonl"
    return {str(r["task_id"]): bool(r["passed"]) for r in read_jsonl(p)} if p.exists() else {}


def parse_choice(text: str) -> Optional[int]:
    for ch in (text or "").strip().upper():
        if ch in "A1":
            return 1
        if ch in "B2":
            return 2
    return None


@app.command()
def run(
    dataset_folder: str = typer.Option(..., "--dataset-folder"),
    split: str = typer.Option("test", "--split"),
    judge: str = typer.Option(..., "--judge"),
    model1: str = typer.Option(..., "--model1"),
    model2: str = typer.Option(..., "--model2"),
    concurrency: int = typer.Option(12, "--concurrency"),
    temperature: float = typer.Option(0.0),
    seed: int = typer.Option(42),
) -> None:
    random.seed(seed)
    src = Path("data/code_generation") / dataset_folder / split
    c1, c2 = load_codes(src, model1), load_codes(src, model2)
    t1, t2 = load_tests(dataset_folder, model1), load_tests(dataset_folder, model2)
    ids = sorted(set(c1) & set(c2), key=lambda x: (len(x), x))
    if not ids:
        raise typer.BadParameter(f"No common tasks for {model1} and {model2} in {src}")

    jobs = []
    for tid in ids:
        swap = random.random() < 0.5
        a, b = (c2[tid], c1[tid]) if swap else (c1[tid], c2[tid])
        ma, mb = (model2, model1) if swap else (model1, model2)
        jobs.append((tid, a, b, ma, mb))

    out_dir = Path("data/self_preference") / dataset_folder / split
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"judge-{safe(judge)}_pair-{safe(model1)}_vs_{safe(model2)}.jsonl"
    if out.exists():
        out.unlink()
    client = OpenRouterClient()
    console.print(f"[blue]{len(jobs)} pairs | judge={judge} | {model1} vs {model2} -> {out}[/]")

    def one(job):
        tid, a, b, ma, mb = job
        rendered = render_prompt(PROMPT_PATH, prompt=a["prompt"], code1=a["generated_code"], code2=b["generated_code"])
        resp = client.generate_code(model=judge, messages=[{"role": "user", "content": str(rendered["user"]).strip()}], temperature=temperature)
        choice = parse_choice(resp)
        chosen = None if choice is None else (ma if choice == 1 else mb)
        return dict(benchmark=a.get("benchmark"), task_id=tid, judge_model=judge, model1=model1, model2=model2,
                    candidate_1_model=ma, candidate_2_model=mb, predicted_candidate=choice, chosen_model=chosen,
                    chose_own=(chosen == judge) if chosen else None,
                    candidate_1_passed=t1.get(tid) if ma == model1 else t2.get(tid),
                    candidate_2_passed=t1.get(tid) if mb == model1 else t2.get(tid),
                    judge_response=resp)

    n = own = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for f in tqdm(as_completed(futs), total=len(futs), desc="Judging", unit="req"):
            try:
                rec = f.result()
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]request failed[/]: {e}")
                continue
            write_jsonl_line(out, rec)
            n += 1
            own += 1 if rec["chose_own"] else 0
    console.print(f"[green]Done.[/] {n} judged; chose own code {own}/{n} = {own / max(1, n):.3f}" if judge in (model1, model2) else f"[green]Done.[/] {n} judged")
    meta = dict(timestamp=datetime.now().isoformat(), task="self_preference", prompt_file=str(PROMPT_PATH),
                prompt_sha256=hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()[:12], judge_model=judge,
                model1=model1, model2=model2, temperature=temperature, seed=seed, processed=n, results_file=str(out))
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    app()
