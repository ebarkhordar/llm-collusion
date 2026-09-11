"""Carry the normalized-code judgments over to the corrected normalizer.

The September reruns on normalized code (data/*/mbpp-sanitized-obfuscated/) were made with a
normalizer version that left lambda parameters and nested function names unrenamed. This script
copies every one of those judgment files to data/*/mbpp-sanitized-normalized/ and re-judges only
the items in which either solution differs between data/code_generation_obfuscated and
data/code_generation_normalized, keeping the original A/B order, judge, and prompt. Items with an
empty solution are copied unchanged (the analysis excludes them). Self-preference records also get
their unit-test outcomes refreshed from data/tests/mbpp-sanitized-normalized.

Usage:
  python rejudge_normalized.py --dry-run
  python rejudge_normalized.py --only self_preference --only self_recognition
"""
from __future__ import annotations

import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console

from _paths import DATA, PROMPTS

from src.lib import OpenRouterClient, read_jsonl, render_prompt

app = typer.Typer(add_completion=False)
console = Console()

OLD, NEW = "mbpp-sanitized-obfuscated", "mbpp-sanitized-normalized"
PROMPT = {
    "self_recognition": PROMPTS / "model_attribution" / "self_recognition_pair.md",
    "target_identification": PROMPTS / "model_attribution" / "target_identification.md",
    "self_preference": PROMPTS / "model_attribution" / "self_preference.md",
}
JUDGE_KEY = {"self_recognition": "evaluator_model", "target_identification": "judge_model", "self_preference": "judge_model"}
RESPONSE_KEY = {"self_recognition": "evaluator_response", "target_identification": "judge_response", "self_preference": "judge_response"}


def safe(m: str) -> str:
    return m.replace("/", "-").replace(":", "-")


def load_codes(folder: str, model: str) -> Dict[str, dict]:
    p = DATA / folder / "mbpp-sanitized" / "test" / f"{safe(model)}.jsonl"
    return {str(r["task_id"]): r for r in read_jsonl(p)}


def load_tests(model: str) -> Dict[str, bool]:
    p = DATA / "tests" / NEW / f"tests-{safe(model)}.jsonl"
    return {str(r["task_id"]): bool(r["passed"]) for r in read_jsonl(p)} if p.exists() else {}


def parse_choice(text: str) -> Optional[int]:
    for ch in (text or "").strip().upper():
        if ch in "A1":
            return 1
        if ch in "B2":
            return 2
    return None


def render(task: str, rec: dict, code1: str, code2: str, prompt: str) -> str:
    kw = dict(prompt=prompt, code1=code1, code2=code2)
    if task == "target_identification":
        names = [rec["candidate_1_model"], rec["candidate_2_model"]]
        random.shuffle(names)  # name order stays independent of solution order
        kw.update(target_model=rec["target_model"], name_first=names[0], name_second=names[1])
    return str(render_prompt(PROMPT[task], **kw)["user"]).strip()


def rejudge(task: str, rec: dict, text: str, client: OpenRouterClient) -> dict:
    resp = client.generate_code(model=rec[JUDGE_KEY[task]], messages=[{"role": "user", "content": text}], temperature=0.0)
    choice = parse_choice(resp)
    out = dict(rec, rejudged=True)
    out[RESPONSE_KEY[task]] = resp
    if task == "self_recognition":
        out["predicted_candidate"] = choice
        out["is_correct"] = None if choice is None else choice == rec["gold_candidate"]
    elif task == "target_identification":
        pred = None if choice is None else str(choice)
        out["predicted_target_code_id"] = pred
        out["is_correct"] = None if pred is None else pred == rec["gold_target_code_id"]
    else:
        chosen = None if choice is None else rec[f"candidate_{choice}_model"]
        out.update(predicted_candidate=choice, chosen_model=chosen,
                   chose_own=(chosen == rec["judge_model"]) if chosen else None)
    return out


@app.command()
def run(
    dry_run: bool = typer.Option(False, "--dry-run"),
    only: List[str] = typer.Option([], "--only", help="Restrict to these tasks"),
    concurrency: int = typer.Option(8, "--concurrency"),
    seed: int = typer.Option(42),
) -> None:
    random.seed(seed)
    client = None if dry_run else OpenRouterClient()
    total_calls = 0
    for task in PROMPT:
        if only and task not in only:
            continue
        src_dir, dst_dir = DATA / task / OLD / "test", DATA / task / NEW / "test"
        for src in sorted(src_dir.glob("*.jsonl")):
            recs = list(read_jsonl(src))
            models = {m for r in recs for m in (r["candidate_1_model"], r["candidate_2_model"])}
            old = {m: load_codes("code_generation_obfuscated", m) for m in models}
            new = {m: load_codes("code_generation_normalized", m) for m in models}
            tests = {m: load_tests(m) for m in models}
            jobs, out = [], list(recs)
            for i, r in enumerate(recs):
                t, m1, m2 = str(r["task_id"]), r["candidate_1_model"], r["candidate_2_model"]
                if task == "self_preference":
                    out[i] = dict(r, candidate_1_passed=tests[m1].get(t), candidate_2_passed=tests[m2].get(t))
                changed = any(old[m][t]["generated_code"] != new[m][t]["generated_code"] for m in (m1, m2))
                empty = any(not new[m][t]["generated_code"].strip() for m in (m1, m2))
                if changed and not empty:
                    jobs.append((i, render(task, r, new[m1][t]["generated_code"], new[m2][t]["generated_code"], new[m1][t]["prompt"])))
            judge = recs[0][JUDGE_KEY[task]]
            console.print(f"{task:22s} {src.name[:90]:90s} judge={judge:32s} re-judge {len(jobs)}/{len(recs)}")
            total_calls += len(jobs)
            if dry_run:
                continue
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = {ex.submit(rejudge, task, out[i], text, client): i for i, text in jobs}
                failed = 0
                for f in as_completed(futs):
                    try:
                        out[futs[f]] = f.result()
                    except Exception as e:  # noqa: BLE001
                        failed += 1
                        console.print(f"[red]request failed[/]: {e}")
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            dst.write_text("".join(json.dumps(r) + "\n" for r in out))
            meta_src = src.with_suffix(".meta.json")
            meta = json.loads(meta_src.read_text()) if meta_src.exists() else {}
            meta.update(rejudged_from=str(src.relative_to(DATA.parent)), rejudged_items=len(jobs), rejudge_failures=failed,
                        rejudge_timestamp=datetime.now().isoformat(), rejudge_prompt_sha256=hashlib.sha256(PROMPT[task].read_bytes()).hexdigest()[:12])
            dst.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    console.print(f"[green]Total re-judge calls: {total_calls}[/]")


if __name__ == "__main__":
    app()
