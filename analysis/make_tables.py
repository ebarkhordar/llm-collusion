#!/usr/bin/env python3
"""Regenerate every table in the paper from the raw JSONL data.

Reads:
  data/tests/                      pass@1 for the 5 core models (+ obfuscated variant if present)
  data/code_generation/            original code (for heuristic baselines)
  data/code_generation_normalized/ normalized code (corrected normalizer)
  data/self_recognition/           pairwise self-recognition (Task 1a)
  data/self_recognition_single/    individual-presentation self-recognition (Task 1b)
  data/target_identification/      target identification (Task 2), post-fix prompt only

Writes LaTeX table fragments to --out (default: the paper's latex/tables directory)
and prints a Markdown summary to stdout.

Usage:
  .venv/bin/python analysis/make_tables.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
# Set PAPER_DIR to the paper repo to write straight into it; otherwise build/ in this repo.
PAPER_DIR = os.environ.get("PAPER_DIR")
DEFAULT_OUT = (Path(PAPER_DIR) / "latex" / "tables") if PAPER_DIR else (ROOT / "build" / "tables")

# Prompt hash of target_identification.md after the name-order/position decoupling fix
# (commit c8dbb1b). Runs with any other hash are excluded.
FIXED_TI_PROMPT_HASH = "6c6f4625bb59"
# Judgments on normalized code, carried over to the corrected normalizer by
# experiments/rejudge_normalized.py (plus runs made directly on it).
NORM_RUNS = "mbpp-sanitized-normalized"

SHORT = {
    "openai/gpt-5": "GPT-5",
    "openai/gpt-5.3-codex": "GPT-5.3-Codex",
    "openai/gpt-5.4": "GPT-5.4",
    "anthropic/claude-haiku-4.5": "Claude-Haiku-4.5",
    "anthropic/claude-opus-4.6": "Claude-Opus-4.6",
    "google/gemini-2.5-flash": "Gemini-2.5-Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini-3.1-Flash-Lite",
    "x-ai/grok-4-fast": "Grok-4-Fast",
    "x-ai/grok-code-fast-1": "Grok-Code-Fast-1",
    "deepseek/deepseek-chat-v3-0324": "DeepSeek-V3",
    "deepseek/deepseek-v3.2": "DeepSeek-V3.2",
    "mistralai/codestral-2508": "Codestral-2508",
    "qwen/qwen3-coder-next": "Qwen3-Coder-Next",
    "xiaomi/mimo-v2-pro": "MiMo-V2-Pro",
    "meta-llama/llama-4-maverick": "Llama-4-Maverick",
}
CORE_MODELS = [
    "openai/gpt-5",
    "anthropic/claude-haiku-4.5",
    "google/gemini-2.5-flash",
    "x-ai/grok-4-fast",
    "deepseek/deepseek-chat-v3-0324",
]
DATASETS = [("humaneval", "HumanEval"), ("mbpp-sanitized", "MBPP"), ("ds1000", "DS-1000")]


def short(m: str) -> str:
    return SHORT.get(m, m.split("/")[-1])


def safe(m: str) -> str:
    return m.replace("/", "-").replace(":", "-")


def read_jsonl(p: Path) -> List[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ── statistics ────────────────────────────────────────────────────────────


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return (max(0.0, c - m), min(1.0, c + m))


def binom_p(k: int, n: int, p0: float = 0.5) -> float:
    """Two-sided exact binomial test."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) * p0**i * (1 - p0) ** (n - i) for i in range(n + 1)]
    obs = pmf[k]
    return min(1.0, sum(v for v in pmf if v <= obs + 1e-15))


def holm(pvals: List[float]) -> List[float]:
    """Holm step-down adjusted p-values."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def normal_p(z: float) -> float:
    """Two-sided p-value of a standard normal statistic."""
    return math.erfc(abs(z) / math.sqrt(2))


def signed(x: float) -> str:
    """Percentage-point difference with an explicit sign and no negative zero."""
    return f"{x:+.1f}".replace("-0.0", "+0.0")


def stars(p: float) -> str:
    return "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}"


# ── code features / heuristics ────────────────────────────────────────────


def has_docstring(code: str) -> bool:
    return bool(re.search(r'""".*?"""', code, re.DOTALL) or re.search(r"'''.*?'''", code, re.DOTALL))


def n_comments(code: str) -> int:
    return sum(1 for l in code.split("\n") if l.strip().startswith("#"))


def n_lines(code: str) -> int:
    return sum(1 for l in code.split("\n") if l.strip())


def has_type_hints(code: str) -> bool:
    return bool(
        re.search(r":\s*(int|str|float|bool|list|dict|tuple|set|List|Dict|Optional|Tuple|Any|Iterable|Sequence)\b", code)
        or re.search(r"->\s*", code)
    )


# Each heuristic returns a score per code; the heuristic "picks" the code with the higher score,
# with ties counted as 0.5 (expected accuracy under random tie-breaking).
HEURISTICS: Dict[str, Callable[[str], float]] = {
    "Longer code": lambda c: float(len(c)),
    "Shorter code": lambda c: -float(len(c)),
    "Has docstring": lambda c: float(has_docstring(c)),
    "No docstring": lambda c: -float(has_docstring(c)),
    "More comments": lambda c: float(n_comments(c)),
    "Has type hints": lambda c: float(has_type_hints(c)),
    "No type hints": lambda c: -float(has_type_hints(c)),
}


def heuristic_accuracy(pairs: Iterable[Tuple[str, str]]) -> Dict[str, float]:
    """pairs: (target_code, other_code). Returns expected accuracy of each heuristic."""
    pairs = list(pairs)
    out = {}
    for name, fn in HEURISTICS.items():
        s = 0.0
        for t, o in pairs:
            a, b = fn(t), fn(o)
            s += 1.0 if a > b else 0.5 if a == b else 0.0
        out[name] = s / max(1, len(pairs))
    return out


def best_heuristic(pairs: Iterable[Tuple[str, str]]) -> Tuple[str, float]:
    acc = heuristic_accuracy(pairs)
    name = max(acc, key=acc.get)
    return name, acc[name]


# ── loaders ───────────────────────────────────────────────────────────────


@lru_cache(maxsize=None)
def load_code(dataset_folder: str, model: str, obfuscated: bool = False) -> Dict[str, str]:
    """obfuscated=True reads the normalized code (corrected normalizer). Callers must not mutate."""
    base = DATA / ("code_generation_normalized" if obfuscated else "code_generation")
    p = base / dataset_folder / "test" / f"{safe(model)}.jsonl"
    if not p.exists():
        return {}
    return {str(r["task_id"]): r["generated_code"] for r in read_jsonl(p)}


@lru_cache(maxsize=None)
def empty_ids(dataset_folder: str, model: str, obfuscated: bool = False) -> frozenset:
    """Tasks on which the model returned no code (GPT-5 exhausted its 2,000-token budget on 45)."""
    return frozenset(t for t, c in load_code(dataset_folder, model, obfuscated).items() if not c.strip())


def drop_empty(recs: List[dict], obfuscated: bool = False, dataset_folder: str = "mbpp-sanitized") -> List[dict]:
    """Drop pairwise items in which either solution is empty."""
    return [r for r in recs if not any(str(r["task_id"]) in empty_ids(dataset_folder, r[k], obfuscated)
                                       for k in ("candidate_1_model", "candidate_2_model"))]


def nonempty_ids(dataset_folder: str, models: Iterable[str], obfuscated: bool = False) -> List[str]:
    codes = [load_code(dataset_folder, m, obfuscated) for m in models]
    common = set.intersection(*(set(c) for c in codes))
    return sorted(t for t in common if all(c[t].strip() for c in codes))


def position_balanced(recs: List[dict], ev: str, key: str = "evaluator_model") -> float:
    """Mean of the accuracies with the evaluator's own solution in position A and in position B."""
    a = [r for r in recs if r["candidate_1_model"] == ev]
    b = [r for r in recs if r["candidate_2_model"] == ev]
    acc = lambda rs: sum(1 for r in rs if r["is_correct"]) / len(rs) if rs else float("nan")
    return (acc(a) + acc(b)) / 2


def pass_at_1(dataset_folder: str, model: str, obfuscated: bool = False, v1: bool = False) -> Optional[Tuple[int, int]]:
    """obfuscated=True reads the tests of the corrected normalizer (mbpp-sanitized-normalized);
    v1=True reads the tests of the normalizer version used in the September LLM reruns."""
    d = ("mbpp-sanitized-obfuscated" if v1 else "mbpp-sanitized-normalized") if obfuscated else f"{dataset_folder}/test"
    p = DATA / "tests" / d / f"tests-{safe(model)}.jsonl"
    if not p.exists():
        return None
    recs = read_jsonl(p)
    return sum(1 for r in recs if r["passed"]), len(recs)


# ── Table: code generation ────────────────────────────────────────────────


def table_codegen(out: Path) -> str:
    rows = []
    md = ["| Model | HumanEval | MBPP | DS-1000 | Overall |", "|---|---|---|---|---|"]
    for m in CORE_MODELS:
        cells, tk, tn = [], 0, 0
        for ds, _ in DATASETS:
            r = pass_at_1(ds, m)
            if r is None:
                cells.append("--")
                continue
            k, n = r
            tk += k
            tn += n
            cells.append(pct(k / n))
        overall = pct(tk / tn) if tn else "--"
        rows.append((m, cells, overall))
        md.append(f"| {short(m)} | " + " | ".join(cells) + f" | {overall} |")
    rows.sort(key=lambda r: -float(r[2]))
    tex = [
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{HumanEval} & \textbf{MBPP} & \textbf{DS-1000} & \textbf{Overall} \\",
        r"\midrule",
    ]
    for m, cells, overall in rows:
        tex.append(f"{short(m)} & " + " & ".join(cells) + f" & {overall} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "codegen.tex").write_text("\n".join(tex) + "\n")
    return "\n".join(md)


# ── Table: pairwise self-recognition ──────────────────────────────────────


def table_pair_sr(out: Path) -> Tuple[str, Dict[str, dict]]:
    d = DATA / "self_recognition" / "mbpp-sanitized" / "test"
    md = ["| Evaluator | Opponent | N | Acc | 95% CI | p | Pos-bal | P(A) | Best heuristic | Agree w/ docstring |", "|---|---|---|---|---|---|---|---|---|---|"]
    tex = [
        r"\begin{tabular}{@{}llcccccc@{}}",
        r"\toprule",
        r"\textbf{Evaluator} & \textbf{Other model} & \textbf{Acc. (\%)} & \textbf{95\% CI} & \textbf{Pos.-bal. (\%)} & \textbf{P(A)} & \textbf{Best heuristic} & \textbf{Heur. acc. (\%)} \\",
        r"\midrule",
    ]
    summary: Dict[str, dict] = {}
    results = []
    for m in CORE_MODELS:
        p = d / f"{safe(m)}.jsonl"
        if not p.exists():
            continue
        recs = read_jsonl(p)
        ev = m
        parsed = drop_empty([r for r in recs if r["predicted_candidate"] is not None])
        n = len(parsed)
        k = sum(1 for r in parsed if r["is_correct"])
        pos_a = sum(1 for r in parsed if r["predicted_candidate"] == 1) / n
        opp = next(r["candidate_1_model"] if r["candidate_2_model"] == ev else r["candidate_2_model"] for r in recs)
        own = load_code("mbpp-sanitized", ev)
        oth = load_code("mbpp-sanitized", opp)
        pairs = [(own[str(r["task_id"])], oth[str(r["task_id"])]) for r in parsed]
        hacc = heuristic_accuracy(pairs)
        hname = max(hacc, key=hacc.get)
        # agreement between the evaluator's choice and the "has docstring" heuristic
        agree, cnt = 0, 0
        for r in parsed:
            c1 = own[str(r["task_id"])] if r["candidate_1_model"] == ev else oth[str(r["task_id"])]
            c2 = own[str(r["task_id"])] if r["candidate_2_model"] == ev else oth[str(r["task_id"])]
            d1, d2 = has_docstring(c1), has_docstring(c2)
            if d1 == d2:
                continue
            cnt += 1
            pick_doc = 1 if d1 else 2
            agree += int(r["predicted_candidate"] == pick_doc)
        agree_rate = agree / cnt if cnt else float("nan")
        lo, hi = wilson(k, n)
        pv = binom_p(k, n)
        pb = position_balanced(parsed, ev)
        results.append((ev, opp, n, k, k / n, lo, hi, pv, pos_a, hname, hacc[hname], agree_rate, cnt, pb))
        summary[ev] = dict(opp=opp, n=n, acc=k / n, lo=lo, hi=hi, p=pv, pos_a=pos_a, pos_bal=pb, heur=hname, heur_acc=hacc[hname], agree=agree_rate, agree_n=cnt, hacc=hacc)
    results.sort(key=lambda r: -r[4])
    for ev, opp, n, k, acc, lo, hi, pv, pos_a, hname, ha, ag, cnt, pb in results:
        md.append(f"| {short(ev)} | {short(opp)} | {n} | {pct(acc)} | [{pct(lo)}, {pct(hi)}] | {pv:.2g} | {pct(pb)} | {pct(pos_a)} | {hname} | {pct(ag)} ({cnt}) |")
        tex.append(
            f"{short(ev)} & {short(opp)} & {pct(acc)}{stars(pv)} & [{pct(lo)}, {pct(hi)}] & {pct(pb)} & {pct(pos_a, 0)} & {hname} & {pct(ha)} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "pair_sr.tex").write_text("\n".join(tex) + "\n")
    return "\n".join(md), summary


# ── Table: IPP self-recognition ───────────────────────────────────────────


def table_ipp(out: Path) -> Tuple[str, Dict[Tuple[str, str], dict]]:
    md = ["| Evaluator | Dataset | N | abstain | P(yes\\|own) | P(yes\\|other) | Bal. acc | Raw acc |", "|---|---|---|---|---|---|---|---|"]
    stats: Dict[Tuple[str, str], dict] = {}
    for ds, dsname in DATASETS:
        for m in CORE_MODELS:
            p = DATA / "self_recognition_single" / ds / "test" / f"{safe(m)}.jsonl"
            if not p.exists():
                continue
            recs = [r for r in read_jsonl(p) if str(r["task_id"]) not in empty_ids(ds, r["code_model"])]
            n_all = len(recs)
            abst = sum(1 for r in recs if r["predicted"] is None)
            own = [r for r in recs if r["expected"] == "yes" and r["predicted"] is not None]
            oth = [r for r in recs if r["expected"] == "no" and r["predicted"] is not None]
            tpr = sum(1 for r in own if r["predicted"] == "yes") / max(1, len(own))
            fpr = sum(1 for r in oth if r["predicted"] == "yes") / max(1, len(oth))
            tnr = 1 - fpr
            bal = (tpr + tnr) / 2
            raw = sum(1 for r in recs if r["is_correct"]) / n_all
            # balanced accuracy CI via a simple normal approximation on TPR and TNR
            se = math.sqrt(tpr * (1 - tpr) / max(1, len(own)) + tnr * (1 - tnr) / max(1, len(oth))) / 2
            bal_p = normal_p((bal - 0.5) / se) if se > 0 else 1.0
            stats[(ds, m)] = dict(n=n_all, abstain=abst, n_own=len(own), n_oth=len(oth), tpr=tpr, fpr=fpr, tnr=tnr, bal=bal, bal_se=se, bal_p=bal_p, raw=raw)
            md.append(f"| {short(m)} | {dsname} | {n_all} | {abst} | {pct(tpr)} | {pct(fpr)} | {pct(bal)} ± {pct(1.96*se)} | {pct(raw)} |")
    # LaTeX: one row per model, three dataset column groups (P(yes|own), P(yes|other), BA)
    tex = [
        r"\begin{tabular}{@{}l" + "ccc" * len(DATASETS) + "@{}}",
        r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{\textbf{{{name}}}}}" for _, name in DATASETS) + r" \\",
        " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(DATASETS))),
        r"\textbf{Evaluator}" + " & yes$\\mid$own & yes$\\mid$other & BA" * len(DATASETS) + r" \\",
        r"\midrule",
    ]
    for m in CORE_MODELS:
        cells = []
        for ds, _ in DATASETS:
            s = stats.get((ds, m))
            if not s:
                cells += ["--", "--", "--"]
                continue
            cells += [pct(s["tpr"], 0), pct(s["fpr"], 0), f"{pct(s['bal'])}$\\pm${100 * 1.96 * s['bal_se']:.1f}"]
        tex.append(f"{short(m)} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "ipp.tex").write_text("\n".join(tex) + "\n")
    keys = sorted(stats)
    adj = holm([stats[k]["bal_p"] for k in keys])
    md.append("\nBalanced accuracy vs. 50% (normal approx.), Holm over all cells: " + ", ".join(
        f"{short(m)}/{ds}: p={stats[(ds, m)]['bal_p']:.2g} -> {a:.2g}{' (sig)' if a < 0.05 else ''}" for (ds, m), a in zip(keys, adj) if stats[(ds, m)]["bal_p"] < 0.05))
    return "\n".join(md), stats


# ── Table: target identification ──────────────────────────────────────────


def table_target_id(out: Path) -> Tuple[str, List[dict]]:
    d = DATA / "target_identification" / "mbpp-sanitized" / "test"
    runs = []
    for meta_p in sorted(d.glob("*.meta.json")):
        meta = json.loads(meta_p.read_text())
        if meta.get("prompt_sha256") != FIXED_TI_PROMPT_HASH:
            continue
        recs = read_jsonl(meta_p.with_suffix("").with_suffix(".jsonl"))
        parsed = drop_empty([r for r in recs if r["predicted_target_code_id"] is not None])
        n = len(parsed)
        k = sum(1 for r in parsed if r["is_correct"])
        judge, target, m1, m2 = meta["judge_model"], meta["target_model"], meta["model1"], meta["model2"]
        other = m2 if target == m1 else m1
        tcode, ocode = load_code("mbpp-sanitized", target), load_code("mbpp-sanitized", other)
        pairs = [(tcode[str(r["task_id"])], ocode[str(r["task_id"])]) for r in parsed]
        hname, hacc = best_heuristic(pairs)
        lo, hi = wilson(k, n)
        pv = binom_p(k, n)
        pos_a = sum(1 for r in parsed if r["predicted_target_code_id"] == "1") / n
        runs.append(dict(judge=judge, target=target, other=other, pair=tuple(sorted([m1, m2])), n=n, k=k, acc=k / n, lo=lo, hi=hi, p=pv, pos_a=pos_a, heur=hname, heur_acc=hacc, recs=parsed))
    # order: by pair (descending best accuracy), then judge
    pair_best = defaultdict(float)
    for r in runs:
        pair_best[r["pair"]] = max(pair_best[r["pair"]], r["acc"])
    runs.sort(key=lambda r: (-pair_best[r["pair"]], r["pair"], r["judge"], r["target"]))

    md = ["| Pair | Judge | Target | N | Acc | 95% CI | p | P(A) | Best heuristic |", "|---|---|---|---|---|---|---|---|---|"]
    tex = [
        r"\begin{tabular}{@{}llccccl@{}}",
        r"\toprule",
        r"\textbf{Pair} & \textbf{Judge} & \textbf{Target} & \textbf{Acc. (\%)} & \textbf{95\% CI} & \textbf{P(A)} & \textbf{Best heuristic (\%)} \\",
        r"\midrule",
    ]
    last_pair = None
    for r in runs:
        pair_s = f"{short(r['pair'][0])} vs.\\ {short(r['pair'][1])}"
        if last_pair is not None and r["pair"] != last_pair:
            tex.append(r"\addlinespace[2pt]")
        last_pair = r["pair"]
        md.append(f"| {short(r['pair'][0])} vs {short(r['pair'][1])} | {short(r['judge'])} | {short(r['target'])} | {r['n']} | {pct(r['acc'])} | [{pct(r['lo'])}, {pct(r['hi'])}] | {r['p']:.2g} | {pct(r['pos_a'])} | {r['heur']} ({pct(r['heur_acc'])}) |")
        tex.append(
            f"{pair_s} & {short(r['judge'])} & {short(r['target'])} & {pct(r['acc'])}{stars(r['p'])} & [{pct(r['lo'])}, {pct(r['hi'])}] & {pct(r['pos_a'], 0)} & {r['heur']} ({pct(r['heur_acc'])}) \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "target_id.tex").write_text("\n".join(tex) + "\n")
    adj = holm([r["p"] for r in runs])
    md.append("\nHolm within the table: " + ", ".join(
        f"{short(r['judge'])}->{short(r['target'])} ({short(r['other'])}): p={r['p']:.2g} -> {a:.2g}{' (sig)' if a < 0.05 else ''}" for r, a in zip(runs, adj)))
    return "\n".join(md), runs


def consistency_analysis(runs: List[dict]) -> str:
    """For (judge, pair) with both targets run: fraction of tasks where the judge's two answers
    form a consistent partition (different positions), and of those, fraction correct."""
    by = defaultdict(dict)
    for r in runs:
        by[(r["judge"], r["pair"])][r["target"]] = r
    md = ["| Judge | Pair | N | consistent | consistent & correct | consistent & inverted |", "|---|---|---|---|---|---|"]
    for (judge, pair), d in sorted(by.items()):
        if len(d) != 2:
            continue
        rx, ry = list(d.values())
        px = {r["task_id"]: r for r in rx["recs"]}
        py = {r["task_id"]: r for r in ry["recs"]}
        common = [t for t in px if t in py and px[t]["candidate_1_model"] == py[t]["candidate_1_model"]]
        cons = corr = inv = 0
        for t in common:
            a, b = px[t]["predicted_target_code_id"], py[t]["predicted_target_code_id"]
            if a != b:
                cons += 1
                if a == px[t]["gold_target_code_id"]:
                    corr += 1
                else:
                    inv += 1
        n = len(common)
        md.append(f"| {short(judge)} | {short(pair[0])} vs {short(pair[1])} | {n} | {pct(cons/n)} | {pct(corr/n)} | {pct(inv/n)} |")
    return "\n".join(md)


# ── Table: obfuscation ────────────────────────────────────────────────────


def table_obfuscation(out: Path, pair_summary: Dict[str, dict], ti_runs: List[dict]) -> str:
    md = ["| Model | pass@1 orig | pass@1 obf | docstring% orig | docstring% obf | comments/snippet orig | obf | lines orig | obf |", "|---|---|---|---|---|---|---|---|---|"]
    tex = [
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{\textbf{Pass@1 (\%)}} & \multicolumn{2}{c}{\textbf{Docstring (\%)}} & \multicolumn{2}{c}{\textbf{Comments / snippet}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"\textbf{Model} & orig. & norm. & orig. & norm. & orig. & norm. \\",
        r"\midrule",
    ]
    for m in CORE_MODELS:
        o, b = load_code("mbpp-sanitized", m), load_code("mbpp-sanitized", m, obfuscated=True)
        if not o or not b:
            continue
        po, pb = pass_at_1("mbpp-sanitized", m), pass_at_1("mbpp-sanitized", m, obfuscated=True)
        pv1 = pass_at_1("mbpp-sanitized", m, obfuscated=True, v1=True)
        # feature statistics over non-empty solutions
        f = lambda codes, fn: (lambda cs: sum(fn(c) for c in cs) / len(cs))([c for c in codes.values() if c.strip()])
        doc_o, doc_b = f(o, has_docstring), f(b, has_docstring)
        com_o, com_b = f(o, n_comments), f(b, n_comments)
        lin_o, lin_b = f(o, n_lines), f(b, n_lines)
        p1o = pct(po[0] / po[1]) if po else "--"
        p1b = pct(pb[0] / pb[1]) if pb else "--"
        p1v1 = pct(pv1[0] / pv1[1]) if pv1 else "--"
        md.append(f"| {short(m)} | {p1o} | {p1b} (v1: {p1v1}) | {pct(doc_o,0)} | {pct(doc_b,0)} | {com_o:.2f} | {com_b:.2f} | {lin_o:.1f} | {lin_b:.1f} |")
        tex.append(f"{short(m)} & {p1o} & {p1b} & {pct(doc_o,0)} & {pct(doc_b,0)} & {com_o:.2f} & {com_b:.2f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "obfuscation.tex").write_text("\n".join(tex) + "\n")

    # Heuristic-baseline accuracy before vs after obfuscation, for every pair used in Tasks 1a and 2
    md2 = ["| Pair (target vs other) | Best heuristic orig | acc orig | Best heuristic obf | acc obf |", "|---|---|---|---|---|"]
    tex2 = [
        r"\begin{tabular}{@{}llclc@{}}",
        r"\toprule",
        r"\textbf{Target vs.\ other} & \textbf{Best heuristic (orig.)} & \textbf{Acc. (\%)} & \textbf{Best heuristic (norm.)} & \textbf{Acc. (\%)} \\",
        r"\midrule",
    ]
    seen = set()
    combos = [(ev, s["opp"]) for ev, s in pair_summary.items()] + [(r["target"], r["other"]) for r in ti_runs]
    for target, other in combos:
        if (target, other) in seen:
            continue
        seen.add((target, other))
        to, oo = load_code("mbpp-sanitized", target), load_code("mbpp-sanitized", other)
        tb, ob = load_code("mbpp-sanitized", target, True), load_code("mbpp-sanitized", other, True)
        if not (to and oo and tb and ob):
            continue
        ids = [t for t in nonempty_ids("mbpp-sanitized", [target, other]) if t in tb and t in ob]
        h_o = best_heuristic((to[t], oo[t]) for t in ids)
        h_b = best_heuristic((tb[t], ob[t]) for t in ids)
        md2.append(f"| {short(target)} vs {short(other)} | {h_o[0]} | {pct(h_o[1])} | {h_b[0]} | {pct(h_b[1])} |")
        tex2.append(f"{short(target)} vs.\\ {short(other)} & {h_o[0]} & {pct(h_o[1])} & {h_b[0]} & {pct(h_b[1])} \\\\")
    tex2 += [r"\bottomrule", r"\end{tabular}"]
    (out / "obfuscation_heuristics.tex").write_text("\n".join(tex2) + "\n")
    return "\n".join(md) + "\n\n" + "\n".join(md2)


# ── Pairwise self-recognition helper (original or normalized code) ────────


def analyze_pair_file(p: Path, obfuscated: bool) -> Optional[dict]:
    recs = read_jsonl(p)
    if not recs:
        return None
    ev = recs[0]["evaluator_model"]
    parsed = drop_empty([r for r in recs if r["predicted_candidate"] is not None], obfuscated)
    n = len(parsed)
    if n == 0:
        return None
    k = sum(1 for r in parsed if r["is_correct"])
    opp = next(r["candidate_1_model"] if r["candidate_2_model"] == ev else r["candidate_2_model"] for r in recs)
    own, oth = load_code("mbpp-sanitized", ev, obfuscated), load_code("mbpp-sanitized", opp, obfuscated)
    pairs = [(own[str(r["task_id"])], oth[str(r["task_id"])]) for r in parsed if str(r["task_id"]) in own and str(r["task_id"]) in oth]
    hname, hacc = best_heuristic(pairs) if pairs else ("--", float("nan"))
    lo, hi = wilson(k, n)
    longer = heuristic_accuracy(pairs)["Longer code"] if pairs else float("nan")
    return dict(ev=ev, opp=opp, n=n, k=k, acc=k / n, lo=lo, hi=hi, p=binom_p(k, n), heur=hname, heur_acc=hacc, longer=longer,
                pos_a=sum(1 for r in parsed if r["predicted_candidate"] == 1) / n, pos_bal=position_balanced(parsed, ev),
                recs=parsed)


# ── Table: original vs. normalized code (Tasks 1a and 2) ─────────────────


def table_normalized(out: Path, ti_runs: List[dict]) -> str:
    md = ["| Evaluator | Other | Acc orig | Acc norm | Heur orig | Heur norm |", "|---|---|---|---|---|---|"]
    tex = [
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{\textbf{LLM acc. (\%)}} & \multicolumn{2}{c}{\textbf{Best heuristic (\%)}} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"\textbf{Evaluator} & \textbf{Other model} & orig. & norm. & orig. & norm. \\",
        r"\midrule",
    ]
    d_o = DATA / "self_recognition" / "mbpp-sanitized" / "test"
    d_n = DATA / "self_recognition" / NORM_RUNS / "test"
    rows = 0
    norm_pvals: List[Tuple[str, float]] = []
    for m in CORE_MODELS:
        po, pn = d_o / f"{safe(m)}.jsonl", d_n / f"{safe(m)}.jsonl"
        if not (po.exists() and pn.exists()):
            continue
        o, nrm = analyze_pair_file(po, False), analyze_pair_file(pn, True)
        if not o or not nrm:
            continue
        rows += 1
        norm_pvals.append((f"SR {short(m)} {pct(nrm['acc'])} [{pct(nrm['lo'])}, {pct(nrm['hi'])}] n={nrm['n']} pos-bal {pct(nrm['pos_bal'])} (orig pos-bal {pct(o['pos_bal'])})", nrm["p"]))
        md.append(f"| {short(m)} | {short(o['opp'])} | {pct(o['acc'])} (n={o['n']}) | {pct(nrm['acc'])} (n={nrm['n']}) | {o['heur']} {pct(o['heur_acc'])} | {nrm['heur']} {pct(nrm['heur_acc'])} |")
        tex.append(f"{short(m)} & {short(o['opp'])} & {pct(o['acc'])}{stars(o['p'])} & {pct(nrm['acc'])}{stars(nrm['p'])} & {pct(o['heur_acc'])} & {pct(nrm['heur_acc'])} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    if rows:
        (out / "pair_sr_normalized.tex").write_text("\n".join(tex) + "\n")

    # Target identification: match normalized runs to the original post-fix runs
    md2 = ["| Pair | Judge | Target | Acc orig | Acc norm | Heur orig | Heur norm |", "|---|---|---|---|---|---|---|"]
    tex2 = [
        r"\begin{tabular}{@{}lllcccc@{}}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{\textbf{Judge acc. (\%)}} & \multicolumn{2}{c}{\textbf{Best heuristic (\%)}} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"\textbf{Pair} & \textbf{Judge} & \textbf{Target} & orig. & norm. & orig. & norm. \\",
        r"\midrule",
    ]
    d = DATA / "target_identification" / NORM_RUNS / "test"
    orig = {(r["judge"], r["target"], r["pair"]): r for r in ti_runs}
    rows2 = 0
    for meta_p in sorted(d.glob("*.meta.json")) if d.exists() else []:
        meta = json.loads(meta_p.read_text())
        recs = read_jsonl(meta_p.with_suffix("").with_suffix(".jsonl"))
        parsed = drop_empty([r for r in recs if r["predicted_target_code_id"] is not None], obfuscated=True)
        n = len(parsed)
        if n == 0:
            continue
        k = sum(1 for r in parsed if r["is_correct"])
        judge, target, m1, m2 = meta["judge_model"], meta["target_model"], meta["model1"], meta["model2"]
        other = m2 if target == m1 else m1
        tcode, ocode = load_code("mbpp-sanitized", target, True), load_code("mbpp-sanitized", other, True)
        hname, hacc = best_heuristic((tcode[str(r["task_id"])], ocode[str(r["task_id"])]) for r in parsed)
        pv = binom_p(k, n)
        o = orig.get((judge, target, tuple(sorted([m1, m2]))))
        pair_s = f"{short(m1)} vs.\\ {short(m2)}" if m1 <= m2 else f"{short(m2)} vs.\\ {short(m1)}"
        oa = f"{pct(o['acc'])}{stars(o['p'])}" if o else "--"
        oh = pct(o["heur_acc"]) if o else "--"
        rows2 += 1
        lo, hi = wilson(k, n)
        norm_pvals.append((f"TI {short(judge)}->{short(target)} {pct(k / n)} [{pct(lo)}, {pct(hi)}] n={n}", pv))
        md2.append(f"| {pair_s} | {short(judge)} | {short(target)} | {pct(o['acc']) if o else '--'} | {pct(k/n)} (n={n}) | {oh} | {pct(hacc)} |")
        tex2.append(f"{pair_s} & {short(judge)} & {short(target)} & {oa} & {pct(k/n)}{stars(pv)} & {oh} & {pct(hacc)} \\\\")
    tex2 += [r"\bottomrule", r"\end{tabular}"]
    if rows2:
        (out / "target_id_normalized.tex").write_text("\n".join(tex2) + "\n")
    if norm_pvals:
        adj = holm([p for _, p in norm_pvals])
        md2.append("\nHolm-adjusted p (all normalized-code LLM results together):\n" + "\n".join(f"  {n}: p={p:.3g} -> {a:.3g}{' (sig)' if a < 0.05 else ''}" for (n, p), a in zip(norm_pvals, adj)))
        for tag in ("SR", "TI"):
            sub = [(n, p) for n, p in norm_pvals if n.startswith(tag)]
            md2.append(f"Holm within the {tag} table: " + ", ".join(f"{a:.3g}" for a in holm([p for _, p in sub])))
    return "\n".join(md) + "\n\n" + "\n".join(md2)


# ── Table: all-pairs pairwise self-recognition matrix ─────────────────────


def table_pair_matrix(out: Path) -> str:
    d = DATA / "self_recognition" / "mbpp-sanitized" / "test"
    cells: Dict[Tuple[str, str], dict] = {}
    for p in sorted(d.glob("*.jsonl")):
        if "__" in p.name:  # robustness variants (prompt paraphrase, repeat runs)
            continue
        r = analyze_pair_file(p, False)
        if r:
            cells[(r["ev"], r["opp"])] = r
    models = [m for m in CORE_MODELS if any(k[0] == m for k in cells)]
    opps = [m for m in CORE_MODELS if any(k[1] == m for k in cells)]
    md = ["| Evaluator \\ Other | " + " | ".join(short(o) for o in opps) + " |", "|---|" + "---|" * len(opps)]
    tex = [r"\begin{tabular}{@{}l" + "c" * len(opps) + "@{}}", r"\toprule",
           r"\textbf{Evaluator} $\backslash$ \textbf{Other} & " + " & ".join(rf"\textbf{{{short(o)}}}" for o in opps) + r" \\", r"\midrule"]
    for m in models:
        row_md, row_tex = [], []
        for o in opps:
            c = cells.get((m, o))
            if c is None:
                row_md.append("--"); row_tex.append("--")
            else:
                row_md.append(f"{pct(c['acc'])} (longer {pct(c['longer'])}, pos-bal {pct(c['pos_bal'])})")
                row_tex.append(f"{pct(c['acc'])}{stars(c['p'])} ({pct(c['longer'], 0)})")
        md.append(f"| {short(m)} | " + " | ".join(row_md) + " |")
        tex.append(f"{short(m)} & " + " & ".join(row_tex) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    if len(cells) > len(CORE_MODELS):
        (out / "pair_sr_matrix.tex").write_text("\n".join(tex) + "\n")
    # Correlation between evaluator accuracy and P(own solution is the longer one)
    xs, ys = [], []
    for (ev, opp), c in cells.items():
        xs.append(c["longer"])
        ys.append(c["acc"])
    def pearson(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
        return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / den if den else 0.0
    if len(xs) > 2:
        import random as _rnd
        r = pearson(xs, ys)
        rng = _rnd.Random(0)
        perm = list(ys)
        n_perm, hits = 20000, 0
        for _ in range(n_perm):
            rng.shuffle(perm)
            if abs(pearson(xs, perm)) >= abs(r):
                hits += 1
        md.append(f"\nPearson r between evaluator accuracy and P(own code longer) over {len(xs)} cells: {r:.3f} (permutation p = {hits / n_perm:.4f}, {n_perm} shuffles)")
        keys = list(cells.keys())
        loo = []
        for ev in sorted({k[0] for k in keys}):
            idx = [i for i, k in enumerate(keys) if k[0] != ev]
            loo.append((short(ev), pearson([xs[i] for i in idx], [ys[i] for i in idx])))
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        md.append(f"OLS slope of accuracy on P(own code longer): {slope:.2f}; intercept {my - slope * mx:.2f}")
        md.append("Leave-one-evaluator-out r: " + ", ".join(f"without {e}: {v:.3f}" for e, v in loo))
        ps = [cells[k]["p"] for k in keys]
        adj = holm(ps)
        md.append("Holm-adjusted within the matrix: " + ", ".join(f"{short(k[0])} vs {short(k[1])}: p={cells[k]['p']:.2g} -> {a:.2g}{' (sig)' if a < 0.05 else ''}" for k, a in zip(keys, adj)))
    return "\n".join(md)


# ── Table: self-preference ────────────────────────────────────────────────


def table_self_preference(out: Path) -> str:
    """For each (pair X vs Y, code version): P(X chosen) when X judges, when a neutral model Z judges, and when
    Y judges, on the same items in the same order. Delta = P(X|X) - P(X|Y) is the combined self-preference;
    with Z it splits into Delta_X = P(X|X) - P(X|Z) and Delta_Y = P(X|Z) - P(X|Y). Exact McNemar tests."""
    md = ["| Code | Pair (X vs Y) | Z | P(X\\|X) | P(X\\|Z) | P(X\\|Y) | Δ_X | Δ_Y | Δ | Δ test-tied | n tied | detail |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tex = [
        r"\begin{tabular}{@{}lllcccccc@{}}",
        r"\toprule",
        r"\textbf{Code} & \textbf{Pair ($X$ vs.\ $Y$)} & \textbf{Neutral $Z$} & $P(X\mid X)$ & $P(X\mid Z)$ & $P(X\mid Y)$ & $\Delta_X$ & $\Delta_Y$ & $\Delta$ \\",
        r"\midrule",
    ]

    def mcnemar(ra, rb, model):
        a = {r["task_id"]: r["chosen_model"] == model for r in ra}
        b = {r["task_id"]: r["chosen_model"] == model for r in rb}
        common = [t for t in a if t in b]
        n10 = sum(1 for t in common if a[t] and not b[t])
        n01 = sum(1 for t in common if b[t] and not a[t])
        return (binom_p(n10, n10 + n01) if n10 + n01 else 1.0), n10, n01

    def rate(recs, model, tied_only=False):
        sel = [r for r in recs if not tied_only or (r["candidate_1_passed"] == r["candidate_2_passed"])]
        return (sum(1 for r in sel if r["chosen_model"] == model) / len(sel) if sel else float("nan")), len(sel)

    rows = 0
    for ds, dsname, obf in [("mbpp-sanitized", "original", False), (NORM_RUNS, "normalized", True)]:
        d = DATA / "self_preference" / ds / "test"
        if not d.exists():
            continue
        runs: Dict[Tuple[str, str, str], List[dict]] = {}
        for p in sorted(d.glob("*.jsonl")):
            recs = drop_empty([r for r in read_jsonl(p) if r["predicted_candidate"] is not None], obf)
            if recs:
                runs[(recs[0]["judge_model"], recs[0]["model1"], recs[0]["model2"])] = recs
        for m1, m2 in sorted({(a, b) for (_, a, b) in runs}):
            rx, ry = runs.get((m1, m1, m2)), runs.get((m2, m1, m2))
            if not rx or not ry:
                continue
            neutral = [j for (j, a, b) in runs if (a, b) == (m1, m2) and j not in (m1, m2)]
            rz = runs[(neutral[0], m1, m2)] if neutral else None
            px, py = rate(rx, m1)[0], rate(ry, m1)[0]
            pxt, nt = rate(rx, m1, True)
            pyt = rate(ry, m1, True)[0]
            pv, n10, n01 = mcnemar(rx, ry, m1)
            cells = [f"{pct(px)}", "--", f"{pct(py)}", "--", "--"]
            extra = ""
            if rz:
                pz = rate(rz, m1)[0]
                pvx, *_ = mcnemar(rx, rz, m1)
                pvy, *_ = mcnemar(rz, ry, m1)
                cells = [f"{pct(px)}", f"{pct(pz)}", f"{pct(py)}", f"{signed(100*(px-pz))}{stars(pvx)}", f"{signed(100*(pz-py))}{stars(pvy)}"]
                extra = f"Z={short(neutral[0])} P(X|Z)={pct(pz)} Δ_X={100*(px-pz):+.1f} (p={pvx:.2g}) Δ_Y={100*(pz-py):+.1f} (p={pvy:.2g})"
            rows += 1
            md.append(f"| {dsname} | {short(m1)} vs {short(m2)} | {short(neutral[0]) if neutral else '--'} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]} | {100*(px-py):+.1f} (McNemar p={pv:.2g}; discordant {n10}/{n01}; n={len(rx)},{len(ry)}) | {100*(pxt-pyt):+.1f} | {nt} | {extra} |")
            tex.append(f"{dsname} & {short(m1)} vs.\\ {short(m2)} & {short(neutral[0]) if neutral else '--'} & " + " & ".join(cells) + f" & {signed(100*(px-py))}{stars(pv)} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    if rows:
        (out / "self_preference.tex").write_text("\n".join(tex) + "\n")
    return "\n".join(md)


def error_types() -> str:
    """Failure categories over the five core models' original solutions (first error of each failed task)."""
    counts: Dict[str, int] = defaultdict(int)
    n_exec = 0
    for ds, _ in DATASETS:
        for m in CORE_MODELS:
            p = DATA / "tests" / ds / "test" / f"tests-{safe(m)}.jsonl"
            if not p.exists():
                continue
            empty = empty_ids(ds, m)
            for r in read_jsonl(p):
                n_exec += 1
                if r["passed"]:
                    continue
                err = str(r["errors"][0]) if r["errors"] else ""
                if str(r["task_id"]) in empty:
                    kind = "Empty output"
                else:
                    kind = next((k for k in ["TimeoutError", "SyntaxError", "IndentationError", "AssertionError", "NameError", "TypeError"] if k in err), "Other")
                    kind = "SyntaxError" if kind == "IndentationError" else kind
                counts[kind] += 1
    total = sum(counts.values())
    lines = [f"{n_exec} executions, {total} failures"] + [f"  {k}: {v} ({100 * v / total:.1f}%)" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
    return "\n".join(lines)


# ── Table: robustness of Task 1a to prompt paraphrase and repeated runs ───


def table_robustness(out: Path) -> str:
    d = DATA / "self_recognition" / "mbpp-sanitized" / "test"
    md = ["| Evaluator | Other | Variant | Acc base | Acc variant | Item agreement |", "|---|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{@{}lllccc@{}}", r"\toprule",
           r"\textbf{Evaluator} & \textbf{Other model} & \textbf{Variant} & \textbf{Acc.\ base (\%)} & \textbf{Acc.\ variant (\%)} & \textbf{Item agreement (\%)} \\", r"\midrule"]
    rows = 0
    for p in sorted(d.glob("*__*.jsonl")):
        base_name, tag = p.name[:-6].split("__", 1)
        base_p = d / f"{base_name}.jsonl"
        if not base_p.exists():  # the original March runs are stored as <evaluator>.jsonl
            base_p = d / f"{base_name.split('_vs_')[0]}.jsonl"
        if not base_p.exists():
            continue
        v, b = analyze_pair_file(p, False), analyze_pair_file(base_p, False)
        if not v or not b:
            continue
        # Compare the chosen solution, not the letter: A/B order was re-drawn for about half the items.
        chosen = lambda r: r[f"candidate_{r['predicted_candidate']}_model"]
        bv = {r["task_id"]: r for r in v["recs"]}
        bb = {r["task_id"]: r for r in b["recs"]}
        common = [t for t in bv if t in bb]
        agree = sum(1 for t in common if chosen(bv[t]) == chosen(bb[t])) / max(1, len(common))
        same = [t for t in common if bv[t]["candidate_1_model"] == bb[t]["candidate_1_model"]]
        agree_same = sum(1 for t in same if chosen(bv[t]) == chosen(bb[t])) / max(1, len(same))
        label = {"promptv2": "paraphrased prompt", "rerun": "repeat, same prompt"}.get(tag, tag)
        rows += 1
        md.append(f"| {short(v['ev'])} | {short(v['opp'])} | {label} | {pct(b['acc'])} | {pct(v['acc'])} | {pct(agree)} ({len(common)}); same A/B order: {pct(agree_same)} ({len(same)}) |")
        tex.append(f"{short(v['ev'])} & {short(v['opp'])} & {label} & {pct(b['acc'])} & {pct(v['acc'])} & {pct(agree)} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    if rows:
        (out / "robustness.tex").write_text("\n".join(tex) + "\n")
    return "\n".join(md)


# ── main ──────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    print("## Code generation (Pass@1)\n")
    print(table_codegen(out))
    print("\n## Task 1a: pairwise self-recognition (MBPP)\n")
    md, pair_summary = table_pair_sr(out)
    print(md)
    for ev, s in pair_summary.items():
        print(f"  {short(ev)} heuristics: " + ", ".join(f"{k}={pct(v)}" for k, v in sorted(s['hacc'].items(), key=lambda kv: -kv[1])))
    print("\n## Task 1b: individual-presentation self-recognition\n")
    md, ipp = table_ipp(out)
    print(md)
    print("\n## Task 2: target identification (fixed prompt only)\n")
    md, runs = table_target_id(out)
    print(md)
    print("\n### Judge consistency across the two targets of a pair\n")
    print(consistency_analysis(runs))
    print("\n## Obfuscation\n")
    print(table_obfuscation(out, pair_summary, runs))
    print("\n## Original vs. normalized code (LLM judges)\n")
    print(table_normalized(out, runs))
    print("\n## All-pairs pairwise self-recognition (original code)\n")
    print(table_pair_matrix(out))
    print("\n## Self-preference (blind quality judgment)\n")
    print(table_self_preference(out))
    print("\n## Failure types (original code)\n")
    print(error_types())
    print("\n## Robustness of Task 1a (prompt paraphrase, repeat run)\n")
    print(table_robustness(out))
    json.dump(
        {"ipp": {f"{ds}|{m}": {k: v for k, v in s.items()} for (ds, m), s in ipp.items()},
         "pair": {m: {k: v for k, v in s.items() if k != "hacc"} for m, s in pair_summary.items()},
         "target_id": [{k: v for k, v in r.items() if k != "recs"} for r in runs]},
        (out / "summary.json").open("w"), indent=1, default=str,
    )
    print(f"\nWrote tables to {out}")


if __name__ == "__main__":
    main()
