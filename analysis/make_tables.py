#!/usr/bin/env python3
"""Regenerate every table in the paper from the raw JSONL data.

Reads:
  data/tests/                      pass@1 for the 5 core models (+ obfuscated variant if present)
  data/code_generation/            original code (for heuristic baselines)
  data/code_generation_obfuscated/ R&P-obfuscated code (feature statistics)
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
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEFAULT_OUT = Path("/Users/ehsan/LatexProjects/llm-collusion-paper/latex/tables")

# Prompt hash of target_identification.md after the name-order/position decoupling fix
# (commit c8dbb1b). Runs with any other hash are excluded.
FIXED_TI_PROMPT_HASH = "6c6f4625bb59"

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


def load_code(dataset_folder: str, model: str, obfuscated: bool = False) -> Dict[str, str]:
    base = DATA / ("code_generation_obfuscated" if obfuscated else "code_generation")
    p = base / dataset_folder / "test" / f"{safe(model)}.jsonl"
    if not p.exists():
        return {}
    return {str(r["task_id"]): r["generated_code"] for r in read_jsonl(p)}


def pass_at_1(dataset_folder: str, model: str, obfuscated: bool = False) -> Optional[Tuple[int, int]]:
    d = "mbpp-sanitized-obfuscated" if obfuscated else f"{dataset_folder}/test"
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
    md = ["| Evaluator | Opponent | N | Acc | 95% CI | p | P(A) | Best heuristic | Agree w/ docstring |", "|---|---|---|---|---|---|---|---|---|"]
    tex = [
        r"\begin{tabular}{@{}llccccc@{}}",
        r"\toprule",
        r"\textbf{Evaluator} & \textbf{Other model} & \textbf{Acc. (\%)} & \textbf{95\% CI} & \textbf{P(A)} & \textbf{Best heuristic} & \textbf{Heur. acc. (\%)} \\",
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
        parsed = [r for r in recs if r["predicted_candidate"] is not None]
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
        results.append((ev, opp, n, k, k / n, lo, hi, pv, pos_a, hname, hacc[hname], agree_rate, cnt))
        summary[ev] = dict(opp=opp, n=n, acc=k / n, lo=lo, hi=hi, p=pv, pos_a=pos_a, heur=hname, heur_acc=hacc[hname], agree=agree_rate, agree_n=cnt, hacc=hacc)
    results.sort(key=lambda r: -r[4])
    for ev, opp, n, k, acc, lo, hi, pv, pos_a, hname, ha, ag, cnt in results:
        md.append(f"| {short(ev)} | {short(opp)} | {n} | {pct(acc)} | [{pct(lo)}, {pct(hi)}] | {pv:.2g} | {pct(pos_a)} | {hname} | {pct(ag)} ({cnt}) |")
        tex.append(
            f"{short(ev)} & {short(opp)} & {pct(acc)}{stars(pv)} & [{pct(lo)}, {pct(hi)}] & {pct(pos_a, 0)} & {hname} & {pct(ha)} \\\\"
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
            recs = read_jsonl(p)
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
            stats[(ds, m)] = dict(n=n_all, abstain=abst, n_own=len(own), n_oth=len(oth), tpr=tpr, fpr=fpr, tnr=tnr, bal=bal, bal_se=se, raw=raw)
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
            cells += [pct(s["tpr"], 0), pct(s["fpr"], 0), pct(s["bal"])]
        tex.append(f"{short(m)} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "ipp.tex").write_text("\n".join(tex) + "\n")
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
        parsed = [r for r in recs if r["predicted_target_code_id"] is not None]
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
        r"\textbf{Model} & orig. & R\&P & orig. & R\&P & orig. & R\&P \\",
        r"\midrule",
    ]
    for m in CORE_MODELS:
        o, b = load_code("mbpp-sanitized", m), load_code("mbpp-sanitized", m, obfuscated=True)
        if not o or not b:
            continue
        po, pb = pass_at_1("mbpp-sanitized", m), pass_at_1("mbpp-sanitized", m, obfuscated=True)
        f = lambda codes, fn: sum(fn(c) for c in codes.values()) / len(codes)
        doc_o, doc_b = f(o, has_docstring), f(b, has_docstring)
        com_o, com_b = f(o, n_comments), f(b, n_comments)
        lin_o, lin_b = f(o, n_lines), f(b, n_lines)
        p1o = pct(po[0] / po[1]) if po else "--"
        p1b = pct(pb[0] / pb[1]) if pb else "--"
        md.append(f"| {short(m)} | {p1o} | {p1b} | {pct(doc_o,0)} | {pct(doc_b,0)} | {com_o:.2f} | {com_b:.2f} | {lin_o:.1f} | {lin_b:.1f} |")
        tex.append(f"{short(m)} & {p1o} & {p1b} & {pct(doc_o,0)} & {pct(doc_b,0)} & {com_o:.2f} & {com_b:.2f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "obfuscation.tex").write_text("\n".join(tex) + "\n")

    # Heuristic-baseline accuracy before vs after obfuscation, for every pair used in Tasks 1a and 2
    md2 = ["| Pair (target vs other) | Best heuristic orig | acc orig | Best heuristic obf | acc obf |", "|---|---|---|---|---|"]
    tex2 = [
        r"\begin{tabular}{@{}llclc@{}}",
        r"\toprule",
        r"\textbf{Target vs.\ other} & \textbf{Best heuristic (orig.)} & \textbf{Acc. (\%)} & \textbf{Best heuristic (R\&P)} & \textbf{Acc. (\%)} \\",
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
        ids = [t for t in to if t in oo]
        h_o = best_heuristic((to[t], oo[t]) for t in ids)
        h_b = best_heuristic((tb[t], ob[t]) for t in ids)
        md2.append(f"| {short(target)} vs {short(other)} | {h_o[0]} | {pct(h_o[1])} | {h_b[0]} | {pct(h_b[1])} |")
        tex2.append(f"{short(target)} vs.\\ {short(other)} & {h_o[0]} & {pct(h_o[1])} & {h_b[0]} & {pct(h_b[1])} \\\\")
    tex2 += [r"\bottomrule", r"\end{tabular}"]
    (out / "obfuscation_heuristics.tex").write_text("\n".join(tex2) + "\n")
    return "\n".join(md) + "\n\n" + "\n".join(md2)


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
    json.dump(
        {"ipp": {f"{ds}|{m}": {k: v for k, v in s.items()} for (ds, m), s in ipp.items()},
         "pair": {m: {k: v for k, v in s.items() if k != "hacc"} for m, s in pair_summary.items()},
         "target_id": [{k: v for k, v in r.items() if k != "recs"} for r in runs]},
        (out / "summary.json").open("w"), indent=1, default=str,
    )
    print(f"\nWrote tables to {out}")


if __name__ == "__main__":
    main()
