from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich.console import Console
from tqdm import tqdm

from src.lib import read_jsonl, write_jsonl_line


app = typer.Typer(add_completion=False)
console = Console()


@dataclass
class TestOutcome:
    benchmark: str
    task_id: str
    model_name: str
    num_tests: int
    num_passed: int
    passed: bool
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "model_name": self.model_name,
            "num_tests": self.num_tests,
            "num_passed": self.num_passed,
            "passed": self.passed,
            "errors": self.errors,
        }


def strip_code_fences(code: str) -> str:
    s = str(code or "").strip()
    if not s.startswith("```"):
        return s

    # Remove the opening fence line (``` or ```lang)
    lines = s.splitlines()
    if lines:
        lines = lines[1:]

    # Remove a closing fence line if present at the end
    while lines and lines[-1].strip() == "":
        lines.pop()
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines)


def run_single_record(record: Dict[str, Any]) -> TestOutcome:
    benchmark = str(record.get("benchmark", "")).strip()
    task_id = str(record.get("task_id", "")).strip()
    model_name = str(record.get("model_name", "")).strip()
    test_imports: List[str] = [str(x) for x in (record.get("test_imports") or [])]
    test_list: List[str] = [str(x) for x in (record.get("test_list") or [])]
    code = strip_code_fences(record.get("generated_code") or "")

    # Isolated namespace for executing candidate code and tests
    globals_ns: Dict[str, Any] = {"__name__": "tested_module"}
    locals_ns: Dict[str, Any] = globals_ns

    errors: List[str] = []
    passed_count = 0

    try:
        # Execute required imports first
        for imp in test_imports:
            exec(imp, globals_ns, locals_ns)

        # Load candidate solution
        exec(code, globals_ns, locals_ns)

        # Run assertions
        for t in test_list:
            try:
                exec(t, globals_ns, locals_ns)
                passed_count += 1
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        # Fatal error loading code or imports; mark all tests failed
        errors.append(f"SetupError {type(e).__name__}: {e}")

    total_tests = len(test_list)
    return TestOutcome(
        benchmark=benchmark,
        task_id=task_id,
        model_name=model_name,
        num_tests=total_tests,
        num_passed=passed_count,
        passed=(passed_count == total_tests and total_tests > 0),
        errors=errors[:5],  # truncate
    )


def compute_output_path(base_dir: Path, source: str) -> Path:
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = base_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"unit_eval-{source}-{ts}.jsonl"


@app.command()
def run(
    input_path: Path = typer.Option(..., "--input", "-i", help="Path to generations JSONL"),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Where to write per-record outcomes JSONL"),
):
    p = input_path.resolve()
    if not p.exists():
        raise typer.BadParameter(f"Input JSONL not found: {p}")

    # Derive a source tag from input filename for default output naming
    source_tag = p.stem
    default_out = compute_output_path(base_dir=Path("data"), source=source_tag)
    out_path = (output_path or default_out).resolve()

    records = list(read_jsonl(p))
    if not records:
        console.print("[yellow]No records found in input.[/]")
        raise typer.Exit(0)

    # Aggregate stats
    total_records = 0
    tasks_passed = 0
    total_tests = 0
    total_tests_passed = 0
    per_model: Dict[str, Dict[str, int]] = {}

    for rec in tqdm(records, desc="Testing", unit="rec"):
        total_records += 1
        outcome = run_single_record(rec)
        write_jsonl_line(out_path, outcome.to_dict())

        total_tests += outcome.num_tests
        total_tests_passed += outcome.num_passed
        if outcome.passed:
            tasks_passed += 1

        stats = per_model.setdefault(outcome.model_name, {"records": 0, "tasks_passed": 0, "tests": 0, "tests_passed": 0})
        stats["records"] += 1
        stats["tests"] += outcome.num_tests
        stats["tests_passed"] += outcome.num_passed
        if outcome.passed:
            stats["tasks_passed"] += 1

    # Render summary
    console.print(f"[green]Saved per-record results[/] -> {out_path}")

    console.print()
    console.print("[bold]Overall[/]")
    task_acc = (tasks_passed / total_records) if total_records else 0.0
    test_acc = (total_tests_passed / total_tests) if total_tests else 0.0
    console.print(f"Tasks: {tasks_passed}/{total_records} passed ({task_acc:.3f})")
    console.print(f"Tests: {total_tests_passed}/{total_tests} passed ({test_acc:.3f})")

    if per_model:
        console.print()
        console.print("[bold]Per-model[/]")
        for model, s in per_model.items():
            m_task_acc = (s["tasks_passed"] / s["records"]) if s["records"] else 0.0
            m_test_acc = (s["tests_passed"] / s["tests"]) if s["tests"] else 0.0
            console.print(
                f"- {model}: tasks {s['tasks_passed']}/{s['records']} ({m_task_acc:.3f}), "
                f"tests {s['tests_passed']}/{s['tests']} ({m_test_acc:.3f})"
            )



if __name__ == "__main__":
    app()


