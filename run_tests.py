from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from glob import glob

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


def compute_output_path(base_dir: Path, source_dir: str, model_name: str) -> Path:
    # Normalize model name for filesystem (replace / with -)
    safe_model = model_name.replace("/", "-")
    # Extract timestamp from source directory (e.g., "20251027-170334" from "data/results/20251027-170334")
    timestamp = Path(source_dir).name
    out_dir = base_dir / "tests" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"tests-{safe_model}.jsonl"


@app.command()
def run(
    input_path: Path = typer.Option(..., "--input", "-i", help="Path to input directory or JSONL file"),
):
    p = input_path.resolve()
    if not p.exists():
        raise typer.BadParameter(f"Input not found: {p}")

    # Collect all jsonl files to process
    jsonl_files: List[Path] = []
    
    if p.is_dir():
        # Find all jsonl files in the directory
        jsonl_files = [Path(f) for f in glob(str(p / "*.jsonl"))]
        if not jsonl_files:
            console.print("[yellow]No JSONL files found in directory.[/]")
            raise typer.Exit(0)
    elif p.is_file() and p.suffix == ".jsonl":
        jsonl_files = [p]
    else:
        raise typer.BadParameter(f"Input must be a directory or JSONL file: {p}")

    # Determine the source directory for output naming
    if p.is_dir():
        source_dir = str(p)
    else:
        # For a file, find the parent results directory
        # e.g., if input is data/results/20251027-170334/file.jsonl, source_dir should be data/results/20251027-170334
        source_dir = str(p.parent)
    
    # Process each JSONL file (one per model)
    all_model_outputs = {}
    overall_stats = {"total_records": 0, "tasks_passed": 0, "total_tests": 0, "total_tests_passed": 0}
    model_stats = {}
    
    for jsonl_file in sorted(jsonl_files):
        # Determine output path for this model
        try:
            records = list(read_jsonl(jsonl_file))
            if not records:
                console.print(f"[yellow]No records found in {jsonl_file}[/]")
                continue
            
            # Get model name from first record
            model_name = records[0].get("model_name")
            if not model_name:
                console.print(f"[red]No model_name found in {jsonl_file}[/]")
                continue
            
            out_path = compute_output_path(base_dir=Path("data"), source_dir=source_dir, model_name=model_name)
            all_model_outputs[model_name] = out_path
            
            # Aggregate stats
            per_model: Dict[str, Dict[str, int]] = {}
            model_total = 0
            model_passed = 0
            model_tests = 0
            model_tests_passed = 0

            for rec in tqdm(records, desc=f"Testing {jsonl_file.name}", unit="rec"):
                overall_stats["total_records"] += 1
                model_total += 1
                outcome = run_single_record(rec)
                write_jsonl_line(out_path, outcome.to_dict())

                overall_stats["total_tests"] += outcome.num_tests
                overall_stats["total_tests_passed"] += outcome.num_passed
                model_tests += outcome.num_tests
                model_tests_passed += outcome.num_passed
                if outcome.passed:
                    overall_stats["tasks_passed"] += 1
                    model_passed += 1
            
            # Store per-model stats
            model_stats[model_name] = {
                "records": model_total,
                "tasks_passed": model_passed,
                "tests": model_tests,
                "tests_passed": model_tests_passed
            }
        except Exception as e:
            console.print(f"[red]Error processing {jsonl_file}: {e}[/]")
            continue

    # Print overall summary
    if overall_stats["total_records"] > 0:
        console.print()
        console.print("[bold]Results:[/]")
        
        # Show per-model results
        for model, s in model_stats.items():
            m_task_acc = (s["tasks_passed"] / s["records"]) if s["records"] else 0.0
            m_test_acc = (s["tests_passed"] / s["tests"]) if s["tests"] else 0.0
            console.print(
                f"  {model}: {s['tasks_passed']}/{s['records']} tasks ({m_task_acc:.3f}), "
                f"{s['tests_passed']}/{s['tests']} tests ({m_test_acc:.3f})"
            )
        
        # Overall stats
        task_acc = (overall_stats["tasks_passed"] / overall_stats["total_records"]) if overall_stats["total_records"] else 0.0
        test_acc = (overall_stats["total_tests_passed"] / overall_stats["total_tests"]) if overall_stats["total_tests"] else 0.0
        console.print(f"\nOverall: {overall_stats['tasks_passed']}/{overall_stats['total_records']} tasks ({task_acc:.3f}), "
                     f"{overall_stats['total_tests_passed']}/{overall_stats['total_tests']} tests ({test_acc:.3f})")
        
        # Show where results were saved
        if all_model_outputs:
            results_dir = list(all_model_outputs.values())[0].parent
            console.print(f"\n[green]Saved to: {results_dir}[/]")



if __name__ == "__main__":
    app()


