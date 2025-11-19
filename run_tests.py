from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from glob import glob
import multiprocessing

import typer
from rich.console import Console
from tqdm import tqdm
from src.common.types import TestOutcome

from src.lib import read_jsonl, write_jsonl_line


app = typer.Typer(add_completion=False)
console = Console()


def _execute_code_worker(
    test_imports: List[str],
    code: str,
    test_list: List[str],
    result_queue: Any,
) -> None:
    """Worker function that runs in a separate process to execute code."""
    errors: List[str] = []
    passed_count = 0
    
    try:
        # Isolated namespace for executing candidate code and tests
        globals_ns: Dict[str, Any] = {"__name__": "tested_module"}
        
        # Execute required imports first
        for imp in test_imports:
            exec(imp, globals_ns, globals_ns)

        # Load candidate solution
        exec(code, globals_ns, globals_ns)

        # Run assertions
        for t in test_list:
            try:
                exec(t, globals_ns, globals_ns)
                passed_count += 1
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        # Fatal error loading code or imports
        errors.append(f"SetupError {type(e).__name__}: {e}")
    
    # Send results back to parent process
    result_queue.put((passed_count, errors))


def run_single_record(record: Dict[str, Any], timeout_seconds: int = 5) -> TestOutcome:
    benchmark = str(record.get("benchmark", "")).strip()
    task_id = str(record.get("task_id", "")).strip()
    model_name = str(record.get("model_name", "")).strip()
    test_imports: List[str] = [str(x) for x in (record.get("test_imports") or [])]
    test_list: List[str] = [str(x) for x in (record.get("test_list") or [])]
    code = str(record.get("generated_code") or "").strip()

    errors: List[str] = []
    passed_count = 0
    
    # Use spawn context for better compatibility across platforms
    ctx = multiprocessing.get_context('spawn')
    
    # Create a queue for the worker to send results back
    result_queue = ctx.Queue()
    
    # Create and start the worker process
    process = ctx.Process(
        target=_execute_code_worker,
        args=(test_imports, code, test_list, result_queue),
    )
    process.start()
    
    # Wait for the process to complete with timeout
    process.join(timeout=timeout_seconds)
    
    if process.is_alive():
        # Process is still running - timeout occurred
        process.terminate()
        process.join(timeout=1)  # Give it 1 second to terminate gracefully
        if process.is_alive():
            process.kill()  # Force kill if still alive
            process.join()
        errors.append("TimeoutError: Code execution timed out")
    else:
        # Process completed - get results
        if not result_queue.empty():
            passed_count, errors = result_queue.get()

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


def compute_output_path(base_dir: Path, source_path: Path, model_name: str) -> Path:
    # Normalize model name for filesystem (replace / with -)
    safe_model = model_name.replace("/", "-")
    
    # Check if this is mbpp-sanitized structure
    if "mbpp-sanitized" in source_path.parts:
        # Find the index of mbpp-sanitized in the path
        parts = source_path.parts
        mbpp_idx = parts.index("mbpp-sanitized")
        
        # Get the subfolder (e.g., "prompt", "test", "train", "validation")
        if mbpp_idx + 1 < len(parts):
            subfolder = parts[mbpp_idx + 1]
            out_dir = base_dir / "tests" / "mbpp-sanitized" / subfolder
        else:
            out_dir = base_dir / "tests" / "mbpp-sanitized"
    else:
        # Extract timestamp from source directory (e.g., "20251027-170334" from "data/code_generation/20251027-170334")
        timestamp = source_path.name
        out_dir = base_dir / "tests" / timestamp
    
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"tests-{safe_model}.jsonl"


@app.command()
def run(
    input_path: Path = typer.Option(..., "--input", "-i", help="Path to input directory or JSONL file"),
    timeout: int = typer.Option(5, "--timeout", "-t", help="Timeout in seconds for code execution"),
):
    p = input_path.resolve()
    if not p.exists():
        raise typer.BadParameter(f"Input not found: {p}")

    # Collect all jsonl files to process
    jsonl_files: List[Path] = []
    
    if p.is_dir():
        # Check if this is mbpp-sanitized directory - use recursive search
        if "mbpp-sanitized" in p.parts:
            jsonl_files = [Path(f) for f in glob(str(p / "**/*.jsonl"), recursive=True)]
        else:
            # Find all jsonl files in the directory (non-recursive)
            jsonl_files = [Path(f) for f in glob(str(p / "*.jsonl"))]
        
        if not jsonl_files:
            console.print("[yellow]No JSONL files found in directory.[/]")
            raise typer.Exit(0)
    elif p.is_file() and p.suffix == ".jsonl":
        jsonl_files = [p]
    else:
        raise typer.BadParameter(f"Input must be a directory or JSONL file: {p}")

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
            
            # Use the parent directory of the JSONL file as the source path
            source_path = jsonl_file.parent
            out_path = compute_output_path(base_dir=Path("data"), source_path=source_path, model_name=model_name)
            all_model_outputs[model_name] = out_path
            
            # Aggregate stats
            model_total = 0
            model_passed = 0
            model_tests = 0
            model_tests_passed = 0

            for rec in tqdm(records, desc=f"Testing {jsonl_file.name}", unit="rec"):
                overall_stats["total_records"] += 1
                model_total += 1
                outcome = run_single_record(rec, timeout_seconds=timeout)
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


