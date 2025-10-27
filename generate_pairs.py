from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional
from collections import Counter
import ast

import typer
from rich.console import Console
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.lib import write_jsonl_line, render_prompt, OpenRouterClient
from src.datasets.mbpp import load_mbpp
from src.common.types import TaskExample, GenerationRecord
from src.lib import load_config

app = typer.Typer(add_completion=False)
console = Console()


def get_dataset_registry() -> Dict[str, Callable[[int, int], List[TaskExample]]]:
    """Return a mapping from dataset key to its loader.

    New datasets can be registered here by adding an entry mapping a dataset
    identifier (CLI value) to a loader function that returns TaskExample items.
    """
    return {
        "mbpp": load_mbpp,
    }


def load_tasks(dataset: str, start_index: int, end_index: int) -> List[TaskExample]:
    registry = get_dataset_registry()
    key = dataset.lower()
    if key not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise typer.BadParameter(f"Unknown dataset '{dataset}'. Available: {available}")
    loader = registry[key]
    return loader(start_index=start_index, end_index=end_index)


def compute_output_path(base_dir: Path, source: str, model_name: str, timestamp: str) -> Path:
    # Normalize model name for filesystem (replace / with -)
    safe_model = model_name.replace("/", "-")
    out_path = base_dir / "results" / timestamp / f"{source}-{safe_model}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def _extract_function_names_from_test(line: str) -> List[str]:
    names: List[str] = []
    try:
        tree = ast.parse(line)
    except Exception:
        return names

    class CallVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
            func = node.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            self.generic_visit(node)

    CallVisitor().visit(tree)
    return names


def _extract_function_name_from_code(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            # Return the first top-level function definition
            return node.name
    return None


def extract_function_name(task: TaskExample) -> Optional[str]:
    # Prefer names referenced in tests
    counts: Counter[str] = Counter()
    for line in task.test_list:
        for name in _extract_function_names_from_test(line):
            # Filter obvious non-solution calls
            if name in {"print", "len", "range", "int", "float", "str", "list", "set", "dict", "tuple"}:
                continue
            counts[name] += 1
    if counts:
        return counts.most_common(1)[0][0]

    # Fallback to the reference code's first function def
    return _extract_function_name_from_code(task.code)


def build_messages(task: TaskExample, gen_prompt_path: Path) -> List[Dict[str, str]]:
    fn_name = extract_function_name(task) or ""
    rendered = render_prompt(gen_prompt_path, prompt=task.prompt, function_name=fn_name)
    return [
        {"role": "user", "content": str(rendered.get("user", "")).strip()},
    ]


def make_record(task: TaskExample, model_name: str, code: str) -> GenerationRecord:
    return GenerationRecord(task=task, model_name=model_name, generated_code=code)


def execute(dataset: str, start_index: int, end_index: int) -> None:
    config_path = Path("configs/config.yaml")
    cfg = load_config(config_path)

    # Read models from per-dataset config
    source = dataset
    ds_cfg = cfg.get("datasets", {}).get(source, {})
    models: List[str] = list(ds_cfg.get("models", []))
    if len(models) < 2:
        raise typer.BadParameter(
            f"Config for dataset '{source}' must specify at least two models."
        )

    paths = cfg.get("paths", {})
    data_dir = Path(paths.get("data_dir", "data"))

    client = OpenRouterClient(api_key=cfg.get("api", {}).get("openrouter_api_key") or None)
    gen_prompt_path = Path("prompts/generation.md")

    tasks = load_tasks(source, start_index, end_index)
    console.print(f"Generating code for {len(tasks)} tasks using models: {models[:2]}")

    # Determine concurrency level
    max_workers = int(cfg.get("api", {}).get("concurrency", 4))
    
    # Create timestamp for this run
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create output paths for each model
    model_outputs = {
        model: compute_output_path(data_dir, source, model, timestamp)
        for model in models[:2]
    }

    def submit_job(task: TaskExample, model: str) -> Tuple[TaskExample, str, str]:
        messages = build_messages(task, gen_prompt_path)
        code = client.generate_code(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        return task, model, code

    jobs: List[Tuple[TaskExample, str]] = [
        (task, model)
        for task in tasks
        for model in models[:2]
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(submit_job, task, model) for task, model in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating", unit="req"):
            try:
                task, model_name, code = future.result()
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]Generation failed[/]: {e}")
                continue
            record = make_record(task, model_name, code)
            # Write to the appropriate model-specific file
            write_jsonl_line(model_outputs[model_name], record.to_dict())

    # Print all output files and the results directory
    results_dir = model_outputs[models[0]].parent
    console.print(f"\n[green]Results saved in:[/] {results_dir}")
    for model, output_path in model_outputs.items():
        console.print(f"  {model} -> {output_path.name}")


@app.command()
def run(
    dataset: str = typer.Option("mbpp", help="Dataset name (e.g., mbpp)"),
    start_index: int = typer.Option(0, help="Start index (inclusive)"),
    end_index: int = typer.Option(10, help="End index (exclusive)"),
):
    execute(dataset=dataset, start_index=start_index, end_index=end_index)


if __name__ == "__main__":
    app()
