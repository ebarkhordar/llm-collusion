#!/usr/bin/env python3
"""Redaction & Paraphrasing (R&P) obfuscation pipeline.

Applies AST-based transformations to Python code to strip stylistic fingerprints:
  1. Remove all comments and docstrings
  2. Rename local variables and parameters to generic names (v0, v1, ...)
  3. Normalize whitespace and formatting via ast.unparse()

Usage:
  poetry run python obfuscate.py --dataset-folder mbpp-sanitized --split test
"""

import ast
import json
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import typer
from rich.console import Console
from tqdm import tqdm

console = Console()
app = typer.Typer(add_completion=False)

# ── AST Transformers ──────────────────────────────────────────────────────


class DocstringRemover(ast.NodeTransformer):
    """Remove docstrings from functions, classes, and module level."""

    def _strip_docstring(self, node: ast.AST) -> ast.AST:
        if (
            hasattr(node, "body")
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
            # If body is now empty, add a `pass` statement
            if not node.body:
                node.body = [ast.Pass()]
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        node = self._strip_docstring(node)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node = self._strip_docstring(node)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        node = self._strip_docstring(node)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node = self._strip_docstring(node)
        self.generic_visit(node)
        return node


class VariableRenamer(ast.NodeTransformer):
    """Rename local variables and function parameters to generic names (v0, v1, ...).

    Preserves:
    - Built-in names (len, range, print, etc.)
    - Top-level function names (to keep test compatibility)
    - Imported names
    - Module-level constants
    """

    BUILTINS: Set[str] = set(dir(__import__("builtins")))

    # Common standard library names to preserve
    STDLIB_NAMES: Set[str] = {
        "self", "cls", "args", "kwargs",
        "True", "False", "None",
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "StopIteration", "RuntimeError", "AttributeError", "NotImplementedError",
        "math", "re", "os", "sys", "json", "collections", "itertools", "functools",
        "defaultdict", "Counter", "deque", "OrderedDict",
        "List", "Dict", "Set", "Tuple", "Optional", "Any", "Union",
    }

    def __init__(self) -> None:
        super().__init__()
        self._mapping: Dict[str, str] = {}
        self._counter = 0
        self._top_level_names: Set[str] = set()
        self._imported_names: Set[str] = set()

    def _get_new_name(self, old_name: str) -> str:
        if old_name in self._mapping:
            return self._mapping[old_name]
        new_name = f"v{self._counter}"
        self._counter += 1
        self._mapping[old_name] = new_name
        return new_name

    def _should_rename(self, name: str) -> bool:
        """Check if a name should be renamed."""
        if name.startswith("_"):  # preserve dunder and private names
            return False
        if name in self.BUILTINS:
            return False
        if name in self.STDLIB_NAMES:
            return False
        if name in self._top_level_names:
            return False
        if name in self._imported_names:
            return False
        return True

    def visit_Module(self, node: ast.Module) -> ast.Module:
        # First pass: collect top-level function/class names and imports
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._top_level_names.add(item.name)
            elif isinstance(item, ast.ClassDef):
                self._top_level_names.add(item.name)
            elif isinstance(item, ast.Import):
                for alias in item.names:
                    name = alias.asname or alias.name
                    self._imported_names.add(name)
            elif isinstance(item, ast.ImportFrom):
                for alias in item.names:
                    name = alias.asname or alias.name
                    self._imported_names.add(name)
        self.generic_visit(node)
        return node

    def _rename_arguments(self, args: ast.arguments) -> None:
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            if self._should_rename(arg.arg):
                arg.arg = self._get_new_name(arg.arg)
        if args.vararg and self._should_rename(args.vararg.arg):
            args.vararg.arg = self._get_new_name(args.vararg.arg)
        if args.kwarg and self._should_rename(args.kwarg.arg):
            args.kwarg.arg = self._get_new_name(args.kwarg.arg)

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        # Lambda parameters must be renamed consistently with their uses in the body
        self._rename_arguments(node.args)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        # Nested class names are renamed like any other local name
        if self._should_rename(node.name):
            node.name = self._get_new_name(node.name)
        self.generic_visit(node)
        return node

    def visit_Global(self, node: ast.Global) -> ast.Global:
        node.names = [self._get_new_name(n) if self._should_rename(n) else n for n in node.names]
        return node

    visit_Nonlocal = visit_Global

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.ExceptHandler:
        if node.name and self._should_rename(node.name):
            node.name = self._get_new_name(node.name)
        self.generic_visit(node)
        return node

    def visit_keyword(self, node: ast.keyword) -> ast.keyword:
        # Keyword arguments at call sites of renamed (nested) functions
        if node.arg and node.arg in self._mapping:
            node.arg = self._mapping[node.arg]
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        # Nested function names are renamed (top-level names are preserved)
        if self._should_rename(node.name):
            node.name = self._get_new_name(node.name)
        # Rename parameters
        for arg in node.args.args:
            if self._should_rename(arg.arg):
                arg.arg = self._get_new_name(arg.arg)
        if node.args.vararg and self._should_rename(node.args.vararg.arg):
            node.args.vararg.arg = self._get_new_name(node.args.vararg.arg)
        if node.args.kwarg and self._should_rename(node.args.kwarg.arg):
            node.args.kwarg.arg = self._get_new_name(node.args.kwarg.arg)
        for arg in node.args.kwonlyargs:
            if self._should_rename(arg.arg):
                arg.arg = self._get_new_name(arg.arg)

        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if self._should_rename(node.id):
            node.id = self._get_new_name(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        # Already handled in visit_FunctionDef
        return node


class TypeAnnotationRemover(ast.NodeTransformer):
    """Remove type annotations from functions and variables."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.returns = None
        for arg in node.args.args + node.args.kwonlyargs:
            arg.annotation = None
        if node.args.vararg:
            node.args.vararg.annotation = None
        if node.args.kwarg:
            node.args.kwarg.annotation = None
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        if node.value is not None:
            return ast.Assign(targets=[node.target], value=node.value)
        return None  # Remove annotation-only statements


# ── Pipeline ──────────────────────────────────────────────────────────────


def obfuscate_code(code: str) -> str:
    """Apply the full R&P pipeline to a piece of Python code.

    Returns the obfuscated code, or the original if parsing fails.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # If code can't be parsed, return it with just comment stripping
        return strip_comments_regex(code)

    # Apply transformations in order
    tree = DocstringRemover().visit(tree)
    tree = TypeAnnotationRemover().visit(tree)
    tree = VariableRenamer().visit(tree)

    # Fix missing line numbers after transforms
    ast.fix_missing_locations(tree)

    try:
        result = ast.unparse(tree)
    except Exception:
        return strip_comments_regex(code)

    return result


def strip_comments_regex(code: str) -> str:
    """Fallback: strip # comments from code using regex (for unparseable code)."""
    lines = []
    for line in code.split("\n"):
        # Remove inline comments (but not inside strings — imperfect but good enough)
        stripped = re.sub(r"#[^'\"\n]*$", "", line).rstrip()
        if stripped:  # keep non-empty lines
            lines.append(stripped)
    return "\n".join(lines)


# ── File Processing ──────────────────────────────────────────────────────


def process_jsonl_file(input_path: Path, output_path: Path) -> int:
    """Process a single JSONL file, obfuscating the generated_code field.

    Returns the number of records processed.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            original = record.get("generated_code", "")
            record["generated_code"] = obfuscate_code(original)
            record["original_code"] = original  # keep original for reference
            fout.write(json.dumps(record) + "\n")
            count += 1

    return count


@app.command()
def run(
    dataset_folder: str = typer.Option(..., "--dataset-folder", help="Dataset folder name"),
    split: str = typer.Option(..., "--split", help="Dataset split"),
    input_dir: Optional[str] = typer.Option(None, "--input-dir", help="Override input directory"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Override output directory"),
) -> None:
    """Apply R&P obfuscation to all code generation JSONL files."""

    data_dir = Path("data")
    src_dir = Path(input_dir) if input_dir else data_dir / "code_generation" / dataset_folder / split
    dst_dir = Path(output_dir) if output_dir else data_dir / "code_generation_obfuscated" / dataset_folder / split
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        console.print(f"[red]Source directory not found: {src_dir}[/]")
        raise typer.Exit(1)

    jsonl_files = sorted(src_dir.glob("*.jsonl"))
    if not jsonl_files:
        console.print(f"[yellow]No JSONL files found in {src_dir}[/]")
        return

    console.print(f"[blue]Obfuscating {len(jsonl_files)} files from {src_dir} → {dst_dir}[/]")

    total = 0
    for f in tqdm(jsonl_files, desc="Obfuscating", unit="file"):
        out = dst_dir / f.name
        n = process_jsonl_file(f, out)
        total += n
        console.print(f"  {f.name}: {n} records obfuscated → {out.name}")

    console.print(f"\n[green]Done.[/] {total} total records obfuscated across {len(jsonl_files)} files.")
    console.print(f"Output: {dst_dir}")


@app.command()
def demo(
    code: str = typer.Argument(None, help="Code to obfuscate (or reads from stdin)"),
) -> None:
    """Demo: obfuscate a single code snippet."""
    import sys

    if code is None:
        code = sys.stdin.read()

    console.print("[dim]─── Original ───[/]")
    console.print(code)
    console.print("\n[dim]─── Obfuscated ───[/]")
    console.print(obfuscate_code(code))


if __name__ == "__main__":
    app()
