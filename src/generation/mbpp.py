"""MBPP-specific code generator.

MBPP (Mostly Basic Python Problems) doesn't provide function names directly,
so we need to extract them from test assertions or reference code.
"""

from __future__ import annotations

import ast
from collections import Counter
from typing import List, Optional

from src.common.types import TaskExample
from src.generation.base import BaseGenerator


def _extract_function_names_from_test(line: str) -> List[str]:
    """Extract function call names from a test assertion line.
    
    Args:
        line: A single test assertion line (e.g., "assert foo(1) == 2")
        
    Returns:
        List of function names called in the line
    """
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
    """Extract the first top-level function name from Python code.
    
    Args:
        code: Python source code
        
    Returns:
        Name of the first function defined, or None
    """
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None


# Builtin names to filter out when extracting function names from tests
BUILTIN_NAMES = frozenset({
    "print", "len", "range", "int", "float", "str", "list", "set", 
    "dict", "tuple", "abs", "min", "max", "sum", "sorted", "reversed",
    "enumerate", "zip", "map", "filter", "all", "any", "isinstance",
    "type", "bool", "round", "pow", "divmod", "hex", "oct", "bin",
    "ord", "chr", "input", "open", "repr", "hash", "id", "iter", "next",
})


class MBPPGenerator(BaseGenerator):
    """Generator for MBPP (Mostly Basic Python Problems) dataset.
    
    MBPP doesn't have an entry_point field, so function names are extracted
    by analyzing test assertions or falling back to the reference code.
    """
    
    def extract_function_name(self, task: TaskExample) -> Optional[str]:
        """Extract function name from MBPP task.
        
        Strategy:
        1. Parse test assertions to find the most commonly called function
        2. Filter out Python builtins
        3. Fall back to the first function defined in reference code
        
        Args:
            task: MBPP task example
            
        Returns:
            The target function name, or None if not determinable
        """
        # Count function calls in test assertions
        counts: Counter[str] = Counter()
        for line in task.test_list:
            for name in _extract_function_names_from_test(line):
                if name not in BUILTIN_NAMES:
                    counts[name] += 1
        
        if counts:
            return counts.most_common(1)[0][0]
        
        # Fallback to the reference code's first function definition
        return _extract_function_name_from_code(task.code)
    
    def get_folder_name(self) -> str:
        """MBPP uses 'mbpp-sanitized' folder."""
        return "mbpp-sanitized"
    
    def get_dataset_key(self) -> str:
        """MBPP config key."""
        return "mbpp"

