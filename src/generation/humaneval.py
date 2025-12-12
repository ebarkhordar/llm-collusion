"""HumanEval-specific code generator.

HumanEval provides an 'entry_point' field directly in the dataset,
making function name extraction trivial.
"""

from __future__ import annotations

import ast
from typing import Dict, List, Optional

from src.common.types import TaskExample
from src.generation.base import BaseGenerator
from src.lib import render_prompt


def _extract_entry_point_from_prompt(prompt: str) -> Optional[str]:
    """Extract function name from the prompt's function signature.
    
    HumanEval prompts typically start with a function definition.
    This is a fallback if we somehow don't have the entry_point stored.
    
    Args:
        prompt: The HumanEval prompt containing the function signature
        
    Returns:
        The function name, or None if not found
    """
    try:
        # HumanEval prompts are partial function definitions
        # Try to parse just enough to get the function name
        tree = ast.parse(prompt + "\n    pass")
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                return node.name
    except Exception:
        pass
    return None


class HumanEvalGenerator(BaseGenerator):
    """Generator for HumanEval dataset.
    
    HumanEval provides function names directly via 'entry_point' in the dataset.
    This is stored in test_imports as "from solution import {entry_point}".
    """
    
    def extract_function_name(self, task: TaskExample) -> Optional[str]:
        """Extract function name from HumanEval task.
        
        HumanEval stores the entry_point in test_imports (added during loading).
        Falls back to parsing the prompt's function signature.
        
        Args:
            task: HumanEval task example
            
        Returns:
            The entry_point function name
        """
        # Entry point is stored in test_imports as "from solution import {name}"
        for imp in task.test_imports:
            if imp.startswith("from solution import "):
                return imp.replace("from solution import ", "").strip()
        
        # Fallback: parse the prompt to find the function definition
        return _extract_entry_point_from_prompt(task.prompt)
    
    def get_folder_name(self) -> str:
        """HumanEval uses 'humaneval' folder."""
        return "humaneval"
    
    def get_dataset_key(self) -> str:
        """HumanEval config key."""
        return "humaneval"
    
    def build_messages(self, task: TaskExample) -> List[Dict[str, str]]:
        """Build messages for HumanEval.
        
        HumanEval prompts already contain the function signature and docstring,
        so we include them as-is in the generation prompt.
        """
        fn_name = self.extract_function_name(task) or ""
        rendered = render_prompt(self.gen_prompt_path, prompt=task.prompt, function_name=fn_name)
        return [
            {"role": "user", "content": str(rendered.get("user", "")).strip()},
        ]

