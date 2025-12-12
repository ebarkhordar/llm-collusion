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
    
    HumanEval prompts contain function signatures, so we parse them to get
    the function name (entry_point).
    """
    
    def extract_function_name(self, task: TaskExample) -> Optional[str]:
        """Extract function name from HumanEval task.
        
        Parses the prompt's function signature to get the entry_point.
        
        Args:
            task: HumanEval task example
            
        Returns:
            The entry_point function name
        """
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

