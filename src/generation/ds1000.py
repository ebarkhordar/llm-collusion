"""DS-1000 specific code generator.

DS-1000 is a data science benchmark with problems spanning multiple libraries.
It has a different structure than MBPP/HumanEval - often involving
code completion rather than function implementation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.common.types import TaskExample
from src.generation.base import BaseGenerator
from src.lib import render_prompt


class DS1000Generator(BaseGenerator):
    """Generator for DS-1000 dataset.
    
    DS-1000 problems are code completion tasks for data science libraries.
    The prompts typically contain context code with a placeholder to fill in.
    """
    
    def extract_function_name(self, task: TaskExample) -> Optional[str]:
        """Extract function name from DS-1000 task.
        
        DS-1000 tasks are often code completion rather than function implementation,
        so there may not be a specific target function name.
        
        Args:
            task: DS-1000 task example
            
        Returns:
            None for most DS-1000 tasks (completion-based)
        """
        # DS-1000 is typically completion-based, not function-based
        # The task_id often contains the library info (e.g., "Pandas/42")
        return None
    
    def get_folder_name(self) -> str:
        """DS-1000 uses 'ds1000' folder."""
        return "ds1000"
    
    def get_dataset_key(self) -> str:
        """DS-1000 config key."""
        return "ds1000"
    
    def build_messages(self, task: TaskExample) -> List[Dict[str, str]]:
        """Build messages for DS-1000.
        
        DS-1000 prompts contain the full context and problem description.
        Since it's completion-based, we don't pass a function_name.
        """
        rendered = render_prompt(self.gen_prompt_path, prompt=task.prompt, function_name="")
        return [
            {"role": "user", "content": str(rendered.get("user", "")).strip()},
        ]

