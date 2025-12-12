"""Base generator class with shared logic for all datasets."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from src.common.types import TaskExample, GenerationRecord
from src.lib import render_prompt


def extract_code_from_response(response: str) -> str:
    """Extract code from LLM response using [CODE] markers.
    
    Expected format: [CODE]python code here[/CODE]
    Falls back to other extraction methods if markers not found.
    
    Args:
        response: The LLM response
        
    Returns:
        The extracted Python code
    """
    # First try: look for [CODE]...[/CODE] markers
    start_marker = "[CODE]"
    end_marker = "[/CODE]"
    
    start_idx = response.find(start_marker)
    if start_idx != -1:
        end_idx = response.find(end_marker, start_idx + len(start_marker))
        if end_idx != -1:
            code = response[start_idx + len(start_marker):end_idx]
            return code.strip()
    
    # Second try: look for markdown code blocks (```python ... ```)
    if "```" in response:
        import re
        # Match ```python or ``` followed by code and closing ```
        pattern = r"```(?:python|py)?\s*\n?(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # Third try: JSON format {"code": "..."}
    try:
        data = json.loads(response)
        if isinstance(data, dict) and "code" in data:
            return data["code"]
    except json.JSONDecodeError:
        pass
    
    # Fallback: return as-is (stripped)
    return response.strip()


class BaseGenerator(ABC):
    """Abstract base class for dataset-specific code generators.
    
    Subclasses must implement:
        - extract_function_name: Get the target function name from a task
        - get_folder_name: Return the folder name for output files
        - get_dataset_key: Return the dataset key for config lookup
    """
    
    def __init__(self, gen_prompt_path: Path = Path("prompts/generation.md")):
        self.gen_prompt_path = gen_prompt_path
    
    @abstractmethod
    def extract_function_name(self, task: TaskExample) -> Optional[str]:
        """Extract the target function name from a task.
        
        This is dataset-specific since different benchmarks provide
        function names in different ways (e.g., entry_point vs parsing tests).
        
        Args:
            task: The task to extract function name from
            
        Returns:
            The function name, or None if it can't be determined
        """
        pass
    
    @abstractmethod
    def get_folder_name(self) -> str:
        """Get the folder name for this dataset's output files.
        
        Returns:
            Folder name to use in data/code_generation/{folder_name}/
        """
        pass
    
    @abstractmethod
    def get_dataset_key(self) -> str:
        """Get the dataset key for config lookup.
        
        This is used to look up models in config.yaml under datasets.{key}.models
        
        Returns:
            Dataset key for configuration
        """
        pass
    
    def build_messages(self, task: TaskExample) -> List[Dict[str, str]]:
        """Build the chat messages for code generation.
        
        Args:
            task: The task to generate code for
            
        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        fn_name = self.extract_function_name(task) or ""
        rendered = render_prompt(self.gen_prompt_path, prompt=task.prompt, function_name=fn_name)
        return [
            {"role": "user", "content": str(rendered.get("user", "")).strip()},
        ]
    
    def make_record(self, task: TaskExample, model_name: str, code: str) -> GenerationRecord:
        """Create a generation record from a completed generation.
        
        Args:
            task: The original task
            model_name: Name of the model that generated the code
            code: The generated code
            
        Returns:
            GenerationRecord with all fields populated
        """
        return GenerationRecord(task=task, model_name=model_name, generated_code=code)
    
    def compute_output_path(self, base_dir: Path, model_name: str, split: str) -> Path:
        """Compute the output file path for a model's generations.
        
        Args:
            base_dir: Base data directory
            model_name: Model name (may contain /)
            split: Dataset split (train, test, etc.)
            
        Returns:
            Path to the output JSONL file
        """
        safe_model = model_name.replace("/", "-")
        folder_name = self.get_folder_name()
        out_path = base_dir / "code_generation" / folder_name / split / f"{safe_model}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path

