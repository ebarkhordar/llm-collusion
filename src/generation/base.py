"""Base generator class with shared logic for all datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.common.types import TaskExample, GenerationRecord
from src.lib import render_prompt


def strip_markdown_code_blocks(text: str) -> str:
    """Extract code from fenced markdown blocks.

    - If one or more fenced code blocks exist, prefer the first with a
      python/py language tag; otherwise return the longest fenced block.
    - If no fences are present, return the original text stripped.
    """
    if "```" not in text:
        return text.strip()

    lines = text.split("\n")
    in_block = False
    current_block_lines: List[str] = []
    current_lang = ""
    blocks: List[Tuple[str, str]] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_block:
                in_block = True
                current_lang = stripped[3:].strip().lower()
                current_block_lines = []
            else:
                in_block = False
                block_text = "\n".join(current_block_lines).strip()
                blocks.append((current_lang, block_text))
                current_lang = ""
                current_block_lines = []
            continue
        if in_block:
            current_block_lines.append(line)

    # Prefer python code block
    for lang, code in blocks:
        if lang in {"python", "py"} and code:
            return code

    # Fall back to longest non-empty block
    non_empty_blocks = [code for _, code in blocks if code]
    if non_empty_blocks:
        return max(non_empty_blocks, key=len)

    return ""


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

