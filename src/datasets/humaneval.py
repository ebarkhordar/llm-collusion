from __future__ import annotations

from typing import List
from datasets import load_dataset  # type: ignore

from src.common.types import TaskExample


def load_humaneval(start_index: int, end_index: int, split: str = "test") -> List[TaskExample]:
    """
    Load HumanEval dataset into TaskExample objects.
    
    HumanEval is OpenAI's hand-written evaluation set for code generation.
    It contains 164 Python programming problems with function signatures and docstrings.
    
    Args:
        start_index: Start index (inclusive)
        end_index: End index (exclusive, -1 for all)
        split: Dataset split - HumanEval only has "test" split
    
    Note:
        HumanEval only has a "test" split. Other split values will still load
        the test split but log a warning.
    """
    # HumanEval only has test split
    if split != "test":
        import warnings
        warnings.warn(f"HumanEval only has 'test' split. Ignoring requested split '{split}'.")
    
    ds = load_dataset("openai/openai_humaneval", split="test")
    dataset_label = "humaneval"

    examples: List[TaskExample] = []
    for i, ex in enumerate(ds):
        if i < start_index:
            continue
        # If end_index is negative (e.g., -1), iterate over the entire dataset
        if end_index >= 0 and i >= end_index:
            break

        # HumanEval schema:
        # - task_id: e.g., "HumanEval/0"
        # - prompt: function signature + docstring (the context given to model)
        # - canonical_solution: reference solution
        # - test: test code as a string
        # - entry_point: function name to call
        
        prompt = ex.get("prompt", "")
        canonical_solution = ex.get("canonical_solution", "")
        test_code = ex.get("test", "")
        entry_point = ex.get("entry_point", "")
        task_id = ex.get("task_id", f"HumanEval/{i}")

        # Parse test code into list of test assertions
        # HumanEval tests are typically in a check() function
        test_lines = [line.strip() for line in test_code.split("\n") if line.strip()]
        
        # Build test imports (HumanEval tests often need the function imported)
        test_imports: List[str] = []
        if entry_point:
            test_imports.append(f"from solution import {entry_point}")

        examples.append(
            TaskExample(
                benchmark=dataset_label,
                task_id=str(task_id),
                prompt=str(prompt),
                code=str(canonical_solution),
                test_imports=test_imports,
                test_list=test_lines,
            )
        )

    return examples

