from __future__ import annotations

from typing import List
from datasets import load_dataset  # type: ignore

from src.common.types import TaskExample


def load_mbpp(start_index: int, end_index: int) -> List[TaskExample]:
    """
    Load MBPP-sanitized dataset into TaskExample objects.
    """
    dataset_name = "mbpp"
    config = "sanitized"
    split = "test"

    # Correct call: pass config as second positional arg
    ds = load_dataset(dataset_name, config, split=split)
    dataset_label = f"{dataset_name}-{config}"

    examples: List[TaskExample] = []
    for i, ex in enumerate(ds):
        if i < start_index:
            continue
        if i >= end_index:
            break

        # MBPP sanitized original schema
        prompt = ex.get("prompt", "")
        test_list_raw = ex.get("test_list")
        test_imports_raw = ex.get("test_imports")
        reference_code = ex.get("code", "")

        # Coerce to lists of strings
        def as_list(value) -> List[str]:
            if value is None:
                return []
            if isinstance(value, list):
                return [str(v) for v in value]
            # If provided as a string, split by newline for robustness
            return [s for s in str(value).splitlines() if s]

        examples.append(
            TaskExample(
                benchmark=dataset_label,
                task_id=str(ex.get("task_id")),
                prompt=str(prompt),
                code=str(reference_code),
                test_imports=as_list(test_imports_raw),
                test_list=as_list(test_list_raw),
            )
        )

    return examples
