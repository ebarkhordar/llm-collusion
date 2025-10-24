from __future__ import annotations

from typing import Iterable, List
from datasets import load_dataset  # type: ignore

from src.common.types import TaskExample


def load_mbpp(limit: int | None = None) -> List[TaskExample]:
    ds = load_dataset("mbpp", split="test")
    examples: List[TaskExample] = []
    for i, ex in enumerate(ds):
        prompt = (
            ex.get("prompt")
            or ex.get("text")
            or ex.get("task_description")
            or "Write a Python function."
        )
        test_list = ex.get("test_list") or None
        challenge_tests = ex.get("challenge_test_list") or None
        setup = ex.get("test_setup_code") or None
        reference = ex.get("code") or None

        examples.append(
            TaskExample(
                dataset_name="mbpp",
                dataset_task_id=str(ex.get("task_id", f"mbpp_{i}")),
                prompt=str(prompt),
                test_list=list(test_list) if test_list else None,
                challenge_test_list=list(challenge_tests) if challenge_tests else None,
                test_setup_code=str(setup) if setup else None,
                reference_solution=str(reference) if reference else None,
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


