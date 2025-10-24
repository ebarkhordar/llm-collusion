from __future__ import annotations

from typing import List, Optional
from datasets import load_dataset  # type: ignore

from src.common.types import TaskExample


def load_mbpp(start_index: int, end_index: int, sanitized: bool = False) -> List[TaskExample]:
    """
    Load MBPP or MBPP-sanitized dataset into TaskExample objects.

    Args:
        start_index: index to start loading from
        end_index: index to stop loading (exclusive)
        sanitized: if True, use the sanitized MBPP dataset version

    Returns:
        List[TaskExample]: list of examples within range
    """

    dataset_name = "google-research-datasets/mbpp-sanitized" if sanitized else "mbpp"
    split = "train" if sanitized else "train+validation+test"
    ds = load_dataset(dataset_name, split=split)

    examples: List[TaskExample] = []
    for i, ex in enumerate(ds):
        if i < start_index:
            continue
        if i >= end_index:
            break

        if sanitized:
            # Sanitized schema
            prompt = ex.get("prompt", "")
            test_list = ex.get("test_list")
            setup = ex.get("test_imports")
            reference = ex.get("code", "")

            examples.append(
                TaskExample(
                    dataset_name="mbpp-sanitized",
                    dataset_task_id=str(ex["task_id"]),
                    prompt=str(prompt),
                    test_list=list(test_list) if test_list else None,
                    challenge_test_list=None,
                    test_setup_code="\n".join(setup) if isinstance(setup, list) else str(setup) if setup else None,
                    reference_solution=str(reference),
                )
            )
        else:
            # Original MBPP schema
            prompt = ex["text"]
            test_list = ex["test_list"]
            challenge_tests = ex["challenge_test_list"]
            setup = ex["test_setup_code"]
            reference = ex["code"]

            examples.append(
                TaskExample(
                    dataset_name="mbpp",
                    dataset_task_id=str(ex["task_id"]),
                    prompt=str(prompt),
                    test_list=list(test_list) if test_list else None,
                    challenge_test_list=list(challenge_tests) if challenge_tests else None,
                    test_setup_code=str(setup) if setup else None,
                    reference_solution=str(reference) if reference else None,
                )
            )

    return examples