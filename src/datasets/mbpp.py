from __future__ import annotations

from typing import List, Optional
from datasets import load_dataset  # type: ignore

from src.common.types import TaskExample

def load_mbpp(
    start_index: int,
    end_index: int,
    sanitized: bool = False
) -> List[TaskExample]:
    """
    Load MBPP (full) or MBPP-sanitized dataset into TaskExample objects.
    """
    dataset_name = "mbpp"
    config = "sanitized" if sanitized else None
    split = "test"  # as per Hub card, only test split present for both config variants

    if config:
        ds = load_dataset(dataset_name, config_name=config, split=split)
        dataset_label = f"{dataset_name}-sanitized"
    else:
        ds = load_dataset(dataset_name, split=split)
        dataset_label = dataset_name

    examples: List[TaskExample] = []
    for i, ex in enumerate(ds):
        if i < start_index:
            continue
        if i >= end_index:
            break

        if sanitized:
            # sanitized schema
            prompt = ex.get("prompt", "")
            test_list = ex.get("test_list")
            setup = ex.get("test_imports")
            reference = ex.get("code", "")
            examples.append(
                TaskExample(
                    dataset_name=dataset_label,
                    dataset_task_id=str(ex["task_id"]),
                    prompt=str(prompt),
                    test_list=list(test_list) if test_list else None,
                    challenge_test_list=None,
                    test_setup_code="\n".join(setup) if isinstance(setup, list) else str(setup) if setup else None,
                    reference_solution=str(reference),
                )
            )
        else:
            # full (original) schema
            prompt = ex.get("text", "")
            test_list = ex.get("test_list")
            challenge_tests = ex.get("challenge_test_list")
            setup = ex.get("test_setup_code")
            reference = ex.get("code", "")
            examples.append(
                TaskExample(
                    dataset_name=dataset_label,
                    dataset_task_id=str(ex["task_id"]),
                    prompt=str(prompt),
                    test_list=list(test_list) if test_list else None,
                    challenge_test_list=list(challenge_tests) if challenge_tests else None,
                    test_setup_code=str(setup) if setup else None,
                    reference_solution=str(reference) if reference else None,
                )
            )
    return examples
