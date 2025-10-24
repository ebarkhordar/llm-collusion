from __future__ import annotations

from typing import List
from datasets import load_dataset  # type: ignore

from src.common.types import TaskExample


def load_mbpp(start_index: int, end_index: int) -> List[TaskExample]:
    ds = load_dataset("mbpp", split="test")
    examples: List[TaskExample] = []
    for i, ex in enumerate(ds):
        if i < start_index:
            continue
        if i >= end_index:
            break
        # MBPP schema fields: task_id, text, code, test_list, test_setup_code, challenge_test_list
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
                test_list=list(test_list) if test_list is not None else None,
                challenge_test_list=list(challenge_tests) if challenge_tests is not None else None,
                test_setup_code=str(setup) if setup is not None else None,
                reference_solution=str(reference) if reference is not None else None,
            )
        )
    return examples


