from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TaskExample:
    dataset_name: str
    dataset_task_id: str
    prompt: str
    test_list: Optional[List[str]] = None
    challenge_test_list: Optional[List[str]] = None
    test_setup_code: Optional[str] = None
    reference_solution: Optional[str] = None


@dataclass
class GenerationRecord:
    dataset_name: str
    dataset_task_id: str
    prompt: str
    model_name: str
    generated_code: str
    test_list: Optional[List[str]] = None
    challenge_test_list: Optional[List[str]] = None
    test_setup_code: Optional[str] = None
    reference_solution: Optional[str] = None
