from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


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
    task: TaskExample
    model_name: str
    generated_code: str

    def to_dict(self) -> Dict[str, Any]:
        # flattened, backward-compatible schema
        return {
            "dataset_name": self.task.dataset_name,
            "dataset_task_id": self.task.dataset_task_id,
            "prompt": self.task.prompt,
            "model_name": self.model_name,
            "generated_code": self.generated_code,
            "test_list": self.task.test_list,
            "challenge_test_list": self.task.challenge_test_list,
            "test_setup_code": self.task.test_setup_code,
            "reference_solution": self.task.reference_solution,
        }


@dataclass
class Pair:
    dataset_name: str
    dataset_task_id: str
    task_prompt: str
    code1: str
    code2: str
    model1: str
    model2: str
