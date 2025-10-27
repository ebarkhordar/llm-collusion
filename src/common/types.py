from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class TaskExample:
    # Use MBPP sanitized original field names; include benchmark label for provenance
    benchmark: str
    task_id: str
    prompt: str
    code: str
    test_imports: List[str]
    test_list: List[str]


@dataclass
class GenerationRecord:
    # Record of a single model generation for a given task
    task: TaskExample
    model_name: str
    generated_code: str

    def to_dict(self) -> Dict[str, Any]:
        # Flattened schema with academically clear field names
        return {
            "benchmark": self.task.benchmark,
            "task_id": self.task.task_id,
            "prompt": self.task.prompt,
            "reference_code": self.task.code,
            "test_imports": self.task.test_imports,
            "test_list": self.task.test_list,
            "model_name": self.model_name,
            "generated_code": self.generated_code,
        }


@dataclass
class Pair:
    benchmark: str
    task_id: str
    task_prompt: str
    code1: str
    code2: str
    model1: str
    model2: str


@dataclass
class SelfRecognitionResult:
    benchmark: str
    task_id: str
    evaluator_model: str
    candidate_1_model: str
    candidate_2_model: str
    predicted_candidate: Optional[int]
    gold_candidate: Optional[int]
    is_correct: Optional[bool]
    evaluator_response: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "evaluator_model": self.evaluator_model,
            "candidate_1_model": self.candidate_1_model,
            "candidate_2_model": self.candidate_2_model,
            "predicted_candidate": self.predicted_candidate,
            "gold_candidate": self.gold_candidate,
            "is_correct": self.is_correct,
            "evaluator_response": self.evaluator_response,
        }

@dataclass
class TestOutcome:
    benchmark: str
    task_id: str
    model_name: str
    num_tests: int
    num_passed: int
    passed: bool
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "model_name": self.model_name,
            "num_tests": self.num_tests,
            "num_passed": self.num_passed,
            "passed": self.passed,
            "errors": self.errors,
        }