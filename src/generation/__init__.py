"""Code generation module with dataset-specific adapters."""

from src.generation.base import BaseGenerator, extract_code_from_response
from src.generation.mbpp import MBPPGenerator
from src.generation.humaneval import HumanEvalGenerator
from src.generation.ds1000 import DS1000Generator

__all__ = [
    "BaseGenerator",
    "extract_code_from_response",
    "MBPPGenerator",
    "HumanEvalGenerator",
    "DS1000Generator",
]


def get_generator(dataset: str) -> type:
    """Get the generator class for a dataset.
    
    Args:
        dataset: Dataset name (mbpp, humaneval, ds1000)
        
    Returns:
        Generator class for the dataset
    """
    registry = {
        "mbpp": MBPPGenerator,
        "humaneval": HumanEvalGenerator,
        "ds1000": DS1000Generator,
    }
    key = dataset.lower()
    if key not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {available}")
    return registry[key]

