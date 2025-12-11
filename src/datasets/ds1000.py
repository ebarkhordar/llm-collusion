from __future__ import annotations

from typing import List, Optional
from datasets import load_dataset  # type: ignore

from src.common.types import TaskExample


def load_ds1000(
    start_index: int, 
    end_index: int, 
    split: str = "test",
    library: Optional[str] = None
) -> List[TaskExample]:
    """
    Load DS-1000 dataset into TaskExample objects.
    
    DS-1000 is a benchmark of 1000 data science problems spanning 7 Python libraries:
    - NumPy (220 problems)
    - Pandas (291 problems)
    - Matplotlib (155 problems)
    - TensorFlow (45 problems)
    - PyTorch (68 problems)
    - SciPy (106 problems)
    - Scikit-learn (115 problems)
    
    Args:
        start_index: Start index (inclusive)
        end_index: End index (exclusive, -1 for all)
        split: Dataset split - DS-1000 only has "test" split
        library: Optional filter for specific library (e.g., "Pandas", "NumPy")
    
    Note:
        DS-1000 only has a "test" split. Other split values will still load
        the test split but log a warning.
    """
    # DS-1000 only has test split
    if split != "test":
        import warnings
        warnings.warn(f"DS-1000 only has 'test' split. Ignoring requested split '{split}'.")
    
    ds = load_dataset("xlangai/DS-1000", split="test")
    dataset_label = "ds1000"

    examples: List[TaskExample] = []
    global_idx = 0
    
    for ex in ds:
        # Optional library filter
        lib = ex.get("metadata", {}).get("library", "") if isinstance(ex.get("metadata"), dict) else ""
        if not lib:
            lib = ex.get("lib", "")
        
        if library and lib.lower() != library.lower():
            continue
            
        if global_idx < start_index:
            global_idx += 1
            continue
        # If end_index is negative (e.g., -1), iterate over the entire dataset
        if end_index >= 0 and global_idx >= end_index:
            break

        # DS-1000 schema varies slightly, but generally includes:
        # - prompt: the problem description/context
        # - reference_code or code_context: solution or context
        # - test: test code
        # - lib: library name (NumPy, Pandas, etc.)
        
        prompt = ex.get("prompt", "")
        reference_code = ex.get("reference_code", ex.get("code_context", ""))
        test_code = ex.get("test", "")
        
        # Build task_id from library and index
        task_id = ex.get("id", f"{lib}/{global_idx}" if lib else str(global_idx))

        # Parse test code into list
        test_lines = [line for line in str(test_code).split("\n") if line.strip()]
        
        # Build test imports based on library
        test_imports: List[str] = []
        lib_lower = lib.lower() if lib else ""
        if lib_lower == "numpy":
            test_imports.append("import numpy as np")
        elif lib_lower == "pandas":
            test_imports.append("import pandas as pd")
        elif lib_lower == "matplotlib":
            test_imports.append("import matplotlib.pyplot as plt")
        elif lib_lower == "tensorflow":
            test_imports.append("import tensorflow as tf")
        elif lib_lower == "pytorch":
            test_imports.append("import torch")
        elif lib_lower == "scipy":
            test_imports.append("import scipy")
        elif lib_lower == "sklearn" or lib_lower == "scikit-learn":
            test_imports.append("import sklearn")

        examples.append(
            TaskExample(
                benchmark=dataset_label,
                task_id=str(task_id),
                prompt=str(prompt),
                code=str(reference_code),
                test_imports=test_imports,
                test_list=test_lines,
            )
        )
        global_idx += 1

    return examples

