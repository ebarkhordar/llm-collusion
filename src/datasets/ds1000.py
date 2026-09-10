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
        
        DS-1000 uses a special testing format where `code_context` contains
        test_execution(solution: str) and test_string(solution: str) functions
        that evaluate generated code by taking it as a string argument.
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
        # Get library from metadata
        metadata = ex.get("metadata", {})
        lib = metadata.get("library", "") if isinstance(metadata, dict) else ""
        
        # Optional library filter
        if library and lib.lower() != library.lower():
            global_idx += 1
            continue
            
        if global_idx < start_index:
            global_idx += 1
            continue
        # If end_index is negative (e.g., -1), iterate over the entire dataset
        if end_index >= 0 and global_idx >= end_index:
            break

        # DS-1000 schema:
        # - prompt: the problem description with setup code (ends with BEGIN SOLUTION)
        # - reference_code: the reference solution
        # - code_context: contains test_execution() and test_string() for evaluation
        # - metadata: contains library, problem_id, etc.
        
        prompt = ex.get("prompt", "")
        reference_code = ex.get("reference_code", "")
        code_context = ex.get("code_context", "")
        
        # Build task_id from library and problem_id
        problem_id = metadata.get("problem_id", global_idx) if isinstance(metadata, dict) else global_idx
        task_id = f"{lib}/{problem_id}" if lib else str(problem_id)

        # DS-1000 uses code_context for testing, not assertion-based test_list
        # The test_list is empty; testing is done via test_execution(solution) function
        # We preserve test_imports based on library for potential use in code execution
        test_imports: List[str] = []
        lib_lower = lib.lower() if lib else ""
        if lib_lower == "numpy":
            test_imports.append("import numpy as np")
        elif lib_lower == "pandas":
            test_imports.append("import pandas as pd")
            test_imports.append("import numpy as np")
        elif lib_lower == "matplotlib":
            test_imports.append("import matplotlib.pyplot as plt")
            test_imports.append("import numpy as np")
        elif lib_lower == "tensorflow":
            test_imports.append("import tensorflow as tf")
            test_imports.append("import numpy as np")
        elif lib_lower == "pytorch":
            test_imports.append("import torch")
            test_imports.append("import numpy as np")
        elif lib_lower == "scipy":
            test_imports.append("import scipy")
            test_imports.append("import numpy as np")
        elif lib_lower == "sklearn" or lib_lower == "scikit-learn":
            test_imports.append("import sklearn")
            test_imports.append("import numpy as np")

        examples.append(
            TaskExample(
                benchmark=dataset_label,
                task_id=str(task_id),
                prompt=str(prompt),
                code=str(reference_code),
                test_imports=test_imports,
                test_list=[],  # DS-1000 uses code_context for testing instead
                code_context=str(code_context) if code_context else None,
            )
        )
        global_idx += 1

    return examples
