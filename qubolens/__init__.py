"""QUBOLens: visual, explainable feature selection."""

from .core import AnnealResult, QuboModel, build_feature_qubo, solve_qubo
from .data import (
    Dataset,
    inspect_tabular_upload,
    load_csv_dataset,
    load_tabular_dataset,
    make_demo,
    parse_tabular_upload,
)
from .pipeline import optimize_dataset

__all__ = [
    "AnnealResult",
    "Dataset",
    "QuboModel",
    "build_feature_qubo",
    "load_csv_dataset",
    "load_tabular_dataset",
    "make_demo",
    "optimize_dataset",
    "inspect_tabular_upload",
    "parse_tabular_upload",
    "solve_qubo",
]

__version__ = "1.0.0"
