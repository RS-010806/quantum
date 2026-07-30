"""QUBOLens: visual, explainable feature selection."""

from .core import AnnealResult, QuboModel, build_feature_qubo, solve_qubo
from .data import Dataset, load_csv_dataset, make_demo
from .pipeline import optimize_dataset

__all__ = [
    "AnnealResult",
    "Dataset",
    "QuboModel",
    "build_feature_qubo",
    "load_csv_dataset",
    "make_demo",
    "optimize_dataset",
    "solve_qubo",
]

__version__ = "1.0.0"
