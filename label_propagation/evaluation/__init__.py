"""
Evaluation Module

Metrics and evaluation utilities for label propagation performance.
"""

from label_propagation.evaluation.metrics import (
    PropagationMetrics,
    compute_precision_recall,
    compute_coverage_lift,
)

__all__ = [
    "PropagationMetrics",
    "compute_precision_recall",
    "compute_coverage_lift",
]
