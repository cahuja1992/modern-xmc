"""
Inference Module

Main label propagation engine that orchestrates:
- kNN retrieval
- Rank-weighted aggregation
- Calibration
"""

from label_propagation.inference.propagate import LabelPropagator, PropagationResult

__all__ = ["LabelPropagator", "PropagationResult"]
