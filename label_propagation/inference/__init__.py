"""
Inference Module

Main label propagation engine that orchestrates:
- kNN retrieval
- Rank-weighted aggregation
- Calibration

V2: Decoupled selection and aggregation architecture
"""

from label_propagation.inference.propagate import LabelPropagator, PropagationResult
from label_propagation.inference.propagate_v2 import LabelPropagatorV2

__all__ = ["LabelPropagator", "PropagationResult", "LabelPropagatorV2"]
