"""
Label Propagation Platform

A geometry-first, classifier-free platform for multi-label assignment 
using semantic neighborhoods.

V2 ARCHITECTURE:
- Decoupled selection (recall/coverage) and aggregation (precision)
- Composable selectors for flexible label candidate generation
- Modular design for maximum flexibility
"""

__version__ = "2.0.0"

from label_propagation.knn import KNNIndex
from label_propagation.aggregation import RankWeightedAggregator
from label_propagation.inference import LabelPropagator
from label_propagation.inference.propagate_v2 import LabelPropagatorV2

# Import selectors
from label_propagation.selectors import (
    LabelSelector,
    AssetNeighborhoodSelector,
    LabelMatchSelector,
    LabelDensitySelector,
    CompositeSelector,
    UnionSelector,
    IntersectionSelector,
    WeightedSelector,
)

__all__ = [
    # Core V1 (backward compatible)
    "KNNIndex",
    "RankWeightedAggregator",
    "LabelPropagator",
    
    # V2 Architecture
    "LabelPropagatorV2",
    
    # Selectors
    "LabelSelector",
    "AssetNeighborhoodSelector",
    "LabelMatchSelector",
    "LabelDensitySelector",
    "CompositeSelector",
    "UnionSelector",
    "IntersectionSelector",
    "WeightedSelector",
]
