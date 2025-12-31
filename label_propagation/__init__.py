"""
Label Propagation Platform

A geometry-first, classifier-free platform for multi-label assignment 
using semantic neighborhoods.
"""

__version__ = "1.0.0"

from label_propagation.knn import KNNIndex
from label_propagation.aggregation import RankWeightedAggregator
from label_propagation.inference import LabelPropagator

__all__ = [
    "KNNIndex",
    "RankWeightedAggregator",
    "LabelPropagator",
]
