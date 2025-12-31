"""
Label Selectors Module

Decouples label selection (recall/coverage) from label aggregation (precision).

Provides multiple selection strategies:
- AssetNeighborhoodSelector: Select labels based on asset-to-asset similarity
- LabelMatchSelector: Select labels based on label-asset semantic matching
- LabelDensitySelector: Filter labels based on popularity/rarity
- CompositeSelector: Combine multiple selection strategies
"""

from label_propagation.selectors.base import LabelSelector, SelectionResult
from label_propagation.selectors.asset_neighborhood import AssetNeighborhoodSelector
from label_propagation.selectors.label_match import LabelMatchSelector
from label_propagation.selectors.label_density import LabelDensitySelector
from label_propagation.selectors.composite import (
    CompositeSelector,
    UnionSelector,
    IntersectionSelector,
    WeightedSelector,
)

__all__ = [
    "LabelSelector",
    "SelectionResult",
    "AssetNeighborhoodSelector",
    "LabelMatchSelector",
    "LabelDensitySelector",
    "CompositeSelector",
    "UnionSelector",
    "IntersectionSelector",
    "WeightedSelector",
]
