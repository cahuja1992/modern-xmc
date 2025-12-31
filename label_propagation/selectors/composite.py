"""
Composite Selectors

Combine multiple label selectors using different strategies:
- Union: Take all labels from any selector
- Intersection: Take only labels selected by all selectors
- Weighted: Combine with configurable weights
"""

from typing import List, Set, Dict, Any, Optional
from enum import Enum
import numpy as np

from label_propagation.selectors.base import LabelSelector, SelectionResult


class CombinationStrategy(Enum):
    """Strategy for combining selector results."""
    UNION = "union"
    INTERSECTION = "intersection"
    WEIGHTED = "weighted"


class CompositeSelector(LabelSelector):
    """
    Combine multiple selectors with configurable strategy.
    
    Allows flexible composition of different selection approaches.
    """
    
    def __init__(
        self,
        selectors: List[LabelSelector],
        strategy: CombinationStrategy = CombinationStrategy.UNION,
        weights: Optional[List[float]] = None,
        min_selectors: int = 1,
    ):
        """
        Initialize composite selector.
        
        Args:
            selectors: List of selectors to combine
            strategy: How to combine results
            weights: Weights for weighted strategy (must sum to 1.0)
            min_selectors: For union, minimum selectors that must select a label
        """
        if not selectors:
            raise ValueError("Must provide at least one selector")
        
        self.selectors = selectors
        self.strategy = strategy
        self.weights = weights
        self.min_selectors = min_selectors
        
        if strategy == CombinationStrategy.WEIGHTED:
            if weights is None:
                # Equal weights
                self.weights = [1.0 / len(selectors)] * len(selectors)
            elif len(weights) != len(selectors):
                raise ValueError("Number of weights must match number of selectors")
            elif abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.0")
        
        # Statistics
        self._total_selections = 0
        self._avg_candidates = 0.0
    
    def select(
        self,
        asset_id: str,
        embedding: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> SelectionResult:
        """
        Combine selections from multiple selectors.
        
        Args:
            asset_id: Asset identifier
            embedding: Asset embedding vector
            context: Optional context for selectors
        
        Returns:
            SelectionResult with combined labels
        """
        # Get results from all selectors
        selector_results = []
        for selector in self.selectors:
            result = selector.select(asset_id, embedding, context)
            selector_results.append(result)
        
        # Combine based on strategy
        if self.strategy == CombinationStrategy.UNION:
            combined = self._combine_union(selector_results)
        elif self.strategy == CombinationStrategy.INTERSECTION:
            combined = self._combine_intersection(selector_results)
        elif self.strategy == CombinationStrategy.WEIGHTED:
            combined = self._combine_weighted(selector_results)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Update statistics
        self._total_selections += 1
        self._avg_candidates = (
            (self._avg_candidates * (self._total_selections - 1) + len(combined))
            / self._total_selections
        )
        
        # Build metadata
        metadata = {
            "selector_type": "composite",
            "strategy": self.strategy.value,
            "n_selectors": len(self.selectors),
            "selector_results": [
                {
                    "selector": type(sel).__name__,
                    "n_candidates": len(res.candidate_labels),
                }
                for sel, res in zip(self.selectors, selector_results)
            ],
        }
        
        return SelectionResult(
            asset_id=asset_id,
            candidate_labels=combined,
            metadata=metadata,
        )
    
    def _combine_union(self, results: List[SelectionResult]) -> Set[str]:
        """Union: Take all labels from any selector."""
        # Count how many selectors selected each label
        label_counts = {}
        for result in results:
            for label_id in result.candidate_labels:
                label_counts[label_id] = label_counts.get(label_id, 0) + 1
        
        # Keep labels selected by at least min_selectors
        return {
            label_id for label_id, count in label_counts.items()
            if count >= self.min_selectors
        }
    
    def _combine_intersection(self, results: List[SelectionResult]) -> Set[str]:
        """Intersection: Take only labels selected by ALL selectors."""
        if not results:
            return set()
        
        # Start with first result
        combined = results[0].candidate_labels.copy()
        
        # Intersect with remaining results
        for result in results[1:]:
            combined &= result.candidate_labels
        
        return combined
    
    def _combine_weighted(self, results: List[SelectionResult]) -> Set[str]:
        """
        Weighted: Combine labels with weighted voting.
        
        Labels are included if their weighted support exceeds 0.5.
        """
        label_scores = {}
        
        for result, weight in zip(results, self.weights):
            for label_id in result.candidate_labels:
                label_scores[label_id] = label_scores.get(label_id, 0.0) + weight
        
        # Include labels with weighted support > 0.5
        return {
            label_id for label_id, score in label_scores.items()
            if score > 0.5
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get selector statistics."""
        selector_stats = [
            sel.get_statistics() for sel in self.selectors
        ]
        
        return {
            "selector_type": "composite",
            "strategy": self.strategy.value,
            "n_selectors": len(self.selectors),
            "total_selections": self._total_selections,
            "avg_candidates_per_asset": self._avg_candidates,
            "selector_statistics": selector_stats,
        }
    
    def __repr__(self) -> str:
        return (
            f"CompositeSelector(strategy={self.strategy.value}, "
            f"n_selectors={len(self.selectors)})"
        )


class UnionSelector(CompositeSelector):
    """Convenience class for union combination."""
    
    def __init__(
        self,
        selectors: List[LabelSelector],
        min_selectors: int = 1,
    ):
        """
        Initialize union selector.
        
        Args:
            selectors: List of selectors
            min_selectors: Minimum selectors that must select a label
        """
        super().__init__(
            selectors=selectors,
            strategy=CombinationStrategy.UNION,
            min_selectors=min_selectors,
        )


class IntersectionSelector(CompositeSelector):
    """Convenience class for intersection combination."""
    
    def __init__(self, selectors: List[LabelSelector]):
        """
        Initialize intersection selector.
        
        Args:
            selectors: List of selectors
        """
        super().__init__(
            selectors=selectors,
            strategy=CombinationStrategy.INTERSECTION,
        )


class WeightedSelector(CompositeSelector):
    """Convenience class for weighted combination."""
    
    def __init__(
        self,
        selectors: List[LabelSelector],
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize weighted selector.
        
        Args:
            selectors: List of selectors
            weights: Weights for each selector (must sum to 1.0)
        """
        super().__init__(
            selectors=selectors,
            strategy=CombinationStrategy.WEIGHTED,
            weights=weights,
        )
