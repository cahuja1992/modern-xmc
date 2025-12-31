"""
Label Density Selector

Filters labels based on their frequency/rarity in the dataset.
Can target popular labels, rare labels, or specific density ranges.

This approach:
"Focus on labels with specific popularity characteristics"
"""

from typing import List, Set, Dict, Any, Optional
import numpy as np
from collections import Counter

from label_propagation.selectors.base import LabelSelector, SelectionResult


class LabelDensitySelector(LabelSelector):
    """
    Filter labels based on their density (popularity/rarity).
    
    Strategy:
    1. Take candidate labels from another selector (or all labels)
    2. Filter based on label frequency in the dataset
    3. Can target popular, rare, or specific density ranges
    
    Focus: Coverage of specific label frequency ranges
    
    This is a filtering selector - typically used in combination with others.
    """
    
    def __init__(
        self,
        labels_db: Dict[str, List[str]],
        min_frequency: Optional[int] = None,
        max_frequency: Optional[int] = None,
        min_percentile: Optional[float] = None,
        max_percentile: Optional[float] = None,
        target_labels: Optional[Set[str]] = None,
    ):
        """
        Initialize label density selector.
        
        Args:
            labels_db: Mapping from asset_id to label_ids
            min_frequency: Minimum label frequency (absolute)
            max_frequency: Maximum label frequency (absolute)
            min_percentile: Minimum frequency percentile (0-100)
            max_percentile: Maximum frequency percentile (0-100)
            target_labels: Specific set of labels to consider (None = all)
        """
        self.labels_db = labels_db
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.target_labels = target_labels
        
        # Compute label frequencies
        self.label_frequencies = self._compute_frequencies()
        self.allowed_labels = self._compute_allowed_labels()
        
        # Statistics
        self._total_selections = 0
        self._avg_candidates = 0.0
        self._avg_filtered_out = 0.0
    
    def _compute_frequencies(self) -> Counter:
        """Compute frequency of each label."""
        counter = Counter()
        for asset_id, labels in self.labels_db.items():
            for label_id in labels:
                counter[label_id] += 1
        return counter
    
    def _compute_allowed_labels(self) -> Set[str]:
        """Compute set of allowed labels based on density criteria."""
        all_labels = set(self.label_frequencies.keys())
        
        # Start with all labels or target subset
        if self.target_labels is not None:
            allowed = self.target_labels & all_labels
        else:
            allowed = all_labels.copy()
        
        # Filter by absolute frequency
        if self.min_frequency is not None:
            allowed = {
                label_id for label_id in allowed
                if self.label_frequencies[label_id] >= self.min_frequency
            }
        
        if self.max_frequency is not None:
            allowed = {
                label_id for label_id in allowed
                if self.label_frequencies[label_id] <= self.max_frequency
            }
        
        # Filter by percentile
        if self.min_percentile is not None or self.max_percentile is not None:
            frequencies = [self.label_frequencies[lid] for lid in all_labels]
            
            if self.min_percentile is not None:
                min_freq = np.percentile(frequencies, self.min_percentile)
                allowed = {
                    label_id for label_id in allowed
                    if self.label_frequencies[label_id] >= min_freq
                }
            
            if self.max_percentile is not None:
                max_freq = np.percentile(frequencies, self.max_percentile)
                allowed = {
                    label_id for label_id in allowed
                    if self.label_frequencies[label_id] <= max_freq
                }
        
        return allowed
    
    def select(
        self,
        asset_id: str,
        embedding: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> SelectionResult:
        """
        Filter candidate labels by density.
        
        Args:
            asset_id: Asset identifier
            embedding: Asset embedding (not used by this selector)
            context: Optional context with 'candidate_labels' to filter
        
        Returns:
            SelectionResult with density-filtered labels
        """
        # Get candidate labels from context or use all allowed
        if context and "candidate_labels" in context:
            candidates = context["candidate_labels"]
            if isinstance(candidates, list):
                candidates = set(candidates)
        else:
            candidates = self.allowed_labels
        
        # Filter by allowed labels
        filtered_candidates = candidates & self.allowed_labels
        
        # Get frequencies for selected labels
        label_frequencies_selected = {
            label_id: self.label_frequencies[label_id]
            for label_id in filtered_candidates
        }
        
        # Update statistics
        n_filtered_out = len(candidates) - len(filtered_candidates)
        self._total_selections += 1
        self._avg_candidates = (
            (self._avg_candidates * (self._total_selections - 1) + len(filtered_candidates))
            / self._total_selections
        )
        self._avg_filtered_out = (
            (self._avg_filtered_out * (self._total_selections - 1) + n_filtered_out)
            / self._total_selections
        )
        
        # Build metadata
        metadata = {
            "selector_type": "label_density",
            "n_candidates_input": len(candidates),
            "n_candidates_output": len(filtered_candidates),
            "n_filtered_out": n_filtered_out,
            "label_frequencies": label_frequencies_selected,
        }
        
        return SelectionResult(
            asset_id=asset_id,
            candidate_labels=filtered_candidates,
            metadata=metadata,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get selector statistics."""
        freq_values = list(self.label_frequencies.values())
        
        return {
            "selector_type": "label_density",
            "n_total_labels": len(self.label_frequencies),
            "n_allowed_labels": len(self.allowed_labels),
            "min_frequency": self.min_frequency,
            "max_frequency": self.max_frequency,
            "min_percentile": self.min_percentile,
            "max_percentile": self.max_percentile,
            "frequency_stats": {
                "min": min(freq_values) if freq_values else 0,
                "max": max(freq_values) if freq_values else 0,
                "mean": np.mean(freq_values) if freq_values else 0,
                "median": np.median(freq_values) if freq_values else 0,
            },
            "total_selections": self._total_selections,
            "avg_candidates_per_asset": self._avg_candidates,
            "avg_filtered_out_per_asset": self._avg_filtered_out,
        }
    
    def __repr__(self) -> str:
        return (
            f"LabelDensitySelector(allowed={len(self.allowed_labels)}, "
            f"total={len(self.label_frequencies)})"
        )
