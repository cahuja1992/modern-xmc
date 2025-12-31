"""
Rank-Weighted Aggregation

Implements the authoritative aggregation mathematics from the LLD:
- Rank-weighted support calculation
- Raw confidence computation
- Multi-label aggregation

NOW DECOUPLED: Works with pre-selected labels from selectors.
Focus: PRECISION (refine the selected labels with confidence scores)
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
from label_propagation.knn.index import Neighbor


@dataclass
class LabelScore:
    """A label with its aggregated score and supporting evidence."""
    label_id: str
    raw_confidence: float
    support: float
    supporting_neighbors: List[Tuple[str, float]]  # (asset_id, similarity)


class RankWeightedAggregator:
    """
    Rank-weighted label aggregation.
    
    Implements the formula:
        support(x,ℓ) = Σ 1[ℓ ∈ L(nᵢ)] · s(x,nᵢ) · w(i)
        where w(i) = 1 / log₂(i + 1)
        
        c_raw(x,ℓ) = support(x,ℓ) / mass(x)
    
    PRECISION FOCUS:
    - Takes pre-selected candidate labels (from selector)
    - Computes confidence scores only for those labels
    - Ranks labels by confidence
    - Filters by minimum confidence
    """
    
    def __init__(self, labels_db: Dict[str, List[str]]):
        """
        Initialize aggregator.
        
        Args:
            labels_db: Mapping from asset_id to list of label_ids
        """
        self.labels_db = labels_db
    
    def rank_weight(self, rank: int) -> float:
        """
        Compute rank weight using logarithmic discounting.
        
        w(i) = 1 / log₂(i + 1)
        
        Args:
            rank: Rank position (1-indexed)
        
        Returns:
            Weight value
        """
        return 1.0 / np.log2(rank + 1)
    
    def aggregate(
        self,
        neighbors: List[Neighbor],
        candidate_labels: Optional[Set[str]] = None,
        min_support: float = 0.0,
        top_k_labels: int = None,
    ) -> List[LabelScore]:
        """
        Aggregate labels from neighbors using rank-weighted voting.
        
        NEW: Can optionally filter to only candidate_labels (from selector).
        
        Args:
            neighbors: List of Neighbor objects from kNN search
            candidate_labels: Optional set of labels to consider (from selector)
            min_support: Minimum support threshold (default: 0.0)
            top_k_labels: Return only top K labels by confidence (default: all)
        
        Returns:
            List of LabelScore objects, sorted by raw confidence (descending)
        """
        # Collect label support
        label_support = defaultdict(float)
        label_neighbors = defaultdict(list)
        
        # Calculate total neighborhood mass
        total_mass = 0.0
        
        for neighbor in neighbors:
            # Get labels for this neighbor
            neighbor_labels = self.labels_db.get(neighbor.asset_id, [])
            
            # Filter to candidate labels if provided
            if candidate_labels is not None:
                neighbor_labels = [
                    label_id for label_id in neighbor_labels
                    if label_id in candidate_labels
                ]
            
            # Compute contribution of this neighbor
            rank_w = self.rank_weight(neighbor.rank)
            contribution = neighbor.similarity * rank_w
            
            total_mass += contribution
            
            # Add support to each label
            for label_id in neighbor_labels:
                label_support[label_id] += contribution
                label_neighbors[label_id].append(
                    (neighbor.asset_id, neighbor.similarity)
                )
        
        # Avoid division by zero
        if total_mass == 0:
            return []
        
        # Compute raw confidence for each label
        label_scores = []
        
        for label_id, support in label_support.items():
            if support < min_support:
                continue
            
            raw_confidence = support / total_mass
            
            # Sort supporting neighbors by similarity
            supporting = sorted(
                label_neighbors[label_id],
                key=lambda x: x[1],
                reverse=True
            )
            
            label_scores.append(LabelScore(
                label_id=label_id,
                raw_confidence=raw_confidence,
                support=support,
                supporting_neighbors=supporting,
            ))
        
        # Sort by raw confidence (descending)
        label_scores.sort(key=lambda x: x.raw_confidence, reverse=True)
        
        # Apply top-k filter if specified
        if top_k_labels is not None:
            label_scores = label_scores[:top_k_labels]
        
        return label_scores
    
    def compute_label_density(
        self,
        neighbors: List[Neighbor],
        label_id: str,
    ) -> int:
        """
        Compute label density: number of neighbors with the given label.
        
        Used in calibration density adjustment.
        
        Args:
            neighbors: List of Neighbor objects
            label_id: Label to check
        
        Returns:
            Count of neighbors with this label
        """
        count = 0
        for neighbor in neighbors:
            neighbor_labels = self.labels_db.get(neighbor.asset_id, [])
            if label_id in neighbor_labels:
                count += 1
        return count
    
    def get_label_statistics(self) -> Dict[str, int]:
        """
        Get statistics about label distribution.
        
        Returns:
            Dictionary mapping label_id to count of assets with that label
        """
        label_counts = defaultdict(int)
        
        for asset_id, labels in self.labels_db.items():
            for label_id in labels:
                label_counts[label_id] += 1
        
        return dict(label_counts)
    
    def get_asset_labels(self, asset_id: str) -> List[str]:
        """Get labels for a specific asset."""
        return self.labels_db.get(asset_id, [])
    
    def __repr__(self) -> str:
        n_assets = len(self.labels_db)
        n_labels = len(self.get_label_statistics())
        return f"RankWeightedAggregator(n_assets={n_assets}, n_labels={n_labels})"
