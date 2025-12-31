"""
Asset Neighborhood Selector

Selects labels based on asset-to-asset similarity.
Uses kNN to find similar assets and collects their labels.

This is the classic neighborhood-based approach:
"If similar assets have these labels, consider them for this asset"
"""

from typing import List, Set, Dict, Any, Optional
import numpy as np
from collections import Counter

from label_propagation.selectors.base import LabelSelector, SelectionResult
from label_propagation.knn import KNNIndex


class AssetNeighborhoodSelector(LabelSelector):
    """
    Select labels from similar assets (neighborhood-based).
    
    Strategy:
    1. Find k nearest neighbors in embedding space
    2. Collect all labels from these neighbors
    3. Optionally filter by minimum occurrence frequency
    
    Focus: Recall through semantic similarity
    """
    
    def __init__(
        self,
        knn_index: KNNIndex,
        labels_db: Dict[str, List[str]],
        k: int = 50,
        min_neighbor_support: int = 1,
        min_similarity: float = 0.0,
        max_labels: Optional[int] = None,
    ):
        """
        Initialize asset neighborhood selector.
        
        Args:
            knn_index: KNN index for neighbor retrieval
            labels_db: Mapping from asset_id to label_ids
            k: Number of neighbors to retrieve
            min_neighbor_support: Minimum neighbors that must have label
            min_similarity: Minimum similarity threshold
            max_labels: Maximum candidate labels to return
        """
        self.knn_index = knn_index
        self.labels_db = labels_db
        self.k = k
        self.min_neighbor_support = min_neighbor_support
        self.min_similarity = min_similarity
        self.max_labels = max_labels
        
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
        Select candidate labels from neighborhood.
        
        Args:
            asset_id: Asset identifier
            embedding: Asset embedding vector
            context: Optional context (can override k, min_similarity)
        
        Returns:
            SelectionResult with labels from neighbors
        """
        # Handle context overrides
        k = context.get("k", self.k) if context else self.k
        min_sim = context.get("min_similarity", self.min_similarity) if context else self.min_similarity
        exclude_ids = context.get("exclude_ids", []) if context else []
        
        # Find neighbors
        neighbors = self.knn_index.search(
            query_embedding=embedding,
            k=k,
            exclude_ids=exclude_ids,
        )
        
        # Filter by similarity
        neighbors = [n for n in neighbors if n.similarity >= min_sim]
        
        # Collect labels from neighbors
        label_counter = Counter()
        neighbor_ids_with_label = {}
        
        for neighbor in neighbors:
            neighbor_labels = self.labels_db.get(neighbor.asset_id, [])
            for label_id in neighbor_labels:
                label_counter[label_id] += 1
                if label_id not in neighbor_ids_with_label:
                    neighbor_ids_with_label[label_id] = []
                neighbor_ids_with_label[label_id].append(neighbor.asset_id)
        
        # Filter by minimum support
        candidate_labels = {
            label_id 
            for label_id, count in label_counter.items()
            if count >= self.min_neighbor_support
        }
        
        # Apply max_labels if specified (keep most frequent)
        if self.max_labels and len(candidate_labels) > self.max_labels:
            top_labels = [
                label_id 
                for label_id, _ in label_counter.most_common(self.max_labels)
            ]
            candidate_labels = set(top_labels)
        
        # Update statistics
        self._total_selections += 1
        self._avg_candidates = (
            (self._avg_candidates * (self._total_selections - 1) + len(candidate_labels))
            / self._total_selections
        )
        
        # Build metadata
        metadata = {
            "selector_type": "asset_neighborhood",
            "n_neighbors": len(neighbors),
            "n_neighbors_filtered": len([n for n in neighbors if n.similarity >= min_sim]),
            "label_frequencies": dict(label_counter),
            "neighbor_ids_per_label": {
                label_id: neighbor_ids_with_label[label_id]
                for label_id in candidate_labels
            },
        }
        
        return SelectionResult(
            asset_id=asset_id,
            candidate_labels=candidate_labels,
            metadata=metadata,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get selector statistics."""
        return {
            "selector_type": "asset_neighborhood",
            "k": self.k,
            "min_neighbor_support": self.min_neighbor_support,
            "min_similarity": self.min_similarity,
            "max_labels": self.max_labels,
            "total_selections": self._total_selections,
            "avg_candidates_per_asset": self._avg_candidates,
            "n_assets_indexed": len(self.knn_index),
            "n_labeled_assets": len(self.labels_db),
        }
    
    def __repr__(self) -> str:
        return (
            f"AssetNeighborhoodSelector(k={self.k}, "
            f"min_support={self.min_neighbor_support})"
        )
