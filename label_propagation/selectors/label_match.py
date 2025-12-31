"""
Label Match Selector

Selects labels based on direct label-to-asset semantic matching.
Computes similarity between asset embedding and label embeddings.

This approach:
"Which labels semantically match this asset's content?"
"""

from typing import List, Set, Dict, Any, Optional
import numpy as np

from label_propagation.selectors.base import LabelSelector, SelectionResult


class LabelMatchSelector(LabelSelector):
    """
    Select labels based on direct label-asset semantic matching.
    
    Strategy:
    1. Compare asset embedding to label embeddings
    2. Select labels above similarity threshold
    3. Optionally limit to top-k matches
    
    Focus: Direct semantic relevance
    
    Requires: Label embeddings (e.g., from label names, descriptions)
    """
    
    def __init__(
        self,
        label_embeddings: Dict[str, np.ndarray],
        min_similarity: float = 0.5,
        top_k: Optional[int] = None,
        normalize: bool = True,
    ):
        """
        Initialize label match selector.
        
        Args:
            label_embeddings: Map from label_id to label embedding
            min_similarity: Minimum similarity threshold
            top_k: Return only top K matches (None = all above threshold)
            normalize: Whether to normalize embeddings for cosine similarity
        """
        self.label_embeddings = label_embeddings
        self.min_similarity = min_similarity
        self.top_k = top_k
        self.normalize = normalize
        
        # Preprocess label embeddings
        self.label_ids = list(label_embeddings.keys())
        self.embedding_matrix = np.array([
            label_embeddings[label_id] for label_id in self.label_ids
        ]).astype(np.float32)
        
        if self.normalize:
            norms = np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self.embedding_matrix = self.embedding_matrix / norms
        
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
        Select labels based on semantic match to asset.
        
        Args:
            asset_id: Asset identifier
            embedding: Asset embedding vector
            context: Optional context (can override min_similarity, top_k)
        
        Returns:
            SelectionResult with semantically matched labels
        """
        # Handle context overrides
        min_sim = context.get("min_similarity", self.min_similarity) if context else self.min_similarity
        top_k = context.get("top_k", self.top_k) if context else self.top_k
        
        # Normalize query embedding
        if self.normalize:
            query = embedding.reshape(1, -1)
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm
        else:
            query = embedding.reshape(1, -1)
        
        # Compute similarities
        similarities = np.dot(self.embedding_matrix, query.T).squeeze()
        
        # Filter by threshold
        above_threshold = similarities >= min_sim
        candidate_indices = np.where(above_threshold)[0]
        
        if len(candidate_indices) == 0:
            candidate_labels = set()
            label_similarities = {}
        else:
            # Get similarities for candidates
            candidate_sims = similarities[candidate_indices]
            
            # Sort by similarity (descending)
            sorted_indices = np.argsort(candidate_sims)[::-1]
            
            # Apply top_k if specified
            if top_k is not None:
                sorted_indices = sorted_indices[:top_k]
            
            # Get labels
            candidate_labels = set()
            label_similarities = {}
            
            for idx in sorted_indices:
                global_idx = candidate_indices[idx]
                label_id = self.label_ids[global_idx]
                similarity = float(candidate_sims[idx])
                
                candidate_labels.add(label_id)
                label_similarities[label_id] = similarity
        
        # Update statistics
        self._total_selections += 1
        self._avg_candidates = (
            (self._avg_candidates * (self._total_selections - 1) + len(candidate_labels))
            / self._total_selections
        )
        
        # Build metadata
        metadata = {
            "selector_type": "label_match",
            "label_similarities": label_similarities,
            "n_labels_evaluated": len(self.label_ids),
            "n_above_threshold": len(candidate_indices),
        }
        
        return SelectionResult(
            asset_id=asset_id,
            candidate_labels=candidate_labels,
            metadata=metadata,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get selector statistics."""
        return {
            "selector_type": "label_match",
            "n_labels": len(self.label_ids),
            "min_similarity": self.min_similarity,
            "top_k": self.top_k,
            "total_selections": self._total_selections,
            "avg_candidates_per_asset": self._avg_candidates,
        }
    
    def __repr__(self) -> str:
        return (
            f"LabelMatchSelector(n_labels={len(self.label_ids)}, "
            f"min_sim={self.min_similarity})"
        )
