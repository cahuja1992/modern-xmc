"""
kNN Index Implementation

Deterministic neighborhood retrieval using cosine similarity.
Supports exact and approximate (FAISS) nearest neighbor search.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class Neighbor:
    """A single neighbor with metadata."""
    asset_id: str
    similarity: float
    rank: int


class KNNIndex:
    """
    kNN Index for semantic neighborhood retrieval.
    
    Uses cosine similarity and supports deterministic retrieval
    with both exact and approximate nearest neighbor algorithms.
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        asset_ids: List[str],
        use_faiss: bool = True,
        normalize: bool = True,
        index_type: str = "Flat",
    ):
        """
        Initialize kNN index.
        
        Args:
            embeddings: Asset embeddings, shape (n_assets, embedding_dim)
            asset_ids: List of asset IDs corresponding to embeddings
            use_faiss: Whether to use FAISS for approximate search
            normalize: Whether to normalize embeddings for cosine similarity
            index_type: FAISS index type ("Flat" for exact, "IVF" for approximate)
        """
        if len(embeddings) != len(asset_ids):
            raise ValueError("Number of embeddings must match number of asset IDs")
        
        if len(set(asset_ids)) != len(asset_ids):
            raise ValueError("Asset IDs must be unique")
        
        self.asset_ids = list(asset_ids)
        self.asset_id_to_idx = {aid: idx for idx, aid in enumerate(asset_ids)}
        self.use_faiss = use_faiss
        self.normalize = normalize
        self.dimension = embeddings.shape[1]
        
        # Normalize embeddings for cosine similarity
        if normalize:
            embeddings = self._normalize_vectors(embeddings)
        
        self.embeddings = embeddings.astype(np.float32)
        
        # Build index
        if use_faiss:
            self._build_faiss_index(index_type)
        else:
            # Use exact search with numpy
            self.index = None
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return vectors / norms
    
    def _build_faiss_index(self, index_type: str):
        """Build FAISS index for approximate search."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS not installed. Install with: pip install faiss-cpu"
            )
        
        if index_type == "Flat":
            # Exact search using inner product (= cosine for normalized vectors)
            self.index = faiss.IndexFlatIP(self.dimension)
        elif index_type == "IVF":
            # Approximate search
            nlist = min(100, len(self.embeddings) // 10)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.train(self.embeddings)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.index.add(self.embeddings)
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 50,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Neighbor]:
        """
        Find k nearest neighbors for a query embedding.
        
        Args:
            query_embedding: Query vector, shape (embedding_dim,)
            k: Number of neighbors to retrieve
            exclude_ids: Asset IDs to exclude from results
        
        Returns:
            List of Neighbor objects, sorted by similarity (descending)
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query
        if self.normalize:
            query_embedding = self._normalize_vectors(query_embedding)
        
        query_embedding = query_embedding.astype(np.float32)
        
        # Exclude indices
        exclude_indices = set()
        if exclude_ids:
            exclude_indices = {
                self.asset_id_to_idx[aid] 
                for aid in exclude_ids 
                if aid in self.asset_id_to_idx
            }
        
        # Retrieve more than k to account for exclusions
        k_fetch = min(k + len(exclude_indices) + 10, len(self.embeddings))
        
        if self.use_faiss and self.index is not None:
            # FAISS search
            similarities, indices = self.index.search(query_embedding, k_fetch)
            similarities = similarities[0]
            indices = indices[0]
        else:
            # Exact numpy search
            similarities = np.dot(self.embeddings, query_embedding.T).squeeze()
            indices = np.argsort(similarities)[::-1][:k_fetch]
            similarities = similarities[indices]
        
        # Build neighbor list
        neighbors = []
        rank = 1
        for idx, sim in zip(indices, similarities):
            if idx == -1:  # FAISS sentinel value
                continue
            
            if idx in exclude_indices:
                continue
            
            asset_id = self.asset_ids[idx]
            neighbors.append(Neighbor(
                asset_id=asset_id,
                similarity=float(sim),
                rank=rank
            ))
            rank += 1
            
            if len(neighbors) >= k:
                break
        
        return neighbors
    
    def get_embedding(self, asset_id: str) -> Optional[np.ndarray]:
        """Get embedding for an asset ID."""
        if asset_id not in self.asset_id_to_idx:
            return None
        idx = self.asset_id_to_idx[asset_id]
        return self.embeddings[idx]
    
    def save(self, path: str):
        """Save index to disk."""
        import pickle
        
        state = {
            "embeddings": self.embeddings,
            "asset_ids": self.asset_ids,
            "normalize": self.normalize,
            "use_faiss": self.use_faiss,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        if self.use_faiss and self.index is not None:
            import faiss
            faiss.write_index(self.index, f"{path}.faiss")
    
    @classmethod
    def load(cls, path: str) -> "KNNIndex":
        """Load index from disk."""
        import pickle
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        index = cls(
            embeddings=state["embeddings"],
            asset_ids=state["asset_ids"],
            use_faiss=state["use_faiss"],
            normalize=state["normalize"],
        )
        
        if state["use_faiss"]:
            import faiss
            index.index = faiss.read_index(f"{path}.faiss")
        
        return index
    
    def __len__(self) -> int:
        """Number of assets in index."""
        return len(self.asset_ids)
    
    def __repr__(self) -> str:
        return (
            f"KNNIndex(n_assets={len(self)}, dim={self.dimension}, "
            f"use_faiss={self.use_faiss})"
        )
