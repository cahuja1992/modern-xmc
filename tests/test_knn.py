"""Tests for kNN Index module."""

import pytest
import numpy as np
from label_propagation.knn import KNNIndex


class TestKNNIndex:
    """Test suite for KNNIndex."""
    
    def test_init_basic(self):
        """Test basic initialization."""
        embeddings = np.random.randn(100, 128).astype(np.float32)
        asset_ids = [f"asset_{i}" for i in range(100)]
        
        index = KNNIndex(embeddings, asset_ids, use_faiss=False)
        
        assert len(index) == 100
        assert index.dimension == 128
    
    def test_search_exact(self):
        """Test exact kNN search."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 128).astype(np.float32)
        asset_ids = [f"asset_{i}" for i in range(100)]
        
        index = KNNIndex(embeddings, asset_ids, use_faiss=False)
        
        # Search with first embedding
        query = embeddings[0]
        neighbors = index.search(query, k=10)
        
        assert len(neighbors) <= 10
        assert all(n.similarity >= -1 and n.similarity <= 1 for n in neighbors)
        
        # Check ranks are sequential
        ranks = [n.rank for n in neighbors]
        assert ranks == list(range(1, len(neighbors) + 1))
    
    def test_search_exclude(self):
        """Test neighbor exclusion."""
        embeddings = np.random.randn(50, 128).astype(np.float32)
        asset_ids = [f"asset_{i}" for i in range(50)]
        
        index = KNNIndex(embeddings, asset_ids, use_faiss=False)
        
        query = embeddings[0]
        neighbors = index.search(query, k=10, exclude_ids=["asset_0", "asset_1"])
        
        neighbor_ids = [n.asset_id for n in neighbors]
        assert "asset_0" not in neighbor_ids
        assert "asset_1" not in neighbor_ids
    
    def test_get_embedding(self):
        """Test embedding retrieval."""
        embeddings = np.random.randn(20, 64).astype(np.float32)
        asset_ids = [f"asset_{i}" for i in range(20)]
        
        # Test with normalization disabled
        index = KNNIndex(embeddings, asset_ids, use_faiss=False, normalize=False)
        
        retrieved = index.get_embedding("asset_5")
        assert retrieved is not None
        assert np.allclose(retrieved, embeddings[5])
        
        # Test non-existent
        assert index.get_embedding("nonexistent") is None
    
    def test_normalization(self):
        """Test vector normalization for cosine similarity."""
        embeddings = np.random.randn(30, 128).astype(np.float32)
        asset_ids = [f"asset_{i}" for i in range(30)]
        
        index = KNNIndex(embeddings, asset_ids, normalize=True, use_faiss=False)
        
        # Check embeddings are normalized
        for i in range(len(index.embeddings)):
            norm = np.linalg.norm(index.embeddings[i])
            assert abs(norm - 1.0) < 1e-5
    
    def test_duplicate_asset_ids(self):
        """Test that duplicate asset IDs raise error."""
        embeddings = np.random.randn(10, 64).astype(np.float32)
        asset_ids = ["asset_0"] * 10
        
        with pytest.raises(ValueError, match="must be unique"):
            KNNIndex(embeddings, asset_ids)
    
    def test_mismatched_lengths(self):
        """Test that mismatched embeddings/IDs raise error."""
        embeddings = np.random.randn(10, 64).astype(np.float32)
        asset_ids = [f"asset_{i}" for i in range(5)]
        
        with pytest.raises(ValueError, match="must match"):
            KNNIndex(embeddings, asset_ids)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
