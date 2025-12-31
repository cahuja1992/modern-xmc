"""Tests for propagation module."""

import pytest
import numpy as np
from label_propagation.knn import KNNIndex
from label_propagation.inference import LabelPropagator


class TestLabelPropagator:
    """Test suite for LabelPropagator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Create embeddings
        self.embeddings = np.random.randn(100, 128).astype(np.float32)
        self.asset_ids = [f"asset_{i}" for i in range(100)]
        
        # Create labels
        self.labels_db = {
            f"asset_{i}": [f"label_{i % 10}"] for i in range(50)
        }
        
        # Build index
        self.index = KNNIndex(self.embeddings, self.asset_ids, use_faiss=False)
    
    def test_propagate_basic(self):
        """Test basic label propagation."""
        propagator = LabelPropagator(
            knn_index=self.index,
            labels_db=self.labels_db,
            k=10,
        )
        
        # Propagate to new asset
        new_embedding = np.random.randn(128)
        result = propagator.propagate("new_asset", new_embedding)
        
        assert result.asset_id == "new_asset"
        assert isinstance(result.labels, list)
    
    def test_min_confidence_filter(self):
        """Test minimum confidence filtering."""
        propagator = LabelPropagator(
            knn_index=self.index,
            labels_db=self.labels_db,
            k=10,
            min_confidence=0.5,
        )
        
        new_embedding = self.embeddings[0]
        result = propagator.propagate("new_asset", new_embedding)
        
        # All returned labels should have confidence >= 0.5
        for label in result.labels:
            assert label.confidence >= 0.5
    
    def test_top_k_labels(self):
        """Test top-k label filtering."""
        propagator = LabelPropagator(
            knn_index=self.index,
            labels_db=self.labels_db,
            k=20,
            top_k_labels=3,
        )
        
        new_embedding = self.embeddings[0]
        result = propagator.propagate("new_asset", new_embedding)
        
        # Should return at most 3 labels
        assert len(result.labels) <= 3
    
    def test_exclude_ids(self):
        """Test asset exclusion."""
        propagator = LabelPropagator(
            knn_index=self.index,
            labels_db=self.labels_db,
            k=10,
        )
        
        new_embedding = self.embeddings[0]
        result = propagator.propagate(
            "new_asset",
            new_embedding,
            exclude_ids=["asset_0", "asset_1"]
        )
        
        # Excluded assets should not appear in supporting neighbors
        for label in result.labels:
            if label.supporting_neighbors:
                neighbor_ids = [n["asset_id"] for n in label.supporting_neighbors]
                assert "asset_0" not in neighbor_ids
                assert "asset_1" not in neighbor_ids
    
    def test_supporting_neighbors(self):
        """Test supporting neighbor tracking."""
        propagator = LabelPropagator(
            knn_index=self.index,
            labels_db=self.labels_db,
            k=10,
            include_neighbors=True,
        )
        
        new_embedding = self.embeddings[0]
        result = propagator.propagate("new_asset", new_embedding)
        
        if result.labels:
            # Check supporting neighbors exist
            label = result.labels[0]
            if label.supporting_neighbors:
                assert all("asset_id" in n for n in label.supporting_neighbors)
                assert all("similarity" in n for n in label.supporting_neighbors)
    
    def test_propagate_batch(self):
        """Test batch propagation."""
        propagator = LabelPropagator(
            knn_index=self.index,
            labels_db=self.labels_db,
            k=10,
        )
        
        assets = [
            {
                "asset_id": f"new_asset_{i}",
                "embedding": np.random.randn(128),
            }
            for i in range(5)
        ]
        
        results = propagator.propagate_batch(assets, show_progress=False)
        
        assert len(results) == 5
        assert all(r.asset_id.startswith("new_asset_") for r in results)
    
    def test_get_explanation(self):
        """Test explanation generation."""
        propagator = LabelPropagator(
            knn_index=self.index,
            labels_db=self.labels_db,
            k=10,
        )
        
        # Use embedding similar to labeled asset
        embedding = self.embeddings[0]
        
        explanation = propagator.get_explanation(
            "test_asset",
            embedding,
            "label_0"
        )
        
        assert "label_id" in explanation
        assert "assigned" in explanation
    
    def test_result_to_dict(self):
        """Test result serialization."""
        propagator = LabelPropagator(
            knn_index=self.index,
            labels_db=self.labels_db,
            k=10,
        )
        
        new_embedding = self.embeddings[0]
        result = propagator.propagate("new_asset", new_embedding)
        
        result_dict = result.to_dict()
        
        assert "asset_id" in result_dict
        assert "labels" in result_dict
        assert isinstance(result_dict["labels"], list)
    
    def test_no_neighbors(self):
        """Test handling when no neighbors found."""
        # Empty labels DB
        propagator = LabelPropagator(
            knn_index=self.index,
            labels_db={},
            k=10,
        )
        
        new_embedding = np.random.randn(128)
        result = propagator.propagate("new_asset", new_embedding)
        
        # Should return empty labels
        assert len(result.labels) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
