"""Tests for label selectors module."""

import pytest
import numpy as np
from label_propagation.knn import KNNIndex
from label_propagation.selectors import (
    AssetNeighborhoodSelector,
    LabelMatchSelector,
    LabelDensitySelector,
    UnionSelector,
    IntersectionSelector,
    WeightedSelector,
)


class TestAssetNeighborhoodSelector:
    """Test suite for AssetNeighborhoodSelector."""
    
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
    
    def test_basic_selection(self):
        """Test basic label selection from neighbors."""
        selector = AssetNeighborhoodSelector(
            knn_index=self.index,
            labels_db=self.labels_db,
            k=10,
        )
        
        result = selector.select("new_asset", self.embeddings[50])
        
        assert result.asset_id == "new_asset"
        assert isinstance(result.candidate_labels, set)
        assert len(result.candidate_labels) > 0
    
    def test_min_support_filter(self):
        """Test minimum support filtering."""
        selector = AssetNeighborhoodSelector(
            knn_index=self.index,
            labels_db=self.labels_db,
            k=20,
            min_neighbor_support=3,  # Label must appear in 3+ neighbors
        )
        
        result = selector.select("new_asset", self.embeddings[50])
        
        # All labels should have at least 3 supporting neighbors
        for label_id in result.candidate_labels:
            support = result.metadata["label_frequencies"][label_id]
            assert support >= 3
    
    def test_max_labels(self):
        """Test max labels constraint."""
        selector = AssetNeighborhoodSelector(
            knn_index=self.index,
            labels_db=self.labels_db,
            k=50,
            max_labels=5,
        )
        
        result = selector.select("new_asset", self.embeddings[50])
        
        assert len(result.candidate_labels) <= 5
    
    def test_statistics(self):
        """Test statistics collection."""
        selector = AssetNeighborhoodSelector(
            knn_index=self.index,
            labels_db=self.labels_db,
            k=10,
        )
        
        # Make a few selections
        for i in range(5):
            selector.select(f"new_{i}", self.embeddings[50 + i])
        
        stats = selector.get_statistics()
        assert stats["total_selections"] == 5
        assert "avg_candidates_per_asset" in stats


class TestLabelMatchSelector:
    """Test suite for LabelMatchSelector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Create label embeddings
        self.label_embeddings = {
            f"label_{i}": np.random.randn(128).astype(np.float32)
            for i in range(20)
        }
    
    def test_basic_matching(self):
        """Test basic label-asset matching."""
        selector = LabelMatchSelector(
            label_embeddings=self.label_embeddings,
            min_similarity=0.0,
        )
        
        query = np.random.randn(128).astype(np.float32)
        result = selector.select("asset_1", query)
        
        assert result.asset_id == "asset_1"
        assert len(result.candidate_labels) > 0
        assert "label_similarities" in result.metadata
    
    def test_similarity_threshold(self):
        """Test similarity threshold filtering."""
        selector = LabelMatchSelector(
            label_embeddings=self.label_embeddings,
            min_similarity=0.8,  # High threshold
        )
        
        query = np.random.randn(128).astype(np.float32)
        result = selector.select("asset_1", query)
        
        # All selected labels should have high similarity
        for label_id in result.candidate_labels:
            sim = result.metadata["label_similarities"][label_id]
            assert sim >= 0.8
    
    def test_top_k(self):
        """Test top-k selection."""
        selector = LabelMatchSelector(
            label_embeddings=self.label_embeddings,
            min_similarity=0.0,
            top_k=5,
        )
        
        query = np.random.randn(128).astype(np.float32)
        result = selector.select("asset_1", query)
        
        assert len(result.candidate_labels) <= 5


class TestLabelDensitySelector:
    """Test suite for LabelDensitySelector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create labels with varying frequencies
        self.labels_db = {}
        
        # Popular labels (appear 20+ times)
        for i in range(20):
            self.labels_db[f"asset_{i}"] = ["popular_1", "popular_2"]
        
        # Medium labels (appear 10 times)
        for i in range(20, 30):
            self.labels_db[f"asset_{i}"] = ["medium_1", "medium_2"]
        
        # Rare labels (appear 2 times)
        for i in range(30, 32):
            self.labels_db[f"asset_{i}"] = ["rare_1", "rare_2"]
    
    def test_min_frequency(self):
        """Test minimum frequency filtering."""
        selector = LabelDensitySelector(
            labels_db=self.labels_db,
            min_frequency=10,  # Only labels with 10+ occurrences
        )
        
        stats = selector.get_statistics()
        allowed = selector.allowed_labels
        
        # Should include popular and medium, exclude rare
        assert "popular_1" in allowed
        assert "medium_1" in allowed
        assert "rare_1" not in allowed
    
    def test_max_frequency(self):
        """Test maximum frequency filtering."""
        selector = LabelDensitySelector(
            labels_db=self.labels_db,
            max_frequency=15,  # Only labels with <=15 occurrences
        )
        
        allowed = selector.allowed_labels
        
        # Should include medium and rare, exclude popular
        assert "popular_1" not in allowed
        assert "medium_1" in allowed
        assert "rare_1" in allowed
    
    def test_percentile_filter(self):
        """Test percentile-based filtering."""
        selector = LabelDensitySelector(
            labels_db=self.labels_db,
            min_percentile=25,
            max_percentile=75,
        )
        
        allowed = selector.allowed_labels
        assert len(allowed) > 0
    
    def test_filtering_candidates(self):
        """Test filtering candidate labels."""
        selector = LabelDensitySelector(
            labels_db=self.labels_db,
            min_frequency=10,
        )
        
        # Provide candidates including rare labels
        candidates = {"popular_1", "rare_1", "medium_1"}
        
        result = selector.select(
            "test_asset",
            np.zeros(128),
            context={"candidate_labels": candidates}
        )
        
        # Should filter out rare_1
        assert "rare_1" not in result.candidate_labels
        assert "popular_1" in result.candidate_labels


class TestCompositeSelectors:
    """Test suite for composite selectors."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Setup for neighborhood selector
        embeddings = np.random.randn(50, 64).astype(np.float32)
        asset_ids = [f"asset_{i}" for i in range(50)]
        labels_db = {
            f"asset_{i}": [f"label_{i % 10}"] for i in range(30)
        }
        index = KNNIndex(embeddings, asset_ids, use_faiss=False)
        
        self.selector1 = AssetNeighborhoodSelector(
            knn_index=index,
            labels_db=labels_db,
            k=10,
        )
        
        # Setup for label match selector
        label_embeddings = {
            f"label_{i}": np.random.randn(64).astype(np.float32)
            for i in range(15)
        }
        
        self.selector2 = LabelMatchSelector(
            label_embeddings=label_embeddings,
            min_similarity=0.1,
        )
        
        self.test_embedding = np.random.randn(64).astype(np.float32)
    
    def test_union_selector(self):
        """Test union combination."""
        selector = UnionSelector([self.selector1, self.selector2])
        
        result = selector.select("asset_test", self.test_embedding)
        
        # Should have labels from both selectors
        assert len(result.candidate_labels) > 0
        assert result.metadata["strategy"] == "union"
    
    def test_intersection_selector(self):
        """Test intersection combination."""
        selector = IntersectionSelector([self.selector1, self.selector2])
        
        result = selector.select("asset_test", self.test_embedding)
        
        # Should only have labels selected by BOTH
        assert result.metadata["strategy"] == "intersection"
    
    def test_weighted_selector(self):
        """Test weighted combination."""
        selector = WeightedSelector(
            [self.selector1, self.selector2],
            weights=[0.7, 0.3]
        )
        
        result = selector.select("asset_test", self.test_embedding)
        
        assert result.metadata["strategy"] == "weighted"
    
    def test_min_selectors_union(self):
        """Test union with minimum selector requirement."""
        selector = UnionSelector(
            [self.selector1, self.selector2],
            min_selectors=2  # Label must be selected by both
        )
        
        result = selector.select("asset_test", self.test_embedding)
        
        # Should be similar to intersection
        assert result.metadata["strategy"] == "union"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
