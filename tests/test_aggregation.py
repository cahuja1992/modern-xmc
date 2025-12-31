"""Tests for aggregation module."""

import pytest
import numpy as np
from label_propagation.aggregation import RankWeightedAggregator
from label_propagation.knn.index import Neighbor


class TestRankWeightedAggregator:
    """Test suite for RankWeightedAggregator."""
    
    def test_rank_weight(self):
        """Test rank weight computation."""
        labels_db = {}
        aggregator = RankWeightedAggregator(labels_db)
        
        # w(1) = 1 / log2(2) = 1.0
        assert abs(aggregator.rank_weight(1) - 1.0) < 1e-6
        
        # w(2) = 1 / log2(3) ≈ 0.631
        assert abs(aggregator.rank_weight(2) - 0.6309) < 0.001
        
        # Weights should decrease with rank
        w1 = aggregator.rank_weight(1)
        w2 = aggregator.rank_weight(2)
        w3 = aggregator.rank_weight(3)
        
        assert w1 > w2 > w3
    
    def test_aggregate_basic(self):
        """Test basic label aggregation."""
        labels_db = {
            "asset_1": ["label_A", "label_B"],
            "asset_2": ["label_A"],
            "asset_3": ["label_B", "label_C"],
        }
        
        aggregator = RankWeightedAggregator(labels_db)
        
        neighbors = [
            Neighbor("asset_1", similarity=0.9, rank=1),
            Neighbor("asset_2", similarity=0.8, rank=2),
            Neighbor("asset_3", similarity=0.7, rank=3),
        ]
        
        scores = aggregator.aggregate(neighbors)
        
        # Should have at least the labels from neighbors
        label_ids = [s.label_id for s in scores]
        assert "label_A" in label_ids
        assert "label_B" in label_ids
        
        # Scores should be sorted by confidence
        confidences = [s.raw_confidence for s in scores]
        assert confidences == sorted(confidences, reverse=True)
    
    def test_aggregate_empty_neighbors(self):
        """Test aggregation with empty neighbors."""
        labels_db = {"asset_1": ["label_A"]}
        aggregator = RankWeightedAggregator(labels_db)
        
        scores = aggregator.aggregate([])
        assert len(scores) == 0
    
    def test_aggregate_no_labels(self):
        """Test aggregation when neighbors have no labels."""
        labels_db = {}
        aggregator = RankWeightedAggregator(labels_db)
        
        neighbors = [
            Neighbor("asset_1", similarity=0.9, rank=1),
            Neighbor("asset_2", similarity=0.8, rank=2),
        ]
        
        scores = aggregator.aggregate(neighbors)
        assert len(scores) == 0
    
    def test_aggregate_top_k(self):
        """Test top-k label filtering."""
        labels_db = {
            f"asset_{i}": [f"label_{i}"] for i in range(10)
        }
        
        aggregator = RankWeightedAggregator(labels_db)
        
        neighbors = [
            Neighbor(f"asset_{i}", similarity=0.9 - i*0.05, rank=i+1)
            for i in range(10)
        ]
        
        scores = aggregator.aggregate(neighbors, top_k_labels=3)
        
        assert len(scores) == 3
    
    def test_compute_label_density(self):
        """Test label density computation."""
        labels_db = {
            "asset_1": ["label_A", "label_B"],
            "asset_2": ["label_A"],
            "asset_3": ["label_B"],
            "asset_4": ["label_C"],
        }
        
        aggregator = RankWeightedAggregator(labels_db)
        
        neighbors = [
            Neighbor("asset_1", similarity=0.9, rank=1),
            Neighbor("asset_2", similarity=0.8, rank=2),
            Neighbor("asset_3", similarity=0.7, rank=3),
        ]
        
        # label_A appears in 2 neighbors
        density_a = aggregator.compute_label_density(neighbors, "label_A")
        assert density_a == 2
        
        # label_B appears in 2 neighbors
        density_b = aggregator.compute_label_density(neighbors, "label_B")
        assert density_b == 2
        
        # label_C appears in 0 neighbors
        density_c = aggregator.compute_label_density(neighbors, "label_C")
        assert density_c == 0
    
    def test_get_label_statistics(self):
        """Test label statistics computation."""
        labels_db = {
            "asset_1": ["label_A", "label_B"],
            "asset_2": ["label_A"],
            "asset_3": ["label_B", "label_C"],
        }
        
        aggregator = RankWeightedAggregator(labels_db)
        stats = aggregator.get_label_statistics()
        
        assert stats["label_A"] == 2
        assert stats["label_B"] == 2
        assert stats["label_C"] == 1
    
    def test_supporting_neighbors(self):
        """Test that supporting neighbors are tracked correctly."""
        labels_db = {
            "asset_1": ["label_A"],
            "asset_2": ["label_A"],
            "asset_3": ["label_B"],
        }
        
        aggregator = RankWeightedAggregator(labels_db)
        
        neighbors = [
            Neighbor("asset_1", similarity=0.95, rank=1),
            Neighbor("asset_2", similarity=0.85, rank=2),
            Neighbor("asset_3", similarity=0.75, rank=3),
        ]
        
        scores = aggregator.aggregate(neighbors)
        
        # Find label_A score
        label_a_score = next(s for s in scores if s.label_id == "label_A")
        
        # Should have 2 supporting neighbors
        assert len(label_a_score.supporting_neighbors) == 2
        
        # Supporting neighbors should be sorted by similarity
        sims = [sim for _, sim in label_a_score.supporting_neighbors]
        assert sims == sorted(sims, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
