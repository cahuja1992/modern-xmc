"""
Selector Combinations Example

Demonstrates the V2 architecture with decoupled selection and aggregation.
Shows various selector combinations for different use cases.
"""

import numpy as np
from label_propagation import (
    KNNIndex,
    RankWeightedAggregator,
    LabelPropagatorV2,
    AssetNeighborhoodSelector,
    LabelMatchSelector,
    LabelDensitySelector,
    UnionSelector,
    IntersectionSelector,
    WeightedSelector,
)


def create_sample_data():
    """Create sample data for demonstration."""
    np.random.seed(42)
    
    # Create asset embeddings
    n_assets = 500
    embedding_dim = 256
    embeddings = np.random.randn(n_assets, embedding_dim).astype(np.float32)
    asset_ids = [f"asset_{i}" for i in range(n_assets)]
    
    # Create labels (seed data)
    labels_db = {}
    for i in range(300):
        n_labels = np.random.randint(1, 4)
        labels = [f"label_{np.random.randint(0, 50)}" for _ in range(n_labels)]
        labels_db[f"asset_{i}"] = list(set(labels))
    
    # Create label embeddings (e.g., from label names/descriptions)
    label_embeddings = {
        f"label_{i}": np.random.randn(embedding_dim).astype(np.float32)
        for i in range(50)
    }
    
    return embeddings, asset_ids, labels_db, label_embeddings


def demonstrate_selectors():
    """Demonstrate different selector strategies."""
    print("=" * 70)
    print("LABEL PROPAGATION V2 - SELECTOR COMBINATIONS")
    print("=" * 70)
    print()
    
    # Setup data
    print("Setting up data...")
    embeddings, asset_ids, labels_db, label_embeddings = create_sample_data()
    
    index = KNNIndex(embeddings, asset_ids, use_faiss=False, normalize=True)
    aggregator = RankWeightedAggregator(labels_db)
    
    print(f"  - {len(embeddings)} assets")
    print(f"  - {len(labels_db)} labeled")
    print(f"  - {len(label_embeddings)} labels")
    print()
    
    # Test asset
    test_asset_id = "test_asset"
    test_embedding = embeddings[300]
    
    # ========================================================================
    # STRATEGY 1: Asset Neighborhood Only (Classic approach)
    # ========================================================================
    print("─" * 70)
    print("STRATEGY 1: Asset Neighborhood Selector")
    print("Focus: Labels from similar assets")
    print("─" * 70)
    
    selector_neighborhood = AssetNeighborhoodSelector(
        knn_index=index,
        labels_db=labels_db,
        k=50,
        min_neighbor_support=2,
    )
    
    propagator1 = LabelPropagatorV2(
        selector=selector_neighborhood,
        aggregator=aggregator,
        knn_index=index,
        k=50,
        min_confidence=0.2,
        top_k_labels=5,
    )
    
    result1 = propagator1.propagate(test_asset_id, test_embedding)
    
    print(f"Candidates selected: {result1.selection_metadata['n_candidates']}")
    print(f"Final labels: {len(result1.labels)}")
    if result1.labels:
        print("\nTop labels:")
        for label in result1.labels[:3]:
            print(f"  - {label.label_id}: {label.confidence:.4f}")
    print()
    
    # ========================================================================
    # STRATEGY 2: Label Match Only (Direct semantic matching)
    # ========================================================================
    print("─" * 70)
    print("STRATEGY 2: Label Match Selector")
    print("Focus: Direct label-asset semantic matching")
    print("─" * 70)
    
    selector_match = LabelMatchSelector(
        label_embeddings=label_embeddings,
        min_similarity=0.3,
        top_k=20,
    )
    
    propagator2 = LabelPropagatorV2(
        selector=selector_match,
        aggregator=aggregator,
        knn_index=index,
        k=50,
        min_confidence=0.2,
        top_k_labels=5,
    )
    
    result2 = propagator2.propagate(test_asset_id, test_embedding)
    
    print(f"Candidates selected: {result2.selection_metadata['n_candidates']}")
    print(f"Final labels: {len(result2.labels)}")
    if result2.labels:
        print("\nTop labels:")
        for label in result2.labels[:3]:
            print(f"  - {label.label_id}: {label.confidence:.4f}")
    print()
    
    # ========================================================================
    # STRATEGY 3: Popular Labels Only (Focus on frequent labels)
    # ========================================================================
    print("─" * 70)
    print("STRATEGY 3: Density Filter (Popular Labels)")
    print("Focus: Only consider frequently occurring labels")
    print("─" * 70)
    
    selector_popular = LabelDensitySelector(
        labels_db=labels_db,
        min_percentile=50,  # Top 50% most popular labels
    )
    
    # Combine with neighborhood selector
    selector_combined = UnionSelector(
        selectors=[selector_neighborhood, selector_popular],
        min_selectors=2,  # Label must be selected by BOTH
    )
    
    propagator3 = LabelPropagatorV2(
        selector=selector_combined,
        aggregator=aggregator,
        knn_index=index,
        k=50,
        min_confidence=0.2,
        top_k_labels=5,
    )
    
    result3 = propagator3.propagate(test_asset_id, test_embedding)
    
    print(f"Candidates selected: {result3.selection_metadata['n_candidates']}")
    print(f"Final labels: {len(result3.labels)}")
    if result3.labels:
        print("\nTop labels:")
        for label in result3.labels[:3]:
            print(f"  - {label.label_id}: {label.confidence:.4f}")
    print()
    
    # ========================================================================
    # STRATEGY 4: Intersection (High Precision)
    # ========================================================================
    print("─" * 70)
    print("STRATEGY 4: Intersection Selector (High Precision)")
    print("Focus: Only labels agreed upon by multiple strategies")
    print("─" * 70)
    
    selector_intersection = IntersectionSelector([
        selector_neighborhood,
        selector_match,
    ])
    
    propagator4 = LabelPropagatorV2(
        selector=selector_intersection,
        aggregator=aggregator,
        knn_index=index,
        k=50,
        min_confidence=0.2,
        top_k_labels=5,
    )
    
    result4 = propagator4.propagate(test_asset_id, test_embedding)
    
    print(f"Candidates selected: {result4.selection_metadata['n_candidates']}")
    print(f"Final labels: {len(result4.labels)}")
    if result4.labels:
        print("\nTop labels:")
        for label in result4.labels[:3]:
            print(f"  - {label.label_id}: {label.confidence:.4f}")
    print()
    
    # ========================================================================
    # STRATEGY 5: Weighted Combination (Balanced)
    # ========================================================================
    print("─" * 70)
    print("STRATEGY 5: Weighted Selector (Balanced)")
    print("Focus: Combine strategies with custom weights")
    print("─" * 70)
    
    selector_weighted = WeightedSelector(
        selectors=[selector_neighborhood, selector_match],
        weights=[0.7, 0.3],  # 70% neighborhood, 30% match
    )
    
    propagator5 = LabelPropagatorV2(
        selector=selector_weighted,
        aggregator=aggregator,
        knn_index=index,
        k=50,
        min_confidence=0.2,
        top_k_labels=5,
    )
    
    result5 = propagator5.propagate(test_asset_id, test_embedding)
    
    print(f"Candidates selected: {result5.selection_metadata['n_candidates']}")
    print(f"Final labels: {len(result5.labels)}")
    if result5.labels:
        print("\nTop labels:")
        for label in result5.labels[:3]:
            print(f"  - {label.label_id}: {label.confidence:.4f}")
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 70)
    print("SUMMARY: Selector Strategy Comparison")
    print("=" * 70)
    print()
    print(f"{'Strategy':<30} {'Candidates':<12} {'Final Labels':<12}")
    print("─" * 70)
    print(f"{'1. Neighborhood':<30} {result1.selection_metadata['n_candidates']:<12} {len(result1.labels):<12}")
    print(f"{'2. Label Match':<30} {result2.selection_metadata['n_candidates']:<12} {len(result2.labels):<12}")
    print(f"{'3. Popular + Neighborhood':<30} {result3.selection_metadata['n_candidates']:<12} {len(result3.labels):<12}")
    print(f"{'4. Intersection':<30} {result4.selection_metadata['n_candidates']:<12} {len(result4.labels):<12}")
    print(f"{'5. Weighted':<30} {result5.selection_metadata['n_candidates']:<12} {len(result5.labels):<12}")
    print()
    
    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print()
    print("✓ Neighborhood: High recall, good for coverage")
    print("✓ Label Match: Direct semantic relevance")
    print("✓ Density Filter: Focus on popular or rare labels")
    print("✓ Intersection: High precision, lower recall")
    print("✓ Weighted: Balanced approach, customizable")
    print()
    print("Choose strategy based on your use case:")
    print("  - Exploration → Union with low min_selectors")
    print("  - High precision → Intersection")
    print("  - Balanced → Weighted combination")
    print("  - Domain-specific → Custom selector composition")
    print()
    print("Example completed successfully!")


if __name__ == "__main__":
    demonstrate_selectors()
