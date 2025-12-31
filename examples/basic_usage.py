"""
Basic Usage Example

Demonstrates the core workflow of the Label Propagation Platform:
1. Create embeddings and labels
2. Build kNN index
3. Create propagator
4. Propagate labels to new assets
"""

import numpy as np
from label_propagation import KNNIndex, LabelPropagator


def main():
    print("=" * 60)
    print("Label Propagation Platform - Basic Usage Example")
    print("=" * 60)
    print()
    
    # Step 1: Create synthetic data
    print("Step 1: Creating synthetic data...")
    np.random.seed(42)
    
    # Embeddings for 1000 assets
    n_assets = 1000
    embedding_dim = 512
    embeddings = np.random.randn(n_assets, embedding_dim).astype(np.float32)
    asset_ids = [f"asset_{i}" for i in range(n_assets)]
    
    # Labels for first 500 assets (seed labels)
    labels_db = {}
    for i in range(500):
        # Assign 1-3 labels per asset
        n_labels = np.random.randint(1, 4)
        labels = [f"label_{np.random.randint(0, 50)}" for _ in range(n_labels)]
        labels_db[f"asset_{i}"] = list(set(labels))
    
    print(f"  - Created {n_assets} embeddings (dim={embedding_dim})")
    print(f"  - Seed labels for {len(labels_db)} assets")
    print()
    
    # Step 2: Build kNN index
    print("Step 2: Building kNN index...")
    index = KNNIndex(
        embeddings=embeddings,
        asset_ids=asset_ids,
        use_faiss=False,  # Use exact search for demo
        normalize=True,
    )
    print(f"  - Index built: {index}")
    print()
    
    # Step 3: Create label propagator
    print("Step 3: Creating label propagator...")
    propagator = LabelPropagator(
        knn_index=index,
        labels_db=labels_db,
        k=50,  # Use 50 nearest neighbors
        min_confidence=0.1,  # Minimum confidence threshold
        top_k_labels=10,  # Return top 10 labels
        include_neighbors=True,
    )
    print(f"  - Propagator created: {propagator}")
    print()
    
    # Step 4: Propagate labels to new assets
    print("Step 4: Propagating labels to new assets...")
    print()
    
    # Example 1: Single asset
    new_asset_id = "new_asset_1"
    new_embedding = embeddings[500]  # Use an unlabeled asset
    
    result = propagator.propagate(new_asset_id, new_embedding)
    
    print(f"Results for {result.asset_id}:")
    print(f"  - Number of labels: {len(result.labels)}")
    print(f"  - Metadata: {result.metadata}")
    print()
    
    if result.labels:
        print("  Top 5 labels:")
        for i, label in enumerate(result.labels[:5], 1):
            print(f"    {i}. {label.label_id}")
            print(f"       Confidence: {label.confidence:.4f}")
            print(f"       Raw confidence: {label.raw_confidence:.4f}")
            print(f"       Density: {label.density}")
            if label.supporting_neighbors:
                print(f"       Top supporting neighbors:")
                for neighbor in label.supporting_neighbors[:3]:
                    print(f"         - {neighbor['asset_id']} "
                          f"(similarity: {neighbor['similarity']:.4f})")
            print()
    
    # Example 2: Batch processing
    print("Step 5: Batch propagation...")
    
    assets_to_label = [
        {
            "asset_id": f"new_asset_{i}",
            "embedding": embeddings[500 + i],
        }
        for i in range(10)
    ]
    
    batch_results = propagator.propagate_batch(assets_to_label, show_progress=False)
    
    print(f"  - Processed {len(batch_results)} assets")
    print(f"  - Average labels per asset: {np.mean([len(r.labels) for r in batch_results]):.2f}")
    print()
    
    # Example 3: Explanation
    print("Step 6: Getting explanation for a specific label...")
    
    if result.labels:
        label_to_explain = result.labels[0].label_id
        explanation = propagator.get_explanation(
            new_asset_id,
            new_embedding,
            label_to_explain
        )
        
        print(f"  Explanation for label '{label_to_explain}':")
        print(f"    - Assigned: {explanation['assigned']}")
        print(f"    - Raw confidence: {explanation.get('raw_confidence', 'N/A')}")
        print(f"    - Support: {explanation.get('support', 'N/A')}")
        print(f"    - Supporting neighbors: {explanation.get('n_supporting_neighbors', 0)}/{explanation.get('total_neighbors', 0)}")
        print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✓ Seed assets: {len(labels_db)}")
    print(f"✓ Total assets: {n_assets}")
    print(f"✓ Propagated labels to: {len(batch_results) + 1} new assets")
    print(f"✓ Average labels per new asset: {np.mean([len(r.labels) for r in batch_results + [result]]):.2f}")
    print()
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
