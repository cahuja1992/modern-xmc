"""
Calibration Example

Demonstrates the full label propagation workflow with LLM-based calibration:
1. Build initial propagator
2. Sample calibration data
3. Run LLM validation (mocked)
4. Train isotonic calibrators
5. Use calibrated propagator
"""

import numpy as np
from label_propagation import KNNIndex, LabelPropagator
from label_propagation.calibration import (
    IsotonicCalibrator,
    CalibrationSampler,
    LLMRunner,
    CalibrationRegistry,
)
import tempfile


def main():
    print("=" * 60)
    print("Label Propagation with Calibration - Example")
    print("=" * 60)
    print()
    
    # Step 1: Create data
    print("Step 1: Creating synthetic data...")
    np.random.seed(42)
    
    n_assets = 500
    embedding_dim = 256
    embeddings = np.random.randn(n_assets, embedding_dim).astype(np.float32)
    asset_ids = [f"asset_{i}" for i in range(n_assets)]
    
    # Create labels
    labels_db = {}
    for i in range(300):
        labels = [f"label_{np.random.randint(0, 20)}" for _ in range(np.random.randint(1, 3))]
        labels_db[f"asset_{i}"] = list(set(labels))
    
    print(f"  - Created {n_assets} assets with {len(labels_db)} labeled")
    print()
    
    # Step 2: Build index and initial propagator
    print("Step 2: Building index and initial propagator...")
    index = KNNIndex(embeddings, asset_ids, use_faiss=False, normalize=True)
    
    propagator_uncalibrated = LabelPropagator(
        knn_index=index,
        labels_db=labels_db,
        k=30,
    )
    print(f"  - Propagator created (uncalibrated)")
    print()
    
    # Step 3: Generate propagation results for calibration sampling
    print("Step 3: Generating propagation results...")
    
    # Propagate to unlabeled assets
    unlabeled_ids = [f"asset_{i}" for i in range(300, 400)]
    propagation_results = {}
    
    for asset_id in unlabeled_ids[:50]:  # Sample 50 for demo
        idx = int(asset_id.split("_")[1])
        embedding = embeddings[idx]
        result = propagator_uncalibrated.propagate(asset_id, embedding)
        propagation_results[asset_id] = result
    
    print(f"  - Generated results for {len(propagation_results)} assets")
    print()
    
    # Step 4: Sample calibration data
    print("Step 4: Sampling calibration data...")
    
    sampler = CalibrationSampler(n_bins=5, samples_per_bin=5, min_confidence=0.2)
    
    # Collect candidates for a specific label
    target_label = "label_5"
    candidates = []
    
    for asset_id, result in propagation_results.items():
        for label in result.labels:
            if label.label_id == target_label:
                candidates.append((
                    asset_id,
                    target_label,
                    label.raw_confidence,
                    label.supporting_neighbors or []
                ))
    
    samples = sampler.sample(candidates)
    print(f"  - Sampled {len(samples)} calibration samples for '{target_label}'")
    
    if samples:
        bin_stats = sampler.get_bin_statistics(samples)
        print(f"  - Bin distribution: {bin_stats}")
    print()
    
    # Step 5: Mock LLM validation
    print("Step 5: Running LLM validation (mocked)...")
    
    llm_runner = LLMRunner(llm_client=None)  # Mock mode
    
    # Create mock asset descriptions
    asset_descriptions = {
        asset_id: f"Description for {asset_id}"
        for asset_id in propagation_results.keys()
    }
    
    # Simulate LLM judgments (in reality, would call actual LLM)
    # For demo: high confidence = YES, low = NO
    llm_judgments = []
    for sample in samples:
        # Mock judgment based on confidence
        if sample.raw_confidence > 0.5:
            agreement = "YES"
            confidence = 0.9
        else:
            agreement = "NO"
            confidence = 0.8
        
        from label_propagation.calibration.llm_runner import LLMJudgment, SemanticAgreement
        judgment = LLMJudgment(
            asset_id=sample.asset_id,
            label_id=sample.label_id,
            agreement=SemanticAgreement(agreement),
            confidence=confidence,
        )
        llm_judgments.append(judgment)
    
    print(f"  - Generated {len(llm_judgments)} LLM judgments")
    print()
    
    # Step 6: Train calibrator
    print("Step 6: Training isotonic calibrator...")
    
    if samples and llm_judgments:
        # Extract data for training
        raw_confidences = np.array([s.raw_confidence for s in samples])
        binary_labels = np.array([
            1 if j.agreement.value == "YES" else 0
            for j in llm_judgments
        ])
        
        # Train calibrator
        calibrator = IsotonicCalibrator(target_label)
        calibrator.train(
            raw_confidences=raw_confidences,
            llm_labels=binary_labels,
            d_min=3,
            c_max=0.9,
            version="v1"
        )
        
        print(f"  - Calibrator trained: {calibrator}")
        print(f"  - Metadata: d_min={calibrator.metadata.d_min}, "
              f"c_max={calibrator.metadata.c_max}, "
              f"n_samples={calibrator.metadata.n_samples}")
        print()
        
        # Step 7: Create calibration registry
        print("Step 7: Creating calibration registry...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = CalibrationRegistry(tmpdir)
            registry.register(calibrator)
            
            print(f"  - Registry created with {len(registry)} calibrators")
            print()
            
            # Step 8: Create calibrated propagator
            print("Step 8: Creating calibrated propagator...")
            
            propagator_calibrated = LabelPropagator(
                knn_index=index,
                labels_db=labels_db,
                k=30,
                calibration_registry=registry,
                min_confidence=0.1,
            )
            
            print(f"  - Calibrated propagator created")
            print()
            
            # Step 9: Compare results
            print("Step 9: Comparing calibrated vs uncalibrated results...")
            print()
            
            test_asset_id = "asset_350"
            test_embedding = embeddings[350]
            
            # Uncalibrated
            result_uncal = propagator_uncalibrated.propagate(test_asset_id, test_embedding)
            
            # Calibrated
            result_cal = propagator_calibrated.propagate(test_asset_id, test_embedding)
            
            print(f"Results for {test_asset_id}:")
            print()
            
            # Find target label in both results
            uncal_label = next((l for l in result_uncal.labels if l.label_id == target_label), None)
            cal_label = next((l for l in result_cal.labels if l.label_id == target_label), None)
            
            if uncal_label and cal_label:
                print(f"  Label: {target_label}")
                print(f"    Uncalibrated confidence: {uncal_label.confidence:.4f}")
                print(f"    Calibrated confidence:   {cal_label.confidence:.4f}")
                print(f"    Raw confidence:          {cal_label.raw_confidence:.4f}")
                print(f"    Density:                 {cal_label.density}")
                print()
            
            print("=" * 60)
            print("Summary")
            print("=" * 60)
            print(f"✓ Trained calibrator for: {target_label}")
            print(f"✓ Calibration samples: {len(samples)}")
            print(f"✓ Calibration improves confidence reliability")
            print()
            print("Calibration example completed successfully!")
    
    else:
        print("  - Not enough samples for calibration demo")
        print("  - Try increasing the number of propagation results")


if __name__ == "__main__":
    main()
