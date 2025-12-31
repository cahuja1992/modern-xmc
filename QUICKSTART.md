# Quick Start Guide

## Installation

```bash
pip install -e .
```

## 5-Minute Tutorial

### Step 1: Prepare Your Data

```python
import numpy as np

# Your embeddings (e.g., from CLIP, BERT, etc.)
embeddings = np.random.randn(1000, 512).astype(np.float32)
asset_ids = [f"asset_{i}" for i in range(1000)]

# Your existing labels (seed data)
labels_db = {
    "asset_0": ["car", "red", "outdoor"],
    "asset_1": ["dog", "pet", "indoor"],
    # ... more labeled assets
}
```

### Step 2: Build kNN Index

```python
from label_propagation import KNNIndex

index = KNNIndex(
    embeddings=embeddings,
    asset_ids=asset_ids,
    use_faiss=True,  # Use FAISS for speed
    normalize=True,  # For cosine similarity
)

# Save for later
index.save("my_index.pkl")
```

### Step 3: Create Propagator

```python
from label_propagation import LabelPropagator

propagator = LabelPropagator(
    knn_index=index,
    labels_db=labels_db,
    k=50,  # Use 50 neighbors
    min_confidence=0.2,  # Threshold
    top_k_labels=10,  # Return top 10 labels
)
```

### Step 4: Propagate Labels

```python
# Single asset
new_embedding = embeddings[500]
result = propagator.propagate("new_asset_1", new_embedding)

for label in result.labels:
    print(f"{label.label_id}: {label.confidence:.3f}")
```

### Step 5: Batch Processing

```python
# Multiple assets
assets = [
    {"asset_id": f"new_{i}", "embedding": embeddings[500+i]}
    for i in range(100)
]

results = propagator.propagate_batch(assets)
```

## Advanced: With Calibration

### Step 1: Sample Calibration Data

```python
from label_propagation.calibration import CalibrationSampler

sampler = CalibrationSampler(n_bins=10, samples_per_bin=10)

# Collect predictions for sampling
candidates = []
for result in propagation_results:
    for label in result.labels:
        candidates.append((
            result.asset_id,
            label.label_id,
            label.raw_confidence,
            label.supporting_neighbors
        ))

samples = sampler.sample(candidates)
```

### Step 2: Run LLM Validation

```python
from label_propagation.calibration import LLMRunner
import openai

# Initialize LLM client
client = openai.OpenAI(api_key="your-key")
llm_runner = LLMRunner(llm_client=client, provider="openai")

# Validate samples
judgments = llm_runner.evaluate_batch(
    samples=[{"asset_id": s.asset_id, "label_id": s.label_id} for s in samples],
    asset_descriptions=descriptions,
    label_definitions=definitions,
)
```

### Step 3: Train Calibrator

```python
from label_propagation.calibration import IsotonicCalibrator
import numpy as np

calibrator = IsotonicCalibrator("my_label")

# Extract training data
raw_confs = np.array([s.raw_confidence for s in samples])
binary_labels = np.array([1 if j.agreement.value == "YES" else 0 for j in judgments])

calibrator.train(
    raw_confidences=raw_confs,
    llm_labels=binary_labels,
    d_min=5,  # Minimum neighbor density
    c_max=0.9,  # Maximum confidence cap
)
```

### Step 4: Register and Use

```python
from label_propagation.calibration import CalibrationRegistry

# Create registry
registry = CalibrationRegistry("./calibration_models")
registry.register(calibrator)

# Create calibrated propagator
calibrated_propagator = LabelPropagator(
    knn_index=index,
    labels_db=labels_db,
    k=50,
    calibration_registry=registry,  # Add calibration
)

# Use as before
result = calibrated_propagator.propagate("new_asset", embedding)
```

## Evaluation

```python
from label_propagation.evaluation import compute_precision_recall

# Collect predictions
predictions = {
    result.asset_id: [l.label_id for l in result.labels]
    for result in results
}

# Compare to ground truth
precision, recall = compute_precision_recall(predictions, ground_truth)
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")
```

## Common Patterns

### Pattern 1: High Precision Mode

```python
propagator = LabelPropagator(
    knn_index=index,
    labels_db=labels_db,
    k=50,
    min_confidence=0.7,  # High threshold
    calibration_registry=registry,  # Use calibration
)
```

### Pattern 2: High Coverage Mode

```python
propagator = LabelPropagator(
    knn_index=index,
    labels_db=labels_db,
    k=100,  # More neighbors
    min_confidence=0.1,  # Low threshold
    top_k_labels=20,  # More labels
)
```

### Pattern 3: Explainable Predictions

```python
explanation = propagator.get_explanation(
    asset_id="my_asset",
    embedding=embedding,
    label_id="car"
)

print(f"Why 'car'?")
print(f"- Raw confidence: {explanation['raw_confidence']:.3f}")
print(f"- Supporting neighbors: {explanation['n_supporting_neighbors']}")
print(f"- Top neighbors:")
for n in explanation['supporting_neighbors'][:3]:
    print(f"  - {n['asset_id']}: similarity={n['similarity']:.3f}")
```

## Performance Tips

### For Large Datasets (>1M assets)

1. **Use FAISS with IVF index**
```python
index = KNNIndex(
    embeddings=embeddings,
    asset_ids=asset_ids,
    use_faiss=True,
    index_type="IVF",
)
```

2. **Batch process everything**
```python
results = propagator.propagate_batch(assets, show_progress=True)
```

3. **Load calibrators lazily**
```python
registry = CalibrationRegistry("./models")
# Calibrators loaded on-demand
```

### For Many Labels (>100K)

1. **Use sparse label storage**
2. **Cache frequently used calibrators**
3. **Consider distributed processing**

## Troubleshooting

### Low confidence scores?
- Check neighbor quality (`propagator.get_explanation()`)
- Verify embedding normalization
- Ensure seed labels are relevant

### Unstable results?
- Use deterministic mode (no randomness)
- Check embedding consistency
- Verify index is not mutated

### Slow performance?
- Enable FAISS
- Use batch processing
- Profile with realistic data

## Next Steps

1. ✅ Run `examples/basic_usage.py` for full example
2. ✅ Run `examples/with_calibration.py` for calibration workflow
3. ✅ Check `CLAUDE.md` for development guide
4. ✅ Review tests in `tests/` for more examples

## Help

- 📖 Full documentation: `README.md`
- 👨‍💻 Developer guide: `CLAUDE.md`
- 🧪 Test examples: `tests/`
- 🎯 PRD+LLD: (original document)
