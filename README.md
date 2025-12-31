# Label Propagation Platform

A geometry-first, classifier-free platform for multi-label assignment using semantic neighborhoods.

## Overview

This platform assigns labels to assets by:
1. Finding semantically similar neighbors in embedding space
2. Aggregating label evidence with rank-weighted voting
3. Calibrating confidence scores per label using LLM judgment

**No classifiers. No softmax. No prediction heads.**

## Key Features

- **Geometry-First**: Semantic similarity drives all behavior
- **Multi-Label by Default**: Assets can receive many labels
- **Scales with Labels**: Adding labels requires no retraining
- **Deterministic & Explainable**: Same inputs always yield same outputs
- **Calibrated Confidence**: Label-specific calibration using LLM-as-judge

## Architecture

```
label_propagation/
├── knn/              # Neighborhood retrieval
├── aggregation/      # Rank-weighted label aggregation
├── calibration/      # LLM-based confidence calibration
├── inference/        # Main propagation engine
└── evaluation/       # Metrics and evaluation
```

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from label_propagation.inference import LabelPropagator
from label_propagation.knn import KNNIndex
import numpy as np

# Load your embeddings and labels
embeddings = np.random.randn(1000, 512)
asset_ids = [f"asset_{i}" for i in range(1000)]
labels = {f"asset_{i}": [f"label_{i%10}"] for i in range(500)}

# Build index
index = KNNIndex(embeddings, asset_ids)

# Create propagator
propagator = LabelPropagator(index, labels, k=50)

# Propagate labels to new asset
new_embedding = np.random.randn(512)
result = propagator.propagate("new_asset", new_embedding)

print(f"Predicted labels: {result['labels']}")
```

## Core Concepts

### Asset
A unit of content with an embedding vector and optional existing labels.

### Neighborhood
The k nearest neighbors of an asset in embedding space (cosine similarity).

### Label Aggregation
Labels are collected from neighbors and weighted by:
- Similarity score
- Rank position (discounted logarithmically)

### Calibration
Raw confidence scores are calibrated per-label using:
- Isotonic regression
- LLM-as-judge validation
- Density adjustment

## Mathematical Foundation

### Rank-Weighted Support

For asset x and label ℓ:

```
support(x,ℓ) = Σ 1[ℓ ∈ L(nᵢ)] · s(x,nᵢ) · w(i)
where w(i) = 1 / log₂(i + 1)
```

### Raw Confidence

```
c_raw(x,ℓ) = support(x,ℓ) / mass(x)
```

### Final Calibrated Confidence

```
conf_final(x,ℓ) = min(c_max(ℓ), g_ℓ(c_raw) · f_density(x,ℓ))
```

Where:
- `g_ℓ`: Isotonic regression model for label ℓ
- `f_density`: Density adjustment factor

## Output Format

```json
{
  "asset_id": "asset_123",
  "labels": [
    {
      "label_id": "L123",
      "confidence": 0.73,
      "supporting_neighbors": [
        {"asset_id": "A1", "similarity": 0.91},
        {"asset_id": "A7", "similarity": 0.88}
      ]
    }
  ]
}
```

## License

MIT
