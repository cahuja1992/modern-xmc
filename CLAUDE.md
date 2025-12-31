# Label Propagation Platform - Developer Guide

## Project Overview

This is a **geometry-first, classifier-free** multi-label assignment platform that uses semantic neighborhoods for label propagation. No classifiers, no softmax, no prediction heads.

## Core Architecture

```
label_propagation/
├── knn/              # Neighborhood retrieval (cosine similarity)
├── aggregation/      # Rank-weighted label aggregation
├── calibration/      # LLM-based confidence calibration
├── inference/        # Main propagation engine
└── evaluation/       # Metrics and evaluation
```

## Key Mathematical Concepts

### Rank-Weighted Support
```
support(x,ℓ) = Σ 1[ℓ ∈ L(nᵢ)] · s(x,nᵢ) · w(i)
where w(i) = 1 / log₂(i + 1)
```

### Raw Confidence
```
c_raw(x,ℓ) = support(x,ℓ) / mass(x)
```

### Calibrated Confidence
```
conf_final(x,ℓ) = min(c_max(ℓ), g_ℓ(c_raw) · f_density(x,ℓ))
```

## Development Workflow

### Setting Up

```bash
# Install dependencies
pip install -e .

# Run tests
pytest tests/ -v

# Run examples
python examples/basic_usage.py
python examples/with_calibration.py
```

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/test_knn.py -v

# With coverage
pytest --cov=label_propagation --cov-report=html
```

### Code Style

- Use Black for formatting: `black label_propagation/`
- Use type hints where possible
- Write docstrings for all public APIs
- Follow PEP 8 conventions

## Key Implementation Details

### 1. Determinism
- All operations must be deterministic
- Use fixed random seeds in tests
- No randomness in inference

### 2. Scalability
- Use FAISS for ANN with large datasets
- Batch processing for propagation
- Lazy loading of calibrators

### 3. Explainability
- Always track supporting neighbors
- Provide explanation methods
- Store metadata with results

## Common Tasks

### Adding a New Aggregation Method

1. Create new file in `label_propagation/aggregation/`
2. Implement aggregation interface
3. Add tests in `tests/test_aggregation.py`
4. Update documentation

### Adding a New Calibration Method

1. Create calibrator class in `label_propagation/calibration/`
2. Implement `train()` and `calibrate()` methods
3. Add registry support
4. Add tests

### Modifying the Index

1. Update `label_propagation/knn/index.py`
2. Ensure backward compatibility
3. Update save/load methods
4. Add migration if needed

## Testing Guidelines

### Unit Tests
- Test each component in isolation
- Mock external dependencies
- Use small synthetic data

### Integration Tests
- Test end-to-end workflows
- Use realistic data sizes
- Verify determinism

### Performance Tests
- Benchmark on large datasets
- Profile memory usage
- Test scalability limits

## Debugging Tips

### Neighborhood Issues
```python
# Inspect neighbors
neighbors = index.search(embedding, k=50)
for n in neighbors:
    print(f"{n.asset_id}: sim={n.similarity}, rank={n.rank}")
```

### Aggregation Issues
```python
# Check label support
scores = aggregator.aggregate(neighbors)
for s in scores:
    print(f"{s.label_id}: conf={s.raw_confidence}, support={s.support}")
```

### Calibration Issues
```python
# Visualize calibration curve
raw, cal = calibrator.get_calibration_curve()
import matplotlib.pyplot as plt
plt.plot(raw, cal)
plt.plot([0,1], [0,1], 'r--')  # Perfect calibration
plt.show()
```

## Performance Optimization

### For Large Datasets (>1M assets)
1. Use FAISS with IVF index
2. Enable batch processing
3. Use lazy calibrator loading
4. Consider distributed processing

### For Many Labels (>100K)
1. Use sparse label storage
2. Cache frequently used calibrators
3. Parallelize calibration training
4. Use incremental updates

## Common Pitfalls

❌ **Don't:**
- Mutate embeddings after index creation
- Use random operations in inference
- Skip normalization for cosine similarity
- Forget to handle edge cases (empty neighbors, missing labels)

✓ **Do:**
- Normalize embeddings for cosine similarity
- Cache index and calibrators
- Use batch processing for efficiency
- Profile before optimizing

## Contact & Support

For questions about implementation details:
1. Check this guide first
2. Review the PRD+LLD document
3. Check existing tests for examples
4. Review code comments and docstrings

## Version History

- v1.0.0 - Initial implementation
  - kNN index with FAISS support
  - Rank-weighted aggregation
  - Isotonic calibration
  - LLM-as-judge integration
  - Comprehensive metrics
