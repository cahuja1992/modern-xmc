# Label Propagation Platform - V2 Architecture

## 🎯 Major Architectural Improvement

Version 2.0 introduces a **decoupled architecture** that separates label selection (recall/coverage) from label aggregation (precision).

## The Problem with V1

In V1, the system did two things simultaneously:
1. **Selection**: Determine which labels to consider
2. **Aggregation**: Compute confidence scores for labels

This tight coupling limited flexibility and made it hard to optimize recall and precision independently.

## The V2 Solution: Decoupled Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LABEL PROPAGATION V2                      │
└─────────────────────────────────────────────────────────────┘

STAGE 1: SELECTION (Recall/Coverage Focus)
┌─────────────────────────────────────────────────────────────┐
│  SELECTOR                                                    │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────┐ │
│  │ Asset           │  │ Label            │  │ Density    │ │
│  │ Neighborhood    │  │ Match            │  │ Filter     │ │
│  └─────────────────┘  └──────────────────┘  └────────────┘ │
│                                                              │
│  Output: Candidate labels to consider                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
STAGE 2: AGGREGATION (Precision Focus)
┌─────────────────────────────────────────────────────────────┐
│  AGGREGATOR                                                  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Rank-Weighted Voting                                    ││
│  │ • Compute confidence for selected labels                ││
│  │ • Apply mathematical framework                          ││
│  │ • Filter by minimum confidence                          ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  Output: Confidence scores for candidates                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
STAGE 3: CALIBRATION (Optional)
┌─────────────────────────────────────────────────────────────┐
│  CALIBRATOR                                                  │
│  • Isotonic regression per label                            │
│  • Density adjustment                                        │
│  • LLM-as-judge validation                                  │
│                                                              │
│  Output: Calibrated confidence scores                       │
└─────────────────────────────────────────────────────────────┘
```

## Key Benefits

### 1. **Independent Optimization**
- Optimize recall (selection) separately from precision (aggregation)
- Choose selection strategy based on use case
- Aggregation focuses only on refining selected labels

### 2. **Composable Selectors**
- Mix and match multiple selection strategies
- Combine with union, intersection, or weighted approaches
- Create custom selectors for domain-specific needs

### 3. **Better Separation of Concerns**
- **Selectors**: "Which labels might apply?" (cast wide net)
- **Aggregators**: "How confident are we?" (refine the net)
- **Calibrators**: "Adjust for real-world accuracy"

### 4. **Flexibility**
- Easy to add new selection strategies
- Easy to experiment with combinations
- Easy to tune for different scenarios

## Label Selectors

### AssetNeighborhoodSelector
**What it does**: Selects labels from similar assets in embedding space

**Use case**: Classic neighborhood-based propagation

**Parameters**:
- `k`: Number of neighbors
- `min_neighbor_support`: Minimum neighbors that must have label
- `min_similarity`: Similarity threshold
- `max_labels`: Cap on candidates

**Example**:
```python
selector = AssetNeighborhoodSelector(
    knn_index=index,
    labels_db=labels_db,
    k=50,
    min_neighbor_support=3,
)
```

### LabelMatchSelector
**What it does**: Selects labels based on direct label-asset semantic matching

**Use case**: When you have label embeddings and want direct relevance

**Parameters**:
- `label_embeddings`: Map of label_id → embedding
- `min_similarity`: Similarity threshold
- `top_k`: Maximum labels to return
- `normalize`: Whether to normalize for cosine similarity

**Example**:
```python
selector = LabelMatchSelector(
    label_embeddings=label_embeddings,
    min_similarity=0.5,
    top_k=20,
)
```

### LabelDensitySelector
**What it does**: Filters labels by their frequency (popular vs rare)

**Use case**: Focus on specific label density ranges

**Parameters**:
- `min_frequency`: Minimum occurrences
- `max_frequency`: Maximum occurrences
- `min_percentile`: Minimum frequency percentile
- `max_percentile`: Maximum frequency percentile

**Example**:
```python
# Focus on popular labels
selector = LabelDensitySelector(
    labels_db=labels_db,
    min_percentile=75,  # Top 25% most popular
)

# Focus on rare labels
selector = LabelDensitySelector(
    labels_db=labels_db,
    max_percentile=25,  # Bottom 25% least popular
)
```

## Composite Selectors

### UnionSelector
Combines labels from multiple selectors (OR logic)

```python
selector = UnionSelector(
    selectors=[selector1, selector2],
    min_selectors=1,  # Label needs to be selected by at least this many
)
```

### IntersectionSelector
Takes only labels selected by ALL selectors (AND logic)

```python
selector = IntersectionSelector(
    selectors=[selector1, selector2, selector3]
)
```

### WeightedSelector
Combines selectors with weighted voting

```python
selector = WeightedSelector(
    selectors=[selector1, selector2],
    weights=[0.7, 0.3],  # Must sum to 1.0
)
```

## Common Selector Combinations

### High Recall (Exploration)
```python
selector = UnionSelector(
    selectors=[
        AssetNeighborhoodSelector(knn_index, labels_db, k=100),
        LabelMatchSelector(label_embeddings, min_similarity=0.3),
    ],
    min_selectors=1,  # Either selector can suggest
)
```

### High Precision (Conservative)
```python
selector = IntersectionSelector(
    selectors=[
        AssetNeighborhoodSelector(knn_index, labels_db, k=50, min_neighbor_support=5),
        LabelMatchSelector(label_embeddings, min_similarity=0.7),
    ]
)
```

### Balanced (Production)
```python
neighborhood = AssetNeighborhoodSelector(knn_index, labels_db, k=50)
match = LabelMatchSelector(label_embeddings, min_similarity=0.5)
density = LabelDensitySelector(labels_db, min_percentile=25, max_percentile=95)

selector = WeightedSelector(
    selectors=[
        UnionSelector([neighborhood, match], min_selectors=1),
        density,
    ],
    weights=[0.8, 0.2],
)
```

### Domain-Specific (Popular Labels Only)
```python
# First select from neighborhood
neighborhood = AssetNeighborhoodSelector(knn_index, labels_db, k=50)

# Then filter to only popular labels
popular_filter = LabelDensitySelector(labels_db, min_percentile=75)

# Requires both
selector = IntersectionSelector([neighborhood, popular_filter])
```

## Usage Example

```python
from label_propagation import (
    KNNIndex,
    RankWeightedAggregator,
    LabelPropagatorV2,
    AssetNeighborhoodSelector,
    LabelMatchSelector,
    UnionSelector,
)

# Build index
index = KNNIndex(embeddings, asset_ids)
aggregator = RankWeightedAggregator(labels_db)

# Create selector strategy
neighborhood_selector = AssetNeighborhoodSelector(
    knn_index=index,
    labels_db=labels_db,
    k=50,
)

match_selector = LabelMatchSelector(
    label_embeddings=label_embeddings,
    min_similarity=0.5,
)

# Combine strategies
selector = UnionSelector([neighborhood_selector, match_selector])

# Create propagator
propagator = LabelPropagatorV2(
    selector=selector,
    aggregator=aggregator,
    knn_index=index,
    k=50,
)

# Propagate labels
result = propagator.propagate("new_asset", embedding)
```

## Migration from V1 to V2

V1 is still fully supported for backward compatibility. To migrate:

### V1 Code:
```python
from label_propagation import LabelPropagator

propagator = LabelPropagator(
    knn_index=index,
    labels_db=labels_db,
    k=50,
)
```

### V2 Equivalent:
```python
from label_propagation import (
    LabelPropagatorV2,
    AssetNeighborhoodSelector,
    RankWeightedAggregator,
)

selector = AssetNeighborhoodSelector(
    knn_index=index,
    labels_db=labels_db,
    k=50,
)

aggregator = RankWeightedAggregator(labels_db)

propagator = LabelPropagatorV2(
    selector=selector,
    aggregator=aggregator,
    knn_index=index,
    k=50,
)
```

V2 gives you the flexibility to change selection strategy without touching aggregation logic!

## When to Use Which Selector?

| Scenario | Recommended Selector |
|----------|---------------------|
| **General purpose** | AssetNeighborhoodSelector |
| **Have label embeddings** | UnionSelector(Neighborhood + LabelMatch) |
| **High precision needed** | IntersectionSelector(Neighborhood + Match) |
| **Focus on popular labels** | Neighborhood + DensityFilter(high percentile) |
| **Discover rare labels** | Neighborhood + DensityFilter(low percentile) |
| **Cold start (few labels)** | LabelMatchSelector (if you have label embeddings) |
| **Domain expert knowledge** | Custom WeightedSelector combination |

## Performance Considerations

### Selection Stage
- **Neighborhood**: O(log n) with FAISS, O(n) exact
- **LabelMatch**: O(m) where m = number of labels
- **Density**: O(1) (precomputed)
- **Composite**: Sum of component costs

### Aggregation Stage
- Same as V1: O(k · |candidates|)
- But now |candidates| is controlled by selector!

### Memory
- Selectors: Minimal overhead
- Can cache selector results for batch processing

## Testing

Run selector tests:
```bash
pytest tests/test_selectors.py -v
```

Run all tests:
```bash
pytest tests/ -v
```

## Examples

See `examples/selector_combinations.py` for comprehensive demonstrations of all selector strategies.

## Summary

V2 architecture provides:
- ✅ **Decoupled** selection and aggregation
- ✅ **Composable** selector strategies
- ✅ **Flexible** optimization of recall vs precision
- ✅ **Backward compatible** with V1
- ✅ **Extensible** for custom selectors

Choose your selection strategy based on your needs, and let the aggregator handle precision optimization!
