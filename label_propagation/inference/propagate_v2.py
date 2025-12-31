"""
Label Propagation Engine V2

NEW ARCHITECTURE with decoupled selection and aggregation:
1. SELECTOR: Determines which labels to consider (RECALL/COVERAGE focus)
2. AGGREGATOR: Computes confidence for selected labels (PRECISION focus)
3. CALIBRATOR: Refines confidence scores (optional)

This separation allows:
- Mix and match selection strategies
- Optimize recall and precision independently
- Flexible composition of approaches
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np

from label_propagation.selectors import LabelSelector
from label_propagation.aggregation import RankWeightedAggregator
from label_propagation.calibration import CalibrationRegistry
from label_propagation.knn import KNNIndex


@dataclass
class LabelResult:
    """Result for a single label."""
    label_id: str
    confidence: float
    raw_confidence: Optional[float] = None
    density: Optional[int] = None
    selection_metadata: Optional[Dict[str, Any]] = None
    supporting_neighbors: Optional[List[Dict[str, Any]]] = None


@dataclass
class PropagationResult:
    """Complete propagation result for an asset."""
    asset_id: str
    labels: List[LabelResult]
    selection_metadata: Optional[Dict[str, Any]] = None
    aggregation_metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "asset_id": self.asset_id,
            "labels": [
                {
                    "label_id": label.label_id,
                    "confidence": label.confidence,
                    "raw_confidence": label.raw_confidence,
                    "density": label.density,
                    "supporting_neighbors": label.supporting_neighbors,
                }
                for label in self.labels
            ],
            "selection_metadata": self.selection_metadata,
            "aggregation_metadata": self.aggregation_metadata,
        }


class LabelPropagatorV2:
    """
    Label propagation engine with decoupled selection and aggregation.
    
    NEW ARCHITECTURE:
    - Selector determines WHICH labels to consider (recall/coverage)
    - Aggregator determines confidence for selected labels (precision)
    - Calibrator refines confidence (optional)
    
    This provides maximum flexibility and composability.
    """
    
    def __init__(
        self,
        selector: LabelSelector,
        aggregator: RankWeightedAggregator,
        knn_index: KNNIndex,
        k: int = 50,
        calibration_registry: Optional[CalibrationRegistry] = None,
        min_confidence: float = 0.0,
        top_k_labels: Optional[int] = None,
        include_neighbors: bool = True,
        max_neighbors_to_return: int = 5,
    ):
        """
        Initialize label propagator V2.
        
        Args:
            selector: Label selector (determines candidates)
            aggregator: Label aggregator (computes confidence)
            knn_index: KNN index for neighbor retrieval
            k: Number of neighbors to retrieve
            calibration_registry: Optional calibration registry
            min_confidence: Minimum confidence threshold
            top_k_labels: Maximum labels to return
            include_neighbors: Whether to include supporting neighbors
            max_neighbors_to_return: Maximum neighbors per label
        """
        self.selector = selector
        self.aggregator = aggregator
        self.knn_index = knn_index
        self.k = k
        self.calibration_registry = calibration_registry
        self.min_confidence = min_confidence
        self.top_k_labels = top_k_labels
        self.include_neighbors = include_neighbors
        self.max_neighbors_to_return = max_neighbors_to_return
    
    def propagate(
        self,
        asset_id: str,
        embedding: np.ndarray,
        exclude_ids: Optional[List[str]] = None,
        selection_context: Optional[Dict[str, Any]] = None,
    ) -> PropagationResult:
        """
        Propagate labels to a new asset using 2-stage approach.
        
        STAGE 1: SELECTION (Recall/Coverage)
        - Selector determines candidate labels
        
        STAGE 2: AGGREGATION (Precision)
        - Aggregator computes confidence for candidates
        - Calibrator refines confidence (optional)
        
        Args:
            asset_id: Asset identifier
            embedding: Asset embedding vector
            exclude_ids: Optional list of asset IDs to exclude
            selection_context: Optional context for selector
        
        Returns:
            PropagationResult with labeled predictions
        """
        # STAGE 1: SELECTION
        # Determine which labels to consider (recall/coverage focus)
        selection_result = self.selector.select(
            asset_id=asset_id,
            embedding=embedding,
            context=selection_context,
        )
        
        candidate_labels = selection_result.candidate_labels
        
        if not candidate_labels:
            return PropagationResult(
                asset_id=asset_id,
                labels=[],
                selection_metadata={
                    "n_candidates": 0,
                    "selector_type": selection_result.metadata.get("selector_type"),
                },
                aggregation_metadata={"n_neighbors": 0},
            )
        
        # STAGE 2: AGGREGATION
        # Retrieve neighbors for aggregation
        neighbors = self.knn_index.search(
            query_embedding=embedding,
            k=self.k,
            exclude_ids=exclude_ids,
        )
        
        if not neighbors:
            return PropagationResult(
                asset_id=asset_id,
                labels=[],
                selection_metadata=selection_result.metadata,
                aggregation_metadata={"error": "No neighbors found"},
            )
        
        # Aggregate only for candidate labels (precision focus)
        label_scores = self.aggregator.aggregate(
            neighbors=neighbors,
            candidate_labels=candidate_labels,  # NEW: Filter to candidates
            min_support=0.0,  # Apply threshold after calibration
            top_k_labels=None,  # Apply after calibration
        )
        
        # STAGE 3: CALIBRATION (optional)
        label_results = []
        
        for label_score in label_scores:
            label_id = label_score.label_id
            raw_confidence = label_score.raw_confidence
            
            # Get density
            density = self.aggregator.compute_label_density(neighbors, label_id)
            
            # Apply calibration if available
            if self.calibration_registry and self.calibration_registry.has_calibrator(label_id):
                calibrator = self.calibration_registry.get(label_id)
                if calibrator:
                    final_confidence = calibrator.calibrate(raw_confidence, density)
                else:
                    final_confidence = raw_confidence
            else:
                final_confidence = raw_confidence
            
            # Apply minimum confidence threshold
            if final_confidence < self.min_confidence:
                continue
            
            # Format supporting neighbors
            supporting_neighbors = None
            if self.include_neighbors:
                supporting_neighbors = [
                    {
                        "asset_id": neighbor_id,
                        "similarity": float(sim),
                    }
                    for neighbor_id, sim in label_score.supporting_neighbors[:self.max_neighbors_to_return]
                ]
            
            label_results.append(LabelResult(
                label_id=label_id,
                confidence=final_confidence,
                raw_confidence=raw_confidence,
                density=density,
                selection_metadata=selection_result.metadata,
                supporting_neighbors=supporting_neighbors,
            ))
        
        # Sort by confidence and apply top-k
        label_results.sort(key=lambda x: x.confidence, reverse=True)
        if self.top_k_labels is not None:
            label_results = label_results[:self.top_k_labels]
        
        # Create result
        return PropagationResult(
            asset_id=asset_id,
            labels=label_results,
            selection_metadata={
                "n_candidates": len(candidate_labels),
                "selector_type": selection_result.metadata.get("selector_type"),
                "selection_details": selection_result.metadata,
            },
            aggregation_metadata={
                "n_neighbors": len(neighbors),
                "n_labels_raw": len(label_scores),
                "n_labels_final": len(label_results),
            },
        )
    
    def propagate_batch(
        self,
        assets: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> List[PropagationResult]:
        """
        Propagate labels to multiple assets.
        
        Args:
            assets: List of dicts with 'asset_id' and 'embedding'
            show_progress: Whether to show progress bar
        
        Returns:
            List of PropagationResult objects
        """
        results = []
        
        iterator = assets
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(assets, desc="Propagating labels (V2)")
            except ImportError:
                pass
        
        for asset_data in iterator:
            asset_id = asset_data["asset_id"]
            embedding = asset_data["embedding"]
            exclude_ids = asset_data.get("exclude_ids")
            selection_context = asset_data.get("selection_context")
            
            result = self.propagate(
                asset_id=asset_id,
                embedding=embedding,
                exclude_ids=exclude_ids,
                selection_context=selection_context,
            )
            results.append(result)
        
        return results
    
    def __repr__(self) -> str:
        has_calib = self.calibration_registry is not None
        return (
            f"LabelPropagatorV2(selector={type(self.selector).__name__}, "
            f"aggregator={type(self.aggregator).__name__}, "
            f"k={self.k}, calibration={'enabled' if has_calib else 'disabled'})"
        )
