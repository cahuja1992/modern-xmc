"""
Label Propagation Engine

Main inference engine that orchestrates the complete label propagation pipeline:
1. kNN retrieval
2. Rank-weighted aggregation
3. Calibration (optional)
4. Output formatting
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np

from label_propagation.knn import KNNIndex
from label_propagation.aggregation import RankWeightedAggregator
from label_propagation.calibration import CalibrationRegistry


@dataclass
class LabelResult:
    """Result for a single label."""
    label_id: str
    confidence: float
    raw_confidence: Optional[float] = None
    density: Optional[int] = None
    supporting_neighbors: Optional[List[Dict[str, Any]]] = None


@dataclass
class PropagationResult:
    """Complete propagation result for an asset."""
    asset_id: str
    labels: List[LabelResult]
    metadata: Optional[Dict[str, Any]] = None
    
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
            "metadata": self.metadata,
        }


class LabelPropagator:
    """
    Main label propagation engine.
    
    Combines kNN retrieval, rank-weighted aggregation, and optional
    calibration to assign labels to new assets.
    """
    
    def __init__(
        self,
        knn_index: KNNIndex,
        labels_db: Dict[str, List[str]],
        k: int = 50,
        calibration_registry: Optional[CalibrationRegistry] = None,
        min_confidence: float = 0.0,
        top_k_labels: Optional[int] = None,
        include_neighbors: bool = True,
        max_neighbors_to_return: int = 5,
    ):
        """
        Initialize label propagator.
        
        Args:
            knn_index: KNN index for neighbor retrieval
            labels_db: Mapping from asset_id to label_ids
            k: Number of neighbors to retrieve
            calibration_registry: Optional calibration registry for confidence adjustment
            min_confidence: Minimum confidence threshold for returning labels
            top_k_labels: Maximum number of labels to return per asset
            include_neighbors: Whether to include supporting neighbors in output
            max_neighbors_to_return: Maximum neighbors to include per label
        """
        self.knn_index = knn_index
        self.aggregator = RankWeightedAggregator(labels_db)
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
    ) -> PropagationResult:
        """
        Propagate labels to a new asset.
        
        Args:
            asset_id: Asset identifier
            embedding: Asset embedding vector
            exclude_ids: Optional list of asset IDs to exclude from neighbors
        
        Returns:
            PropagationResult with labeled predictions
        """
        # Step 1: Retrieve neighbors
        neighbors = self.knn_index.search(
            query_embedding=embedding,
            k=self.k,
            exclude_ids=exclude_ids,
        )
        
        if not neighbors:
            return PropagationResult(
                asset_id=asset_id,
                labels=[],
                metadata={"error": "No neighbors found"}
            )
        
        # Step 2: Aggregate labels
        label_scores = self.aggregator.aggregate(
            neighbors=neighbors,
            min_support=0.0,  # Apply threshold after calibration
            top_k_labels=None,  # Apply after calibration
        )
        
        # Step 3: Apply calibration (if available)
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
                # No calibration, use raw confidence
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
            metadata={
                "n_neighbors": len(neighbors),
                "n_labels_raw": len(label_scores),
                "n_labels_final": len(label_results),
            }
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
                iterator = tqdm(assets, desc="Propagating labels")
            except ImportError:
                pass
        
        for asset_data in iterator:
            asset_id = asset_data["asset_id"]
            embedding = asset_data["embedding"]
            exclude_ids = asset_data.get("exclude_ids")
            
            result = self.propagate(
                asset_id=asset_id,
                embedding=embedding,
                exclude_ids=exclude_ids,
            )
            results.append(result)
        
        return results
    
    def get_explanation(
        self,
        asset_id: str,
        embedding: np.ndarray,
        label_id: str,
    ) -> Dict[str, Any]:
        """
        Get detailed explanation for why a label was assigned.
        
        Args:
            asset_id: Asset identifier
            embedding: Asset embedding
            label_id: Label to explain
        
        Returns:
            Dictionary with explanation details
        """
        # Get neighbors
        neighbors = self.knn_index.search(embedding, k=self.k)
        
        # Filter neighbors with this label
        neighbors_with_label = []
        for neighbor in neighbors:
            if label_id in self.aggregator.get_asset_labels(neighbor.asset_id):
                neighbors_with_label.append(neighbor)
        
        # Compute aggregation details
        label_scores = self.aggregator.aggregate(neighbors)
        
        matching_score = None
        for score in label_scores:
            if score.label_id == label_id:
                matching_score = score
                break
        
        if matching_score is None:
            return {
                "label_id": label_id,
                "assigned": False,
                "reason": "Label not found in neighborhood"
            }
        
        # Get calibration info
        calibration_info = None
        if self.calibration_registry:
            calibrator = self.calibration_registry.get(label_id)
            if calibrator and calibrator.is_trained:
                density = len(neighbors_with_label)
                final_conf = calibrator.calibrate(matching_score.raw_confidence, density)
                calibration_info = {
                    "raw_confidence": matching_score.raw_confidence,
                    "calibrated_confidence": final_conf,
                    "density": density,
                    "d_min": calibrator.metadata.d_min,
                    "c_max": calibrator.metadata.c_max,
                }
        
        return {
            "label_id": label_id,
            "assigned": True,
            "raw_confidence": matching_score.raw_confidence,
            "support": matching_score.support,
            "n_supporting_neighbors": len(neighbors_with_label),
            "total_neighbors": len(neighbors),
            "supporting_neighbors": [
                {
                    "asset_id": n.asset_id,
                    "similarity": n.similarity,
                    "rank": n.rank,
                }
                for n in neighbors_with_label[:10]
            ],
            "calibration": calibration_info,
        }
    
    def __repr__(self) -> str:
        has_calib = self.calibration_registry is not None
        return (
            f"LabelPropagator(k={self.k}, "
            f"calibration={'enabled' if has_calib else 'disabled'})"
        )
