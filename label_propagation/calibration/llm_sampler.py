"""
Calibration Sample Selection

Selects diverse samples across the confidence spectrum for LLM validation.
Implements stratified sampling to ensure good coverage.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class CalibrationSample:
    """A sample selected for LLM validation."""
    asset_id: str
    label_id: str
    raw_confidence: float
    supporting_neighbors: List[Tuple[str, float]]
    bin_index: int


class CalibrationSampler:
    """
    Sample selector for building calibration datasets.
    
    Uses stratified sampling across confidence bins to ensure
    diverse coverage for isotonic regression training.
    """
    
    def __init__(
        self,
        n_bins: int = 10,
        samples_per_bin: int = 10,
        min_confidence: float = 0.1,
        random_seed: int = 42,
    ):
        """
        Initialize sampler.
        
        Args:
            n_bins: Number of confidence bins
            samples_per_bin: Target samples per bin
            min_confidence: Minimum confidence threshold
            random_seed: Random seed for reproducibility
        """
        self.n_bins = n_bins
        self.samples_per_bin = samples_per_bin
        self.min_confidence = min_confidence
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
    
    def sample(
        self,
        candidates: List[Tuple[str, str, float, List[Tuple[str, float]]]],
    ) -> List[CalibrationSample]:
        """
        Select calibration samples using stratified sampling.
        
        Args:
            candidates: List of (asset_id, label_id, raw_confidence, neighbors)
        
        Returns:
            List of selected CalibrationSample objects
        """
        # Filter by minimum confidence
        candidates = [
            c for c in candidates
            if c[2] >= self.min_confidence
        ]
        
        if not candidates:
            return []
        
        # Create confidence bins
        bin_edges = np.linspace(
            self.min_confidence,
            1.0,
            self.n_bins + 1
        )
        
        # Assign candidates to bins
        binned_candidates = [[] for _ in range(self.n_bins)]
        
        for asset_id, label_id, raw_conf, neighbors in candidates:
            bin_idx = np.digitize([raw_conf], bin_edges)[0] - 1
            bin_idx = max(0, min(bin_idx, self.n_bins - 1))
            
            binned_candidates[bin_idx].append(
                (asset_id, label_id, raw_conf, neighbors, bin_idx)
            )
        
        # Sample from each bin
        selected = []
        
        for bin_idx, bin_candidates in enumerate(binned_candidates):
            if not bin_candidates:
                continue
            
            # Sample up to samples_per_bin from this bin
            n_sample = min(self.samples_per_bin, len(bin_candidates))
            
            # Random sample without replacement
            indices = self.rng.choice(
                len(bin_candidates),
                size=n_sample,
                replace=False
            )
            
            for idx in indices:
                asset_id, label_id, raw_conf, neighbors, bin_idx = bin_candidates[idx]
                selected.append(CalibrationSample(
                    asset_id=asset_id,
                    label_id=label_id,
                    raw_confidence=raw_conf,
                    supporting_neighbors=neighbors,
                    bin_index=bin_idx,
                ))
        
        return selected
    
    def sample_for_label(
        self,
        label_id: str,
        asset_scores: Dict[str, Tuple[float, List[Tuple[str, float]]]],
    ) -> List[CalibrationSample]:
        """
        Sample calibration data for a specific label.
        
        Args:
            label_id: Label to sample for
            asset_scores: Map from asset_id to (raw_confidence, neighbors)
        
        Returns:
            List of CalibrationSample objects
        """
        candidates = [
            (asset_id, label_id, conf, neighbors)
            for asset_id, (conf, neighbors) in asset_scores.items()
        ]
        
        return self.sample(candidates)
    
    def get_bin_statistics(
        self,
        samples: List[CalibrationSample],
    ) -> Dict[int, int]:
        """
        Get sample count per bin.
        
        Args:
            samples: List of calibration samples
        
        Returns:
            Dictionary mapping bin_index to count
        """
        bin_counts = {}
        for sample in samples:
            bin_counts[sample.bin_index] = bin_counts.get(sample.bin_index, 0) + 1
        return bin_counts
    
    def __repr__(self) -> str:
        return (
            f"CalibrationSampler(n_bins={self.n_bins}, "
            f"samples_per_bin={self.samples_per_bin})"
        )
