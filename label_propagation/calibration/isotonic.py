"""
Isotonic Regression Calibration

Implements monotonic mapping from raw confidence to calibrated confidence
using isotonic regression trained on LLM-validated samples.
"""

from typing import List, Tuple, Optional
import numpy as np
from sklearn.isotonic import IsotonicRegression
from dataclasses import dataclass
import pickle


@dataclass
class CalibrationMetadata:
    """Metadata for a calibrated label."""
    label_id: str
    d_min: int  # Minimum density threshold
    c_max: float  # Maximum calibrated confidence
    n_samples: int  # Number of calibration samples
    version: str  # Calibration version


class IsotonicCalibrator:
    """
    Label-specific confidence calibration using isotonic regression.
    
    Learns a monotonic mapping: g_ℓ : [0,1] → [0,1]
    that transforms raw confidence into calibrated confidence.
    """
    
    def __init__(self, label_id: str):
        """
        Initialize calibrator for a specific label.
        
        Args:
            label_id: Label identifier
        """
        self.label_id = label_id
        self.model = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        )
        self.is_trained = False
        self.metadata = None
    
    def train(
        self,
        raw_confidences: np.ndarray,
        llm_labels: np.ndarray,
        d_min: int = 3,
        c_max: float = 0.95,
        version: str = "v1",
    ):
        """
        Train isotonic regression model on LLM-validated samples.
        
        Args:
            raw_confidences: Array of raw confidence scores
            llm_labels: Binary array (1 = YES, 0 = NO) from LLM judge
            d_min: Minimum density threshold for this label
            c_max: Maximum calibrated confidence cap
            version: Calibration version string
        """
        if len(raw_confidences) != len(llm_labels):
            raise ValueError("Number of samples must match labels")
        
        if len(raw_confidences) < 2:
            raise ValueError("Need at least 2 samples for calibration")
        
        # Train isotonic regression
        self.model.fit(raw_confidences, llm_labels)
        self.is_trained = True
        
        # Store metadata
        self.metadata = CalibrationMetadata(
            label_id=self.label_id,
            d_min=d_min,
            c_max=c_max,
            n_samples=len(raw_confidences),
            version=version,
        )
    
    def calibrate(
        self,
        raw_confidence: float,
        density: int,
    ) -> float:
        """
        Apply calibration to raw confidence score.
        
        Implements:
            conf_final = min(c_max, g_ℓ(c_raw) · f_density)
        
        where:
            f_density = min(1, density / d_min)
        
        Args:
            raw_confidence: Raw confidence score from aggregation
            density: Number of neighbors with this label
        
        Returns:
            Calibrated confidence score
        """
        if not self.is_trained:
            raise ValueError(f"Calibrator for {self.label_id} not trained")
        
        # Apply isotonic regression
        calibrated = float(self.model.predict([raw_confidence])[0])
        
        # Apply density adjustment
        if self.metadata.d_min > 0:
            density_factor = min(1.0, density / self.metadata.d_min)
            calibrated = calibrated * density_factor
        
        # Apply maximum cap
        calibrated = min(self.metadata.c_max, calibrated)
        
        return calibrated
    
    def save(self, path: str):
        """Save calibrator to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained calibrator")
        
        state = {
            "label_id": self.label_id,
            "model": self.model,
            "metadata": self.metadata,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: str) -> "IsotonicCalibrator":
        """Load calibrator from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        calibrator = cls(state["label_id"])
        calibrator.model = state["model"]
        calibrator.metadata = state["metadata"]
        calibrator.is_trained = True
        
        return calibrator
    
    def get_calibration_curve(
        self,
        n_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get calibration curve for visualization.
        
        Args:
            n_points: Number of points to sample
        
        Returns:
            Tuple of (raw_confidence_values, calibrated_values)
        """
        if not self.is_trained:
            raise ValueError("Calibrator not trained")
        
        raw_values = np.linspace(0, 1, n_points)
        calibrated_values = self.model.predict(raw_values)
        
        return raw_values, calibrated_values
    
    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return f"IsotonicCalibrator(label={self.label_id}, status={status})"
