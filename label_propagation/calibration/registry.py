"""
Calibration Registry

Manages calibration models for multiple labels.
Handles storage, versioning, and retrieval of calibrators.
"""

from typing import Dict, Optional, List
import os
import json
from pathlib import Path
from label_propagation.calibration.isotonic import IsotonicCalibrator, CalibrationMetadata


class CalibrationRegistry:
    """
    Registry for managing per-label calibration models.
    
    Provides:
    - Centralized storage of calibrators
    - Versioning support
    - Metadata management
    """
    
    def __init__(self, registry_path: str):
        """
        Initialize calibration registry.
        
        Args:
            registry_path: Root directory for storing calibration models
        """
        self.registry_path = Path(registry_path)
        self.calibrators: Dict[str, IsotonicCalibrator] = {}
        self.metadata: Dict[str, CalibrationMetadata] = {}
        
        # Create registry directory
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Load index
        self._load_index()
    
    def _get_calibrator_path(self, label_id: str, version: str = "latest") -> Path:
        """Get file path for a calibrator."""
        label_dir = self.registry_path / label_id
        if version == "latest":
            return label_dir / "calibrator_latest.pkl"
        else:
            return label_dir / f"calibrator_{version}.pkl"
    
    def _get_index_path(self) -> Path:
        """Get path to registry index file."""
        return self.registry_path / "index.json"
    
    def _load_index(self):
        """Load registry index from disk."""
        index_path = self._get_index_path()
        if index_path.exists():
            with open(index_path, "r") as f:
                index_data = json.load(f)
                
                # Reconstruct metadata
                for label_id, meta_dict in index_data.items():
                    self.metadata[label_id] = CalibrationMetadata(**meta_dict)
    
    def _save_index(self):
        """Save registry index to disk."""
        index_data = {
            label_id: {
                "label_id": meta.label_id,
                "d_min": meta.d_min,
                "c_max": meta.c_max,
                "n_samples": meta.n_samples,
                "version": meta.version,
            }
            for label_id, meta in self.metadata.items()
        }
        
        with open(self._get_index_path(), "w") as f:
            json.dump(index_data, f, indent=2)
    
    def register(
        self,
        calibrator: IsotonicCalibrator,
        overwrite: bool = False,
    ):
        """
        Register a calibrator for a label.
        
        Args:
            calibrator: Trained IsotonicCalibrator
            overwrite: Whether to overwrite existing calibrator
        """
        if not calibrator.is_trained:
            raise ValueError("Cannot register untrained calibrator")
        
        label_id = calibrator.label_id
        
        if label_id in self.calibrators and not overwrite:
            raise ValueError(
                f"Calibrator for {label_id} already exists. "
                "Use overwrite=True to replace."
            )
        
        # Store in memory
        self.calibrators[label_id] = calibrator
        self.metadata[label_id] = calibrator.metadata
        
        # Save to disk
        label_dir = self.registry_path / label_id
        label_dir.mkdir(parents=True, exist_ok=True)
        
        calibrator_path = self._get_calibrator_path(
            label_id,
            calibrator.metadata.version
        )
        calibrator.save(str(calibrator_path))
        
        # Save as latest
        latest_path = self._get_calibrator_path(label_id, "latest")
        calibrator.save(str(latest_path))
        
        # Update index
        self._save_index()
    
    def get(
        self,
        label_id: str,
        version: str = "latest",
        load_if_missing: bool = True,
    ) -> Optional[IsotonicCalibrator]:
        """
        Get calibrator for a label.
        
        Args:
            label_id: Label identifier
            version: Version to retrieve ("latest" or specific version)
            load_if_missing: Whether to load from disk if not in memory
        
        Returns:
            IsotonicCalibrator or None if not found
        """
        # Check memory cache
        if label_id in self.calibrators:
            return self.calibrators[label_id]
        
        # Load from disk
        if load_if_missing:
            calibrator_path = self._get_calibrator_path(label_id, version)
            if calibrator_path.exists():
                calibrator = IsotonicCalibrator.load(str(calibrator_path))
                self.calibrators[label_id] = calibrator
                return calibrator
        
        return None
    
    def has_calibrator(self, label_id: str) -> bool:
        """Check if calibrator exists for label."""
        return label_id in self.metadata or label_id in self.calibrators
    
    def list_labels(self) -> List[str]:
        """Get list of all labels with calibrators."""
        return list(self.metadata.keys())
    
    def get_metadata(self, label_id: str) -> Optional[CalibrationMetadata]:
        """Get metadata for a label."""
        return self.metadata.get(label_id)
    
    def remove(self, label_id: str):
        """Remove calibrator for a label."""
        if label_id in self.calibrators:
            del self.calibrators[label_id]
        
        if label_id in self.metadata:
            del self.metadata[label_id]
        
        # Remove from disk
        label_dir = self.registry_path / label_id
        if label_dir.exists():
            import shutil
            shutil.rmtree(label_dir)
        
        # Update index
        self._save_index()
    
    def bulk_load(self, label_ids: List[str], version: str = "latest"):
        """
        Load multiple calibrators into memory.
        
        Args:
            label_ids: List of label IDs to load
            version: Version to load
        """
        for label_id in label_ids:
            self.get(label_id, version=version, load_if_missing=True)
    
    def clear_cache(self):
        """Clear in-memory calibrator cache."""
        self.calibrators.clear()
    
    def export_metadata(self) -> Dict[str, Dict]:
        """
        Export all metadata as dictionary.
        
        Returns:
            Dictionary of label metadata
        """
        return {
            label_id: {
                "label_id": meta.label_id,
                "d_min": meta.d_min,
                "c_max": meta.c_max,
                "n_samples": meta.n_samples,
                "version": meta.version,
            }
            for label_id, meta in self.metadata.items()
        }
    
    def __len__(self) -> int:
        """Number of registered calibrators."""
        return len(self.metadata)
    
    def __repr__(self) -> str:
        return (
            f"CalibrationRegistry(path={self.registry_path}, "
            f"n_labels={len(self)})"
        )
