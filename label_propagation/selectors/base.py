"""
Base Label Selector

Abstract base class for all label selection strategies.
Separates label selection (recall/coverage) from aggregation (precision).
"""

from abc import ABC, abstractmethod
from typing import List, Set, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class SelectionResult:
    """
    Result from a label selection operation.
    
    Contains candidate labels with selection metadata.
    Does NOT include confidence scores (those come from aggregation).
    """
    asset_id: str
    candidate_labels: Set[str]
    metadata: Dict[str, Any]
    
    def __len__(self) -> int:
        """Number of candidate labels."""
        return len(self.candidate_labels)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "asset_id": self.asset_id,
            "candidate_labels": list(self.candidate_labels),
            "n_candidates": len(self.candidate_labels),
            "metadata": self.metadata,
        }


class LabelSelector(ABC):
    """
    Abstract base class for label selection strategies.
    
    Purpose: Determine WHICH labels to consider (recall/coverage focus)
    
    The selector's job is to:
    1. Retrieve candidate labels that might apply to an asset
    2. Focus on maximizing recall and coverage
    3. Filter out obviously irrelevant labels
    
    The selector does NOT:
    - Compute confidence scores (that's the aggregator's job)
    - Make final decisions about which labels to return
    - Consider precision optimization
    
    Design philosophy:
    - Selectors are recall-oriented (cast a wide net)
    - Aggregators are precision-oriented (refine the net)
    """
    
    @abstractmethod
    def select(
        self,
        asset_id: str,
        embedding: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> SelectionResult:
        """
        Select candidate labels for an asset.
        
        Args:
            asset_id: Asset identifier
            embedding: Asset embedding vector
            context: Optional context information
        
        Returns:
            SelectionResult with candidate labels
        """
        pass
    
    def select_batch(
        self,
        assets: List[Dict[str, Any]],
    ) -> List[SelectionResult]:
        """
        Select labels for multiple assets.
        
        Args:
            assets: List of dicts with 'asset_id' and 'embedding'
        
        Returns:
            List of SelectionResult objects
        """
        results = []
        for asset_data in assets:
            result = self.select(
                asset_id=asset_data["asset_id"],
                embedding=asset_data["embedding"],
                context=asset_data.get("context"),
            )
            results.append(result)
        return results
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the selector.
        
        Returns:
            Dictionary with selector statistics
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
