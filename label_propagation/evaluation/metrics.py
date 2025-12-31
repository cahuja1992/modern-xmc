"""
Evaluation Metrics

Implements metrics for assessing label propagation quality:
- Precision & Recall
- Coverage lift
- Label distribution analysis
- Stability metrics
"""

from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Result from a metric computation."""
    metric_name: str
    value: float
    per_label: Optional[Dict[str, float]] = None
    metadata: Optional[Dict] = None


class PropagationMetrics:
    """
    Comprehensive metrics for evaluating label propagation.
    
    Computes precision, recall, coverage, and stability metrics
    on golden datasets.
    """
    
    def __init__(self):
        self.results = []
    
    def compute_precision_recall(
        self,
        predictions: Dict[str, List[str]],
        ground_truth: Dict[str, List[str]],
        per_label: bool = True,
    ) -> Tuple[MetricResult, MetricResult]:
        """
        Compute precision and recall.
        
        Args:
            predictions: Map from asset_id to predicted label_ids
            ground_truth: Map from asset_id to true label_ids
            per_label: Whether to compute per-label metrics
        
        Returns:
            Tuple of (precision_result, recall_result)
        """
        # Overall metrics
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        # Per-label metrics
        label_tp = defaultdict(int)
        label_fp = defaultdict(int)
        label_fn = defaultdict(int)
        
        # Compute confusion matrix
        for asset_id in predictions:
            pred_labels = set(predictions.get(asset_id, []))
            true_labels = set(ground_truth.get(asset_id, []))
            
            tp = pred_labels & true_labels
            fp = pred_labels - true_labels
            fn = true_labels - pred_labels
            
            total_tp += len(tp)
            total_fp += len(fp)
            total_fn += len(fn)
            
            if per_label:
                for label in tp:
                    label_tp[label] += 1
                for label in fp:
                    label_fp[label] += 1
                for label in fn:
                    label_fn[label] += 1
        
        # Compute overall precision and recall
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        
        # Compute per-label metrics
        per_label_precision = None
        per_label_recall = None
        
        if per_label:
            per_label_precision = {}
            per_label_recall = {}
            
            all_labels = set(label_tp.keys()) | set(label_fp.keys()) | set(label_fn.keys())
            
            for label in all_labels:
                tp = label_tp[label]
                fp = label_fp[label]
                fn = label_fn[label]
                
                per_label_precision[label] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                per_label_recall[label] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        precision_result = MetricResult(
            metric_name="precision",
            value=precision,
            per_label=per_label_precision,
            metadata={"tp": total_tp, "fp": total_fp}
        )
        
        recall_result = MetricResult(
            metric_name="recall",
            value=recall,
            per_label=per_label_recall,
            metadata={"tp": total_tp, "fn": total_fn}
        )
        
        return precision_result, recall_result
    
    def compute_f1(
        self,
        precision: float,
        recall: float,
    ) -> float:
        """Compute F1 score from precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def compute_coverage_lift(
        self,
        predictions: Dict[str, List[str]],
        seed_labels: Dict[str, List[str]],
        per_label: bool = True,
    ) -> MetricResult:
        """
        Compute coverage lift: increase in label coverage vs seed labels.
        
        Args:
            predictions: Propagated labels
            seed_labels: Original seed labels
            per_label: Whether to compute per-label lift
        
        Returns:
            MetricResult with coverage lift
        """
        # Count labels in seed
        seed_label_counts = defaultdict(int)
        for asset_id, labels in seed_labels.items():
            for label in labels:
                seed_label_counts[label] += 1
        
        # Count labels in predictions
        pred_label_counts = defaultdict(int)
        for asset_id, labels in predictions.items():
            for label in labels:
                pred_label_counts[label] += 1
        
        # Overall lift
        total_seed = sum(seed_label_counts.values())
        total_pred = sum(pred_label_counts.values())
        overall_lift = (total_pred - total_seed) / total_seed if total_seed > 0 else 0.0
        
        # Per-label lift
        per_label_lift = None
        if per_label:
            per_label_lift = {}
            all_labels = set(seed_label_counts.keys()) | set(pred_label_counts.keys())
            
            for label in all_labels:
                seed_count = seed_label_counts[label]
                pred_count = pred_label_counts[label]
                
                if seed_count > 0:
                    lift = (pred_count - seed_count) / seed_count
                else:
                    lift = float('inf') if pred_count > 0 else 0.0
                
                per_label_lift[label] = lift
        
        return MetricResult(
            metric_name="coverage_lift",
            value=overall_lift,
            per_label=per_label_lift,
            metadata={
                "seed_total": total_seed,
                "pred_total": total_pred,
            }
        )
    
    def compute_label_distribution(
        self,
        labels_db: Dict[str, List[str]],
    ) -> Dict[str, int]:
        """
        Compute label frequency distribution.
        
        Args:
            labels_db: Map from asset_id to label_ids
        
        Returns:
            Dictionary mapping label_id to count
        """
        label_counts = defaultdict(int)
        for asset_id, labels in labels_db.items():
            for label in labels:
                label_counts[label] += 1
        return dict(label_counts)
    
    def compute_average_labels_per_asset(
        self,
        labels_db: Dict[str, List[str]],
    ) -> float:
        """
        Compute average number of labels per asset.
        
        Args:
            labels_db: Map from asset_id to label_ids
        
        Returns:
            Average labels per asset
        """
        if not labels_db:
            return 0.0
        
        total_labels = sum(len(labels) for labels in labels_db.values())
        return total_labels / len(labels_db)
    
    def compute_stability(
        self,
        run1_predictions: Dict[str, List[str]],
        run2_predictions: Dict[str, List[str]],
    ) -> MetricResult:
        """
        Compute stability: label overlap between two runs.
        
        Measures how consistent label propagation is across runs.
        
        Args:
            run1_predictions: Predictions from first run
            run2_predictions: Predictions from second run
        
        Returns:
            MetricResult with stability score (Jaccard similarity)
        """
        asset_ids = set(run1_predictions.keys()) & set(run2_predictions.keys())
        
        if not asset_ids:
            return MetricResult("stability", 0.0)
        
        jaccard_scores = []
        
        for asset_id in asset_ids:
            labels1 = set(run1_predictions[asset_id])
            labels2 = set(run2_predictions[asset_id])
            
            intersection = len(labels1 & labels2)
            union = len(labels1 | labels2)
            
            if union > 0:
                jaccard = intersection / union
                jaccard_scores.append(jaccard)
        
        avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0
        
        return MetricResult(
            metric_name="stability",
            value=avg_jaccard,
            metadata={
                "n_assets": len(asset_ids),
                "min_jaccard": min(jaccard_scores) if jaccard_scores else 0.0,
                "max_jaccard": max(jaccard_scores) if jaccard_scores else 0.0,
            }
        )
    
    def compute_confidence_calibration(
        self,
        predictions: List[Tuple[str, str, float]],  # (asset_id, label_id, confidence)
        ground_truth: Dict[str, List[str]],
        n_bins: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Compute confidence calibration: expected vs observed accuracy per bin.
        
        Args:
            predictions: List of (asset_id, label_id, confidence)
            ground_truth: True labels per asset
            n_bins: Number of confidence bins
        
        Returns:
            Dictionary with calibration data
        """
        # Bin predictions by confidence
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_confidences = [[] for _ in range(n_bins)]
        bin_accuracies = [[] for _ in range(n_bins)]
        
        for asset_id, label_id, confidence in predictions:
            bin_idx = np.digitize([confidence], bin_edges)[0] - 1
            bin_idx = max(0, min(bin_idx, n_bins - 1))
            
            # Check if prediction is correct
            true_labels = set(ground_truth.get(asset_id, []))
            is_correct = 1.0 if label_id in true_labels else 0.0
            
            bin_confidences[bin_idx].append(confidence)
            bin_accuracies[bin_idx].append(is_correct)
        
        # Compute average confidence and accuracy per bin
        avg_confidence = []
        avg_accuracy = []
        bin_counts = []
        
        for conf_list, acc_list in zip(bin_confidences, bin_accuracies):
            if conf_list:
                avg_confidence.append(np.mean(conf_list))
                avg_accuracy.append(np.mean(acc_list))
                bin_counts.append(len(conf_list))
            else:
                avg_confidence.append(0.0)
                avg_accuracy.append(0.0)
                bin_counts.append(0)
        
        return {
            "bin_edges": bin_edges.tolist(),
            "avg_confidence": avg_confidence,
            "avg_accuracy": avg_accuracy,
            "bin_counts": bin_counts,
        }


def compute_precision_recall(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> Tuple[float, float]:
    """
    Convenience function to compute precision and recall.
    
    Args:
        predictions: Predicted labels per asset
        ground_truth: True labels per asset
    
    Returns:
        Tuple of (precision, recall)
    """
    metrics = PropagationMetrics()
    prec_result, rec_result = metrics.compute_precision_recall(
        predictions, ground_truth, per_label=False
    )
    return prec_result.value, rec_result.value


def compute_coverage_lift(
    predictions: Dict[str, List[str]],
    seed_labels: Dict[str, List[str]],
) -> float:
    """
    Convenience function to compute coverage lift.
    
    Args:
        predictions: Propagated labels
        seed_labels: Original seed labels
    
    Returns:
        Coverage lift ratio
    """
    metrics = PropagationMetrics()
    result = metrics.compute_coverage_lift(predictions, seed_labels, per_label=False)
    return result.value
