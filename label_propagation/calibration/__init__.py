"""
Calibration Module

Implements LLM-driven label-specific confidence calibration using:
- Isotonic regression
- Density adjustment
- LLM-as-judge validation
"""

from label_propagation.calibration.isotonic import IsotonicCalibrator
from label_propagation.calibration.llm_runner import LLMRunner
from label_propagation.calibration.llm_sampler import CalibrationSampler
from label_propagation.calibration.registry import CalibrationRegistry

__all__ = [
    "IsotonicCalibrator",
    "LLMRunner",
    "CalibrationSampler",
    "CalibrationRegistry",
]
