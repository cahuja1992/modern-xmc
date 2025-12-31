"""Tests for calibration module."""

import pytest
import numpy as np
import tempfile
import os
from label_propagation.calibration import (
    IsotonicCalibrator,
    CalibrationSampler,
    LLMRunner,
    CalibrationRegistry,
)


class TestIsotonicCalibrator:
    """Test suite for IsotonicCalibrator."""
    
    def test_train_basic(self):
        """Test basic training."""
        calibrator = IsotonicCalibrator("label_A")
        
        # Create synthetic data
        raw_conf = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        llm_labels = np.array([0, 0, 1, 1, 1])
        
        calibrator.train(raw_conf, llm_labels, d_min=3, c_max=0.9)
        
        assert calibrator.is_trained
        assert calibrator.metadata.d_min == 3
        assert calibrator.metadata.c_max == 0.9
    
    def test_calibrate(self):
        """Test calibration transformation."""
        calibrator = IsotonicCalibrator("label_A")
        
        # Create training data where high confidence = correct
        raw_conf = np.linspace(0, 1, 20)
        llm_labels = (raw_conf > 0.5).astype(int)
        
        calibrator.train(raw_conf, llm_labels, d_min=5, c_max=0.95)
        
        # Test calibration
        result = calibrator.calibrate(raw_confidence=0.8, density=10)
        
        assert 0 <= result <= 1
        assert result <= 0.95  # Should respect c_max
    
    def test_density_adjustment(self):
        """Test density-based adjustment."""
        calibrator = IsotonicCalibrator("label_A")
        
        raw_conf = np.array([0.5, 0.7, 0.9])
        llm_labels = np.array([0, 1, 1])
        
        calibrator.train(raw_conf, llm_labels, d_min=5, c_max=1.0)
        
        # Low density should reduce confidence
        result_low = calibrator.calibrate(0.8, density=2)
        result_high = calibrator.calibrate(0.8, density=10)
        
        assert result_low < result_high
    
    def test_untrained_error(self):
        """Test that calibrating untrained model raises error."""
        calibrator = IsotonicCalibrator("label_A")
        
        with pytest.raises(ValueError, match="not trained"):
            calibrator.calibrate(0.5, density=5)
    
    def test_save_load(self):
        """Test saving and loading calibrator."""
        calibrator = IsotonicCalibrator("label_A")
        
        raw_conf = np.array([0.2, 0.5, 0.8])
        llm_labels = np.array([0, 1, 1])
        calibrator.train(raw_conf, llm_labels)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "calibrator.pkl")
            calibrator.save(path)
            
            loaded = IsotonicCalibrator.load(path)
            
            assert loaded.label_id == "label_A"
            assert loaded.is_trained
            
            # Check calibration gives same result
            result1 = calibrator.calibrate(0.6, density=5)
            result2 = loaded.calibrate(0.6, density=5)
            assert abs(result1 - result2) < 1e-6


class TestCalibrationSampler:
    """Test suite for CalibrationSampler."""
    
    def test_sample_basic(self):
        """Test basic sampling."""
        sampler = CalibrationSampler(n_bins=5, samples_per_bin=3)
        
        candidates = [
            (f"asset_{i}", "label_A", 0.1 * i, [])
            for i in range(20)
        ]
        
        samples = sampler.sample(candidates)
        
        # Should sample across bins
        assert len(samples) > 0
        assert all(s.raw_confidence >= 0.1 for s in samples)
    
    def test_stratified_sampling(self):
        """Test that sampling covers confidence spectrum."""
        sampler = CalibrationSampler(n_bins=10, samples_per_bin=5)
        
        # Create candidates with varying confidence
        candidates = [
            (f"asset_{i}", "label_A", np.random.uniform(0.1, 1.0), [])
            for i in range(100)
        ]
        
        samples = sampler.sample(candidates)
        
        # Check bins are populated
        bin_stats = sampler.get_bin_statistics(samples)
        assert len(bin_stats) > 1  # Should use multiple bins


class TestLLMRunner:
    """Test suite for LLMRunner."""
    
    def test_create_prompt(self):
        """Test prompt creation."""
        runner = LLMRunner()
        
        prompt = runner.create_prompt(
            asset_description="A red car",
            label_id="vehicle",
            label_definition="Motorized transport"
        )
        
        assert "red car" in prompt
        assert "vehicle" in prompt
        assert "Motorized transport" in prompt
    
    def test_parse_response(self):
        """Test response parsing."""
        runner = LLMRunner()
        
        response = '{"semantic_agreement": "YES", "confidence": 0.9}'
        parsed = runner.parse_response(response)
        
        assert parsed["semantic_agreement"] == "YES"
        assert parsed["confidence"] == 0.9
    
    def test_evaluate_single(self):
        """Test single evaluation (with mock client)."""
        runner = LLMRunner(llm_client=None)  # Mock mode
        
        judgment = runner.evaluate_single(
            asset_id="asset_1",
            asset_description="A dog",
            label_id="animal",
        )
        
        assert judgment.asset_id == "asset_1"
        assert judgment.label_id == "animal"


class TestCalibrationRegistry:
    """Test suite for CalibrationRegistry."""
    
    def test_register_and_get(self):
        """Test registering and retrieving calibrators."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = CalibrationRegistry(tmpdir)
            
            # Create and train calibrator
            calibrator = IsotonicCalibrator("label_A")
            raw_conf = np.array([0.2, 0.5, 0.8])
            llm_labels = np.array([0, 1, 1])
            calibrator.train(raw_conf, llm_labels)
            
            # Register
            registry.register(calibrator)
            
            # Retrieve
            retrieved = registry.get("label_A")
            assert retrieved is not None
            assert retrieved.label_id == "label_A"
    
    def test_has_calibrator(self):
        """Test checking calibrator existence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = CalibrationRegistry(tmpdir)
            
            assert not registry.has_calibrator("label_A")
            
            calibrator = IsotonicCalibrator("label_A")
            calibrator.train(np.array([0.3, 0.7]), np.array([0, 1]))
            registry.register(calibrator)
            
            assert registry.has_calibrator("label_A")
    
    def test_persistence(self):
        """Test that registry persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and register
            registry1 = CalibrationRegistry(tmpdir)
            calibrator = IsotonicCalibrator("label_A")
            calibrator.train(np.array([0.3, 0.7]), np.array([0, 1]))
            registry1.register(calibrator)
            
            # Load in new instance
            registry2 = CalibrationRegistry(tmpdir)
            retrieved = registry2.get("label_A")
            
            assert retrieved is not None
            assert retrieved.label_id == "label_A"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
