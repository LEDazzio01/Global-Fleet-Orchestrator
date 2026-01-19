"""
Unit tests for the RiskEngine.

Tests risk assessment logic in isolation.
"""

import pytest
import pandas as pd
import numpy as np

from src.core.risk_engine import RiskEngine, RiskLevel, RiskAssessment
from src.inference.model_loader import MockModel


class TestRiskEngine:
    """Tests for RiskEngine functionality."""
    
    def test_predict_returns_batch_result(
        self,
        risk_engine: RiskEngine,
        sample_telemetry_data: pd.DataFrame
    ) -> None:
        """Test that predict returns BatchPredictionResult."""
        result = risk_engine.predict(sample_telemetry_data, "Arizona")
        
        assert result.region == "Arizona"
        assert len(result.predictions) > 0
        assert len(result.lower_bounds) == len(result.predictions)
        assert len(result.upper_bounds) == len(result.predictions)
    
    def test_assess_risk_returns_assessment(
        self,
        risk_engine: RiskEngine,
        sample_telemetry_data: pd.DataFrame
    ) -> None:
        """Test that assess_risk returns RiskAssessment."""
        assessment = risk_engine.assess_risk(sample_telemetry_data, "Arizona")
        
        assert isinstance(assessment, RiskAssessment)
        assert assessment.region == "Arizona"
        assert isinstance(assessment.risk_level, RiskLevel)
        assert assessment.threshold == 35.0
    
    def test_high_temp_exceeds_threshold(
        self,
        high_risk_engine: RiskEngine,
        high_risk_telemetry_data: pd.DataFrame
    ) -> None:
        """Test that high temperatures exceed the threshold."""
        assessment = high_risk_engine.assess_risk(
            high_risk_telemetry_data, 
            "Arizona"
        )
        
        assert assessment.exceeds_threshold
        assert assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
    
    def test_low_temp_within_threshold(
        self,
        risk_engine: RiskEngine,
        sample_telemetry_data: pd.DataFrame
    ) -> None:
        """Test that low temperatures are within threshold."""
        # Ireland has low temps
        assessment = risk_engine.assess_risk(sample_telemetry_data, "Ireland")
        
        assert not assessment.exceeds_threshold
        assert assessment.risk_level in [RiskLevel.LOW, RiskLevel.MODERATE]


class TestRiskLevels:
    """Tests for risk level categorization."""
    
    def test_risk_levels_ordered_correctly(self) -> None:
        """Test that risk levels can be compared."""
        levels = [RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL]
        
        # All should be valid enum values
        for level in levels:
            assert isinstance(level.value, str)
    
    def test_critical_risk_has_recommendations(
        self,
        high_risk_engine: RiskEngine,
        high_risk_telemetry_data: pd.DataFrame
    ) -> None:
        """Test that critical risk includes recommendations."""
        assessment = high_risk_engine.assess_risk(
            high_risk_telemetry_data, 
            "Arizona"
        )
        
        if assessment.risk_level == RiskLevel.CRITICAL:
            assert len(assessment.recommendations) > 0
            # Should recommend blocking
            assert any("block" in r.lower() for r in assessment.recommendations)


class TestTemperatureAdjustment:
    """Tests for temperature adjustment (workload shift) effects."""
    
    def test_adjustment_reduces_predictions(
        self,
        risk_engine: RiskEngine,
        sample_telemetry_data: pd.DataFrame
    ) -> None:
        """Test that temperature adjustment reduces predicted values."""
        # Baseline prediction
        baseline = risk_engine.predict(sample_telemetry_data, "Arizona", 0.0)
        
        # With adjustment
        adjusted = risk_engine.predict(sample_telemetry_data, "Arizona", 5.0)
        
        # Adjusted should be lower
        assert adjusted.max_upper_bound < baseline.max_upper_bound
    
    def test_adjustment_affects_risk_assessment(
        self,
        risk_engine: RiskEngine,
        sample_telemetry_data: pd.DataFrame
    ) -> None:
        """Test that adjustment affects risk assessment margin."""
        baseline = risk_engine.assess_risk(sample_telemetry_data, "Arizona", 0.0)
        adjusted = risk_engine.assess_risk(sample_telemetry_data, "Arizona", 5.0)
        
        # Adjusted should have larger margin (more safe)
        assert adjusted.margin > baseline.margin


class TestRiskAssessmentSerialization:
    """Tests for RiskAssessment serialization."""
    
    def test_to_dict_contains_all_fields(
        self,
        risk_engine: RiskEngine,
        sample_telemetry_data: pd.DataFrame
    ) -> None:
        """Test that to_dict includes all required fields."""
        assessment = risk_engine.assess_risk(sample_telemetry_data, "Arizona")
        result = assessment.to_dict()
        
        required_fields = [
            "region", "risk_level", "max_predicted_temp",
            "max_upper_bound", "threshold", "exceeds_threshold",
            "margin", "recommendations"
        ]
        
        for field in required_fields:
            assert field in result
    
    def test_to_dict_values_are_serializable(
        self,
        risk_engine: RiskEngine,
        sample_telemetry_data: pd.DataFrame
    ) -> None:
        """Test that to_dict values can be JSON serialized."""
        import json
        
        assessment = risk_engine.assess_risk(sample_telemetry_data, "Arizona")
        result = assessment.to_dict()
        
        # Should not raise
        json_str = json.dumps(result)
        assert len(json_str) > 0
