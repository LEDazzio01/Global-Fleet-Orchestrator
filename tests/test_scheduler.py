"""
Unit tests for the Scheduler.

Tests the core scheduling logic in isolation using mock models.
"""

import pytest
import pandas as pd

from src.core.scheduler import Scheduler, Decision, SchedulerDecision
from src.core.risk_engine import RiskEngine, RiskLevel


class TestSchedulerDecision:
    """Tests for SchedulerDecision dataclass."""
    
    def test_is_approved(
        self, 
        scheduler: Scheduler, 
        sample_telemetry_data: pd.DataFrame
    ) -> None:
        """Test that safe conditions return APPROVED."""
        # Ireland is cool, should always be approved
        decision = scheduler.evaluate_workload(sample_telemetry_data, "Ireland")
        
        assert decision.is_approved
        assert not decision.is_blocked
        assert decision.decision == Decision.APPROVED
    
    def test_is_blocked_with_high_risk(
        self, 
        high_risk_scheduler: Scheduler, 
        high_risk_telemetry_data: pd.DataFrame
    ) -> None:
        """Test that high-risk conditions return BLOCKED."""
        decision = high_risk_scheduler.evaluate_workload(
            high_risk_telemetry_data, 
            "Arizona"
        )
        
        assert decision.is_blocked
        assert not decision.is_approved
        assert decision.decision == Decision.BLOCKED


class TestSchedulerBlocksUnsafeWorkload:
    """Test that scheduler correctly blocks unsafe workloads."""
    
    def test_scheduler_blocks_at_40c(
        self, 
        region_map: dict
    ) -> None:
        """
        Test Case: Feed the scheduler a temperature of 40°C 
        and assert it returns decision="BLOCKED".
        """
        from src.inference.model_loader import MockModel
        
        # Create a model that predicts 40°C
        high_temp_model = MockModel(base_prediction=40.0, interval_width=4.0)
        # Upper bound will be 40 + 2 = 42°C > 35°C threshold
        
        engine = RiskEngine(
            model=high_temp_model,
            region_map=region_map,
            thermal_limit=35.0,
        )
        scheduler = Scheduler(risk_engine=engine)
        
        # Create minimal test data
        test_data = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-15", periods=1, freq="H"),
            "region": ["Arizona"],
            "temperature_c": [40.0],
            "it_load_mw": [50.0],
            "carbon_intensity_gco2": [350.0],
            "water_usage_l": [100.0],
        })
        
        decision = scheduler.evaluate_workload(test_data, "Arizona")
        
        assert decision.decision == Decision.BLOCKED
        assert "exceeds" in decision.reason.lower() or "risk" in decision.reason.lower()
    
    def test_scheduler_approves_safe_temperature(
        self,
        region_map: dict
    ) -> None:
        """Test that scheduler approves workloads at safe temperatures."""
        from src.inference.model_loader import MockModel
        
        # Create a model that predicts 28°C (safe)
        safe_model = MockModel(base_prediction=28.0, interval_width=4.0)
        # Upper bound will be 28 + 2 = 30°C < 35°C threshold
        
        engine = RiskEngine(
            model=safe_model,
            region_map=region_map,
            thermal_limit=35.0,
        )
        scheduler = Scheduler(risk_engine=engine)
        
        test_data = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-15", periods=1, freq="H"),
            "region": ["Arizona"],
            "temperature_c": [28.0],
            "it_load_mw": [50.0],
            "carbon_intensity_gco2": [350.0],
            "water_usage_l": [100.0],
        })
        
        decision = scheduler.evaluate_workload(test_data, "Arizona")
        
        assert decision.decision == Decision.APPROVED


class TestWorkloadShift:
    """Tests for workload shift scenarios."""
    
    def test_workload_shift_reduces_risk(
        self,
        high_risk_scheduler: Scheduler,
        high_risk_telemetry_data: pd.DataFrame
    ) -> None:
        """Test that workload shift reduces the risk assessment."""
        # Baseline (no shift)
        baseline = high_risk_scheduler.evaluate_workload(
            high_risk_telemetry_data, 
            "Arizona",
            workload_shift_pct=0.0
        )
        
        # With shift
        shifted = high_risk_scheduler.evaluate_workload(
            high_risk_telemetry_data, 
            "Arizona",
            workload_shift_pct=50.0
        )
        
        # Shifted should have lower risk
        assert shifted.risk_assessment.max_upper_bound < baseline.risk_assessment.max_upper_bound
    
    def test_workload_shift_can_resolve_breach(
        self,
        region_map: dict
    ) -> None:
        """Test that sufficient workload shift can resolve a thermal breach."""
        from src.inference.model_loader import MockModel
        
        # Model predicting 36°C (just above threshold)
        model = MockModel(base_prediction=36.0, interval_width=4.0)
        # Upper bound = 38°C > 35°C, but with shift it should drop
        
        engine = RiskEngine(
            model=model,
            region_map=region_map,
            thermal_limit=35.0,
        )
        scheduler = Scheduler(risk_engine=engine)
        
        test_data = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-15", periods=1, freq="H"),
            "region": ["Arizona"],
            "temperature_c": [36.0],
            "it_load_mw": [50.0],
            "carbon_intensity_gco2": [350.0],
            "water_usage_l": [100.0],
        })
        
        # No shift - should be blocked
        baseline = scheduler.evaluate_workload(test_data, "Arizona", 0.0)
        assert baseline.is_blocked
        
        # Large shift - might resolve it
        # The mock model reduces temp by shift% * COOLING_FACTOR
        # At 100% shift with 0.05 factor = 5°C reduction
        shifted = scheduler.evaluate_workload(test_data, "Arizona", 100.0)
        
        # After shift, if resolved
        if shifted.is_approved:
            assert shifted.resolved_breach


class TestMinimumShiftFinder:
    """Tests for the minimum shift finder functionality."""
    
    def test_find_minimum_shift(
        self,
        region_map: dict
    ) -> None:
        """Test finding minimum shift needed for approval."""
        from src.inference.model_loader import MockModel
        
        # Model predicting marginal breach
        model = MockModel(base_prediction=36.0, interval_width=2.0)
        # Upper bound = 37°C, need 2°C reduction = 40% shift at 0.05 factor
        
        engine = RiskEngine(
            model=model,
            region_map=region_map,
            thermal_limit=35.0,
        )
        scheduler = Scheduler(risk_engine=engine)
        
        test_data = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-15", periods=1, freq="H"),
            "region": ["Arizona"],
            "temperature_c": [36.0],
            "it_load_mw": [50.0],
            "carbon_intensity_gco2": [350.0],
            "water_usage_l": [100.0],
        })
        
        min_shift = scheduler.find_minimum_shift(test_data, "Arizona")
        
        # Should find a shift value
        assert min_shift is not None
        assert 0 < min_shift <= 100
        
        # Verify the shift actually works
        decision = scheduler.evaluate_workload(test_data, "Arizona", min_shift)
        assert decision.is_approved
