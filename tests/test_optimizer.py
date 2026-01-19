"""
Unit tests for the WorkloadOptimizer.

Tests resource efficiency calculations.
"""

import pytest
import pandas as pd

from src.core.optimizer import WorkloadOptimizer, OptimizationResult


@pytest.fixture
def optimizer() -> WorkloadOptimizer:
    """Provide a WorkloadOptimizer instance."""
    return WorkloadOptimizer()


class TestOptimizationCalculation:
    """Tests for optimization calculations."""
    
    def test_calculate_optimization_returns_result(
        self,
        optimizer: WorkloadOptimizer,
        sample_telemetry_data: pd.DataFrame
    ) -> None:
        """Test that calculate_optimization returns OptimizationResult."""
        result = optimizer.calculate_optimization(
            sample_telemetry_data,
            source_region="Arizona",
            target_region="Wyoming",
            workload_shift_pct=30.0
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.source_region == "Arizona"
        assert result.target_region == "Wyoming"
        assert result.workload_shift_pct == 30.0
    
    def test_water_savings_are_positive(
        self,
        optimizer: WorkloadOptimizer,
        sample_telemetry_data: pd.DataFrame
    ) -> None:
        """Test that shifting from Arizona saves water."""
        result = optimizer.calculate_optimization(
            sample_telemetry_data,
            source_region="Arizona",
            target_region="Wyoming",
            workload_shift_pct=50.0
        )
        
        # Arizona uses more water than Wyoming
        assert result.water_saved_l > 0
        assert result.water_saved_pct > 0
    
    def test_zero_shift_no_savings(
        self,
        optimizer: WorkloadOptimizer,
        sample_telemetry_data: pd.DataFrame
    ) -> None:
        """Test that zero shift results in zero savings."""
        result = optimizer.calculate_optimization(
            sample_telemetry_data,
            source_region="Arizona",
            target_region="Wyoming",
            workload_shift_pct=0.0
        )
        
        assert result.water_saved_l == 0
        assert result.water_saved_pct == 0


class TestApplyShiftToData:
    """Tests for data transformation with workload shift."""
    
    def test_apply_shift_reduces_source_load(
        self,
        optimizer: WorkloadOptimizer,
        sample_telemetry_data: pd.DataFrame
    ) -> None:
        """Test that apply_shift_to_data reduces source region load."""
        shifted = optimizer.apply_shift_to_data(
            sample_telemetry_data,
            source_region="Arizona",
            target_region="Wyoming",
            workload_shift_pct=50.0
        )
        
        original_az_load = sample_telemetry_data[
            sample_telemetry_data["region"] == "Arizona"
        ]["it_load_mw"].mean()
        
        shifted_az_load = shifted[
            shifted["region"] == "Arizona"
        ]["it_load_mw"].mean()
        
        assert shifted_az_load < original_az_load
    
    def test_apply_shift_increases_target_load(
        self,
        optimizer: WorkloadOptimizer,
        sample_telemetry_data: pd.DataFrame
    ) -> None:
        """Test that apply_shift_to_data increases target region load."""
        shifted = optimizer.apply_shift_to_data(
            sample_telemetry_data,
            source_region="Arizona",
            target_region="Wyoming",
            workload_shift_pct=50.0
        )
        
        original_wy_load = sample_telemetry_data[
            sample_telemetry_data["region"] == "Wyoming"
        ]["it_load_mw"].mean()
        
        shifted_wy_load = shifted[
            shifted["region"] == "Wyoming"
        ]["it_load_mw"].mean()
        
        assert shifted_wy_load > original_wy_load


class TestEfficiencySummary:
    """Tests for efficiency summary generation."""
    
    def test_get_efficiency_summary_all_regions(
        self,
        optimizer: WorkloadOptimizer,
        sample_telemetry_data: pd.DataFrame
    ) -> None:
        """Test that efficiency summary includes all regions."""
        summary = optimizer.get_efficiency_summary(sample_telemetry_data)
        
        assert "Arizona" in summary
        assert "Ireland" in summary
        assert "Wyoming" in summary
    
    def test_efficiency_summary_has_metrics(
        self,
        optimizer: WorkloadOptimizer,
        sample_telemetry_data: pd.DataFrame
    ) -> None:
        """Test that efficiency summary includes expected metrics."""
        summary = optimizer.get_efficiency_summary(sample_telemetry_data)
        
        for region, metrics in summary.items():
            assert "total_water_l" in metrics
            assert "avg_carbon_gco2" in metrics
            assert "avg_load_mw" in metrics
            assert "avg_temp_c" in metrics


class TestOptimizationResultProperties:
    """Tests for OptimizationResult properties."""
    
    def test_is_water_efficient_when_saving_water(
        self,
        optimizer: WorkloadOptimizer,
        sample_telemetry_data: pd.DataFrame
    ) -> None:
        """Test is_water_efficient property."""
        result = optimizer.calculate_optimization(
            sample_telemetry_data,
            source_region="Arizona",
            target_region="Wyoming",
            workload_shift_pct=50.0
        )
        
        if result.water_saved_l > 0:
            assert result.is_water_efficient
    
    def test_to_dict_serializable(
        self,
        optimizer: WorkloadOptimizer,
        sample_telemetry_data: pd.DataFrame
    ) -> None:
        """Test that to_dict produces serializable output."""
        import json
        
        result = optimizer.calculate_optimization(
            sample_telemetry_data,
            source_region="Arizona",
            target_region="Wyoming",
            workload_shift_pct=30.0
        )
        
        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)
        assert len(json_str) > 0
