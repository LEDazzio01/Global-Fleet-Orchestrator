"""
Workload Optimizer - Calculates resource efficiency metrics.

Handles the calculation of water savings, carbon impact, and
other optimization metrics when shifting workloads between regions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from src.config import settings, get_region_config
from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """
    Result of workload optimization analysis.
    
    Contains before/after metrics for resources.
    """
    
    source_region: str
    target_region: str
    workload_shift_pct: float
    
    # Water metrics
    baseline_water_l: float
    optimized_water_l: float
    water_saved_l: float
    water_saved_pct: float
    
    # Carbon metrics
    baseline_carbon_kg: float
    optimized_carbon_kg: float
    carbon_delta_kg: float
    
    # Load metrics
    source_load_reduction_mw: float
    target_load_increase_mw: float
    
    @property
    def is_water_efficient(self) -> bool:
        """Check if optimization reduces water usage."""
        return self.water_saved_l > 0
    
    @property
    def is_carbon_efficient(self) -> bool:
        """Check if optimization reduces carbon emissions."""
        return self.carbon_delta_kg < 0
    
    @property
    def is_net_positive(self) -> bool:
        """Check if optimization is beneficial overall."""
        # Water savings weighted more heavily in this context
        return self.is_water_efficient or self.is_carbon_efficient
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "source_region": self.source_region,
            "target_region": self.target_region,
            "workload_shift_pct": self.workload_shift_pct,
            "water_saved_l": self.water_saved_l,
            "water_saved_pct": self.water_saved_pct,
            "carbon_delta_kg": self.carbon_delta_kg,
            "is_net_positive": self.is_net_positive,
        }


class WorkloadOptimizer:
    """
    Optimizer for calculating resource efficiency of workload shifts.
    
    Analyzes the impact of moving workloads between regions on
    water usage, carbon emissions, and other metrics.
    
    Example:
        optimizer = WorkloadOptimizer()
        result = optimizer.calculate_optimization(
            data, 
            source="Arizona", 
            target="Wyoming",
            shift_pct=30.0
        )
        print(f"Water saved: {result.water_saved_l:.0f}L")
    """
    
    def __init__(self) -> None:
        """Initialize the WorkloadOptimizer."""
        logger.info("WorkloadOptimizer initialized")
    
    def calculate_optimization(
        self,
        data: pd.DataFrame,
        source_region: str,
        target_region: str,
        workload_shift_pct: float,
    ) -> OptimizationResult:
        """
        Calculate the optimization impact of shifting workload.
        
        Args:
            data: DataFrame with telemetry data.
            source_region: Region to shift workload FROM.
            target_region: Region to shift workload TO.
            workload_shift_pct: Percentage of workload to shift (0-100).
        
        Returns:
            OptimizationResult with before/after metrics.
        """
        shift_factor = workload_shift_pct / 100.0
        
        # Get source region data
        source_data = data[data["region"] == source_region]
        target_data = data[data["region"] == target_region]
        
        if source_data.empty:
            raise ValueError(f"No data for source region: {source_region}")
        if target_data.empty:
            raise ValueError(f"No data for target region: {target_region}")
        
        # Calculate water metrics
        baseline_water = float(data["water_usage_l"].sum())
        source_water = float(source_data["water_usage_l"].sum())
        water_reduction = source_water * shift_factor
        optimized_water = baseline_water - water_reduction
        water_saved = baseline_water - optimized_water
        water_saved_pct = (water_saved / baseline_water * 100) if baseline_water > 0 else 0
        
        # Calculate carbon metrics
        source_carbon_intensity = float(source_data["carbon_intensity_gco2"].mean())
        target_carbon_intensity = float(target_data["carbon_intensity_gco2"].mean())
        source_load = float(source_data["it_load_mw"].sum())
        
        shifted_load = source_load * shift_factor
        # Carbon = intensity (gCO2/kWh) * load (MW) * hours / 1000 (to get kg)
        # Simplified: using the load as a proxy for energy
        baseline_carbon = source_carbon_intensity * source_load / 1000
        carbon_delta = (target_carbon_intensity - source_carbon_intensity) * shifted_load / 1000
        
        result = OptimizationResult(
            source_region=source_region,
            target_region=target_region,
            workload_shift_pct=workload_shift_pct,
            baseline_water_l=baseline_water,
            optimized_water_l=optimized_water,
            water_saved_l=water_saved,
            water_saved_pct=water_saved_pct,
            baseline_carbon_kg=baseline_carbon,
            optimized_carbon_kg=baseline_carbon + carbon_delta,
            carbon_delta_kg=carbon_delta,
            source_load_reduction_mw=shifted_load,
            target_load_increase_mw=shifted_load,
        )
        
        logger.info(
            "Optimization calculated",
            extra=result.to_dict(),
        )
        
        return result
    
    def apply_shift_to_data(
        self,
        data: pd.DataFrame,
        source_region: str,
        target_region: str,
        workload_shift_pct: float,
    ) -> pd.DataFrame:
        """
        Apply workload shift to a copy of the data.
        
        This creates a modified DataFrame showing what metrics would
        look like after the shift, useful for visualization.
        
        Args:
            data: Original DataFrame with telemetry data.
            source_region: Region to shift workload FROM.
            target_region: Region to shift workload TO.
            workload_shift_pct: Percentage of workload to shift.
        
        Returns:
            Modified DataFrame with adjusted values.
        """
        shifted_data = data.copy()
        shift_factor = workload_shift_pct / 100.0
        
        source_mask = shifted_data["region"] == source_region
        target_mask = shifted_data["region"] == target_region
        
        # Reduce source region metrics
        shifted_data.loc[source_mask, "it_load_mw"] *= (1 - shift_factor * 0.8)
        shifted_data.loc[source_mask, "water_usage_l"] *= (1 - shift_factor)
        
        # Increase target region load
        shifted_data.loc[target_mask, "it_load_mw"] *= (1 + shift_factor * 1.5)
        
        return shifted_data
    
    def get_efficiency_summary(
        self,
        data: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get efficiency summary for all regions.
        
        Returns:
            Dictionary mapping region names to their efficiency metrics.
        """
        summary = {}
        
        for region in data["region"].unique():
            region_data = data[data["region"] == region]
            summary[region] = {
                "total_water_l": float(region_data["water_usage_l"].sum()),
                "avg_carbon_gco2": float(region_data["carbon_intensity_gco2"].mean()),
                "avg_load_mw": float(region_data["it_load_mw"].mean()),
                "avg_temp_c": float(region_data["temperature_c"].mean()),
            }
        
        return summary
