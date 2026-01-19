"""
Pytest fixtures shared across all tests.

Provides mock models, sample data, and other reusable test components.
"""

from datetime import datetime
from typing import Dict, Generator
import pandas as pd
import pytest

from src.config import settings
from src.core.risk_engine import RiskEngine
from src.core.scheduler import Scheduler
from src.inference.model_loader import MockModel, MockModelLoader


@pytest.fixture
def mock_model() -> MockModel:
    """Provide a mock model for testing."""
    return MockModel(base_prediction=30.0, interval_width=5.0)


@pytest.fixture
def mock_model_high_risk() -> MockModel:
    """Provide a mock model that predicts high-risk temperatures."""
    return MockModel(base_prediction=38.0, interval_width=6.0)


@pytest.fixture
def mock_loader() -> MockModelLoader:
    """Provide a mock model loader."""
    return MockModelLoader()


@pytest.fixture
def region_map() -> Dict[str, int]:
    """Provide a standard region map."""
    return {
        "Arizona": 0,
        "Ireland": 1,
        "Wyoming": 2,
    }


@pytest.fixture
def sample_telemetry_data() -> pd.DataFrame:
    """Provide sample telemetry data for testing."""
    # Create 24 hours of data for each region
    timestamps = pd.date_range(
        start="2026-01-15 00:00:00",
        periods=24,
        freq="H",
    )
    
    data = []
    for region, base_temp in [("Arizona", 35.0), ("Ireland", 12.0), ("Wyoming", 15.0)]:
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            # Simulate daily temperature cycle
            temp_offset = 5 * (hour - 12) / 12  # Peak at noon
            temp = base_temp + temp_offset
            
            data.append({
                "timestamp": ts,
                "region": region,
                "temperature_c": round(temp, 2),
                "it_load_mw": round(50 + 10 * (hour / 24), 2),
                "carbon_intensity_gco2": round(300 + (50 if region == "Arizona" else 0), 2),
                "water_usage_l": round(100 if region == "Arizona" else 5, 2),
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def high_risk_telemetry_data() -> pd.DataFrame:
    """Provide telemetry data with high-risk temperatures."""
    timestamps = pd.date_range(
        start="2026-01-15 00:00:00",
        periods=24,
        freq="H",
    )
    
    data = []
    for ts in timestamps:
        hour = ts.hour
        # Arizona with very high temps
        temp = 38.0 + 3 * (hour - 12) / 12  # 35-41°C range
        data.append({
            "timestamp": ts,
            "region": "Arizona",
            "temperature_c": round(temp, 2),
            "it_load_mw": 60.0,
            "carbon_intensity_gco2": 350.0,
            "water_usage_l": 150.0,
        })
        # Safe regions
        for region, base in [("Ireland", 12.0), ("Wyoming", 15.0)]:
            data.append({
                "timestamp": ts,
                "region": region,
                "temperature_c": base,
                "it_load_mw": 50.0,
                "carbon_intensity_gco2": 250.0,
                "water_usage_l": 5.0,
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def risk_engine(mock_model: MockModel, region_map: Dict[str, int]) -> RiskEngine:
    """Provide a RiskEngine with mock model."""
    return RiskEngine(
        model=mock_model,
        region_map=region_map,
        thermal_limit=35.0,
    )


@pytest.fixture
def high_risk_engine(
    mock_model_high_risk: MockModel, 
    region_map: Dict[str, int]
) -> RiskEngine:
    """Provide a RiskEngine with high-risk mock model."""
    return RiskEngine(
        model=mock_model_high_risk,
        region_map=region_map,
        thermal_limit=35.0,
    )


@pytest.fixture
def scheduler(risk_engine: RiskEngine) -> Scheduler:
    """Provide a Scheduler with mock risk engine."""
    return Scheduler(risk_engine=risk_engine)


@pytest.fixture
def high_risk_scheduler(high_risk_engine: RiskEngine) -> Scheduler:
    """Provide a Scheduler that will see high-risk conditions."""
    return Scheduler(risk_engine=high_risk_engine)
