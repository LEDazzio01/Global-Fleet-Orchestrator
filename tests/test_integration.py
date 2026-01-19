"""
Integration tests for the data pipeline.

Verifies that generate_data.py produces data that forecast_engine.py
can consume without errors.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import pandas as pd
import numpy as np


class TestDataPipelineIntegration:
    """Integration tests for the data generation and consumption pipeline."""
    
    def test_generated_csv_has_required_columns(self) -> None:
        """Test that generated CSV has all required columns."""
        # Read the existing telemetry data
        csv_path = Path("telemetry_data.csv")
        if not csv_path.exists():
            pytest.skip("telemetry_data.csv not found")
        
        df = pd.read_csv(csv_path)
        
        required_columns = [
            "timestamp",
            "region",
            "temperature_c",
            "it_load_mw",
            "carbon_intensity_gco2",
            "water_usage_l",
        ]
        
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_generated_csv_has_valid_data_types(self) -> None:
        """Test that generated CSV has valid data types."""
        csv_path = Path("telemetry_data.csv")
        if not csv_path.exists():
            pytest.skip("telemetry_data.csv not found")
        
        df = pd.read_csv(csv_path)
        
        # Timestamp should be parseable
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        assert df["timestamp"].dtype == "datetime64[ns]"
        
        # Numeric columns should be numeric
        assert pd.api.types.is_numeric_dtype(df["temperature_c"])
        assert pd.api.types.is_numeric_dtype(df["it_load_mw"])
        assert pd.api.types.is_numeric_dtype(df["carbon_intensity_gco2"])
        assert pd.api.types.is_numeric_dtype(df["water_usage_l"])
    
    def test_generated_csv_has_expected_regions(self) -> None:
        """Test that generated CSV contains expected regions."""
        csv_path = Path("telemetry_data.csv")
        if not csv_path.exists():
            pytest.skip("telemetry_data.csv not found")
        
        df = pd.read_csv(csv_path)
        expected_regions = {"Arizona", "Ireland", "Wyoming"}
        actual_regions = set(df["region"].unique())
        
        assert expected_regions == actual_regions
    
    def test_temperature_values_realistic(self) -> None:
        """Test that temperature values are within realistic bounds."""
        csv_path = Path("telemetry_data.csv")
        if not csv_path.exists():
            pytest.skip("telemetry_data.csv not found")
        
        df = pd.read_csv(csv_path)
        
        # Temperatures should be within -50 to +60 range
        assert df["temperature_c"].min() >= -50
        assert df["temperature_c"].max() <= 60
    
    def test_it_load_values_positive(self) -> None:
        """Test that IT load values are positive."""
        csv_path = Path("telemetry_data.csv")
        if not csv_path.exists():
            pytest.skip("telemetry_data.csv not found")
        
        df = pd.read_csv(csv_path)
        
        assert (df["it_load_mw"] >= 0).all()
    
    def test_water_usage_non_negative(self) -> None:
        """Test that water usage is non-negative."""
        csv_path = Path("telemetry_data.csv")
        if not csv_path.exists():
            pytest.skip("telemetry_data.csv not found")
        
        df = pd.read_csv(csv_path)
        
        assert (df["water_usage_l"] >= 0).all()


class TestModelConsumesData:
    """Tests that the model can consume generated data."""
    
    def test_model_can_load(self) -> None:
        """Test that the trained model can be loaded."""
        import joblib
        
        model_path = Path("conformal_model.pkl")
        if not model_path.exists():
            pytest.skip("conformal_model.pkl not found")
        
        model = joblib.load(model_path)
        
        # Model should have predict_interval method
        assert hasattr(model, "predict_interval")
    
    def test_model_can_predict_on_data(self) -> None:
        """Test that model can make predictions on generated data."""
        import joblib
        import pickle
        
        model_path = Path("conformal_model.pkl")
        region_map_path = Path("region_map.pkl")
        csv_path = Path("telemetry_data.csv")
        
        if not all(p.exists() for p in [model_path, region_map_path, csv_path]):
            pytest.skip("Required files not found")
        
        model = joblib.load(model_path)
        with open(region_map_path, "rb") as f:
            region_map = pickle.load(f)
        
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["region_encoded"] = df["region"].map(region_map)
        df["hour"] = df["timestamp"].dt.hour
        
        # Take a sample for Arizona
        az_data = df[df["region"] == "Arizona"].head(10)
        features = az_data[["region_encoded", "hour", "temperature_c"]]
        
        # Should not raise
        predictions, intervals = model.predict_interval(features)
        
        assert len(predictions) == len(features)
        assert intervals.shape[0] == len(features)
        assert intervals.shape[1] == 2  # lower and upper bounds
    
    def test_predictions_are_reasonable(self) -> None:
        """Test that model predictions are within reasonable range."""
        import joblib
        import pickle
        
        model_path = Path("conformal_model.pkl")
        region_map_path = Path("region_map.pkl")
        csv_path = Path("telemetry_data.csv")
        
        if not all(p.exists() for p in [model_path, region_map_path, csv_path]):
            pytest.skip("Required files not found")
        
        model = joblib.load(model_path)
        with open(region_map_path, "rb") as f:
            region_map = pickle.load(f)
        
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["region_encoded"] = df["region"].map(region_map)
        df["hour"] = df["timestamp"].dt.hour
        
        az_data = df[df["region"] == "Arizona"].head(10)
        features = az_data[["region_encoded", "hour", "temperature_c"]]
        
        predictions, _ = model.predict_interval(features)
        
        # Predictions should be reasonable temperatures
        assert predictions.min() > -50
        assert predictions.max() < 100


class TestTelemetryValidation:
    """Tests for telemetry data validation with Pydantic models."""
    
    def test_csv_passes_pydantic_validation(self) -> None:
        """Test that CSV data passes Pydantic validation."""
        csv_path = Path("telemetry_data.csv")
        if not csv_path.exists():
            pytest.skip("telemetry_data.csv not found")
        
        from src.models.telemetry import TelemetryDataset
        
        # Should not raise ValidationError
        dataset = TelemetryDataset.from_csv(str(csv_path))
        
        assert len(dataset.records) > 0
        assert "Arizona" in dataset.get_regions()
    
    def test_invalid_temperature_raises_error(self) -> None:
        """Test that invalid temperature values are rejected."""
        from pydantic import ValidationError
        from src.models.telemetry import TelemetryRecord
        from datetime import datetime
        
        with pytest.raises(ValidationError):
            TelemetryRecord(
                timestamp=datetime.now(),
                region="Arizona",
                temperature_c=100.0,  # Too high
                it_load_mw=50.0,
                carbon_intensity_gco2=300.0,
                water_usage_l=100.0,
            )
