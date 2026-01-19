"""
Configuration management using Pydantic BaseSettings.

All magic numbers and configurable values are externalized here.
Values can be overridden via environment variables (e.g., GFO_THERMAL_LIMIT_C=40.0).

Usage:
    from src.config import settings
    
    if temperature > settings.THERMAL_LIMIT_C:
        block_workload()
"""

from typing import Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RegionConfig(BaseSettings):
    """Configuration for a single data center region."""
    
    model_config = SettingsConfigDict(extra="allow")
    
    pue_base: float = Field(1.15, description="Power Usage Effectiveness baseline")
    water_factor: float = Field(150.0, description="Liters per degree above threshold")
    carbon_base: float = Field(400.0, description="gCO2/kWh baseline")
    temp_mean: float = Field(30.0, description="Mean temperature in Celsius")
    temp_std: float = Field(8.0, description="Temperature standard deviation")
    cooling_type: str = Field("evaporative", description="Cooling type: evaporative|air|hybrid")


class AppConfig(BaseSettings):
    """
    Main application configuration.
    
    All values can be overridden via environment variables prefixed with 'GFO_'.
    Example: GFO_THERMAL_LIMIT_C=40.0
    """
    
    model_config = SettingsConfigDict(
        env_prefix="GFO_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Thermal Risk Thresholds
    THERMAL_LIMIT_C: float = Field(
        default=35.0,
        description="Maximum safe temperature in Celsius before workload blocking",
        ge=-50.0,
        le=100.0,
    )
    COOLING_FACTOR: float = Field(
        default=0.05,
        description="Temperature reduction per percentage point of workload shift",
        ge=0.0,
        le=1.0,
    )
    
    # Conformal Prediction Settings
    CONFIDENCE_LEVEL: float = Field(
        default=0.95,
        description="Confidence level for prediction intervals (0.0 to 1.0)",
        ge=0.5,
        le=0.99,
    )
    FORECAST_HORIZON_HOURS: int = Field(
        default=24,
        description="How many hours ahead to forecast",
        ge=1,
        le=168,
    )
    
    # Model Paths
    MODEL_PATH: str = Field(
        default="conformal_model.pkl",
        description="Path to the pickled conformal model",
    )
    ONNX_MODEL_PATH: str = Field(
        default="models/thermal_forecast.onnx",
        description="Path to the ONNX model for production inference",
    )
    REGION_MAP_PATH: str = Field(
        default="region_map.pkl",
        description="Path to the region encoding map",
    )
    TELEMETRY_DATA_PATH: str = Field(
        default="telemetry_data.csv",
        description="Path to telemetry CSV data",
    )
    
    # Data Validation Bounds
    TEMPERATURE_MIN_C: float = Field(
        default=-50.0,
        description="Minimum realistic temperature in Celsius",
    )
    TEMPERATURE_MAX_C: float = Field(
        default=60.0,
        description="Maximum realistic temperature in Celsius",
    )
    IT_LOAD_MIN_MW: float = Field(
        default=0.0,
        description="Minimum IT load in megawatts",
    )
    IT_LOAD_MAX_MW: float = Field(
        default=500.0,
        description="Maximum IT load in megawatts",
    )
    
    # Logging Configuration
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level: DEBUG|INFO|WARNING|ERROR|CRITICAL",
    )
    LOG_FORMAT: str = Field(
        default="json",
        description="Log format: json|text",
    )
    
    # Application Settings
    APP_NAME: str = Field(
        default="Global Fleet Orchestrator",
        description="Application name for logging and UI",
    )
    APP_VERSION: str = Field(
        default="2.0.0",
        description="Application version",
    )
    
    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return upper_v
    
    @field_validator("LOG_FORMAT")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format is supported."""
        valid_formats = {"json", "text"}
        lower_v = v.lower()
        if lower_v not in valid_formats:
            raise ValueError(f"LOG_FORMAT must be one of {valid_formats}")
        return lower_v


# Default region configurations matching the original generate_data.py
DEFAULT_REGIONS: Dict[str, RegionConfig] = {
    "Arizona": RegionConfig(
        pue_base=1.15,
        water_factor=150.0,
        carbon_base=400.0,
        temp_mean=30.0,
        temp_std=8.0,
        cooling_type="evaporative",
    ),
    "Ireland": RegionConfig(
        pue_base=1.08,
        water_factor=0.0,
        carbon_base=250.0,
        temp_mean=12.0,
        temp_std=4.0,
        cooling_type="air",
    ),
    "Wyoming": RegionConfig(
        pue_base=1.10,
        water_factor=10.0,
        carbon_base=300.0,
        temp_mean=15.0,
        temp_std=10.0,
        cooling_type="hybrid",
    ),
}


# Singleton instance - import this in other modules
settings = AppConfig()


def get_settings() -> AppConfig:
    """
    Get the application settings instance.
    
    This function is useful for dependency injection patterns.
    
    Returns:
        AppConfig: The application configuration instance.
    """
    return settings


def get_region_config(region_name: str) -> RegionConfig:
    """
    Get configuration for a specific region.
    
    Args:
        region_name: Name of the region (e.g., "Arizona", "Ireland", "Wyoming")
    
    Returns:
        RegionConfig: Configuration for the specified region.
    
    Raises:
        KeyError: If the region is not found in DEFAULT_REGIONS.
    """
    if region_name not in DEFAULT_REGIONS:
        raise KeyError(f"Unknown region: {region_name}. Valid regions: {list(DEFAULT_REGIONS.keys())}")
    return DEFAULT_REGIONS[region_name]
