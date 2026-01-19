"""
Telemetry data models with Pydantic validation.

These models ensure that all telemetry data is validated before
it reaches the prediction engine, preventing garbage-in-garbage-out.
"""

from datetime import datetime
from typing import List, Optional, Literal
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from src.config import settings


class TelemetryRecord(BaseModel):
    """
    A single telemetry record from a data center.
    
    All fields are validated to ensure they fall within realistic bounds.
    This prevents invalid data from corrupting model predictions.
    """
    
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp of the telemetry reading",
    )
    region: str = Field(
        ...,
        description="Data center region identifier",
        min_length=1,
        max_length=100,
    )
    temperature_c: float = Field(
        ...,
        description="Ambient temperature in Celsius",
    )
    it_load_mw: float = Field(
        ...,
        description="IT load in megawatts",
    )
    carbon_intensity_gco2: float = Field(
        ...,
        description="Carbon intensity in grams CO2 per kWh",
        ge=0.0,
        le=2000.0,
    )
    water_usage_l: float = Field(
        ...,
        description="Water usage in liters",
        ge=0.0,
    )
    
    @field_validator("temperature_c")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is within realistic bounds."""
        if not settings.TEMPERATURE_MIN_C <= v <= settings.TEMPERATURE_MAX_C:
            raise ValueError(
                f"Temperature {v}°C is outside realistic bounds "
                f"[{settings.TEMPERATURE_MIN_C}, {settings.TEMPERATURE_MAX_C}]"
            )
        return v
    
    @field_validator("it_load_mw")
    @classmethod
    def validate_it_load(cls, v: float) -> float:
        """Validate IT load is within realistic bounds."""
        if not settings.IT_LOAD_MIN_MW <= v <= settings.IT_LOAD_MAX_MW:
            raise ValueError(
                f"IT load {v} MW is outside realistic bounds "
                f"[{settings.IT_LOAD_MIN_MW}, {settings.IT_LOAD_MAX_MW}]"
            )
        return v
    
    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class TelemetryDataset(BaseModel):
    """
    A collection of telemetry records with dataset-level validation.
    
    Use this for bulk data ingestion where you need to validate
    an entire dataset before processing.
    """
    
    records: List[TelemetryRecord] = Field(
        ...,
        description="List of telemetry records",
        min_length=1,
    )
    
    @model_validator(mode="after")
    def validate_dataset(self) -> "TelemetryDataset":
        """Validate dataset-level constraints."""
        if len(self.records) < 1:
            raise ValueError("Dataset must contain at least one record")
        return self
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the validated dataset to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with all telemetry records.
        """
        return pd.DataFrame([record.model_dump() for record in self.records])
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "TelemetryDataset":
        """
        Create a TelemetryDataset from a pandas DataFrame.
        
        Args:
            df: DataFrame with telemetry data. Must have columns:
                timestamp, region, temperature_c, it_load_mw,
                carbon_intensity_gco2, water_usage_l
        
        Returns:
            TelemetryDataset: Validated dataset.
        
        Raises:
            ValidationError: If any record fails validation.
        """
        records = []
        for _, row in df.iterrows():
            records.append(TelemetryRecord(
                timestamp=pd.to_datetime(row["timestamp"]),
                region=row["region"],
                temperature_c=row["temperature_c"],
                it_load_mw=row["it_load_mw"],
                carbon_intensity_gco2=row["carbon_intensity_gco2"],
                water_usage_l=row["water_usage_l"],
            ))
        return cls(records=records)
    
    @classmethod
    def from_csv(cls, path: str) -> "TelemetryDataset":
        """
        Load and validate telemetry data from a CSV file.
        
        Args:
            path: Path to the CSV file.
        
        Returns:
            TelemetryDataset: Validated dataset.
        
        Raises:
            ValidationError: If any record fails validation.
            FileNotFoundError: If the file does not exist.
        """
        df = pd.read_csv(path)
        return cls.from_dataframe(df)
    
    def get_regions(self) -> List[str]:
        """Get unique regions in the dataset."""
        return list(set(record.region for record in self.records))
    
    def filter_by_region(self, region: str) -> "TelemetryDataset":
        """
        Filter dataset to only include records from a specific region.
        
        Args:
            region: Region name to filter by.
        
        Returns:
            TelemetryDataset: Filtered dataset.
        """
        filtered = [r for r in self.records if r.region == region]
        if not filtered:
            raise ValueError(f"No records found for region: {region}")
        return TelemetryDataset(records=filtered)
    
    def filter_by_date(self, date: datetime) -> "TelemetryDataset":
        """
        Filter dataset to only include records from a specific date.
        
        Args:
            date: Date to filter by (time component is ignored).
        
        Returns:
            TelemetryDataset: Filtered dataset.
        """
        target_date = date.date() if isinstance(date, datetime) else date
        filtered = [r for r in self.records if r.timestamp.date() == target_date]
        if not filtered:
            raise ValueError(f"No records found for date: {target_date}")
        return TelemetryDataset(records=filtered)
