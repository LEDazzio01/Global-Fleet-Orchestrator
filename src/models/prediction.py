"""
Prediction result models.

These models represent the output of the forecasting engine,
including point predictions and confidence intervals.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, field_validator


class ConfidenceInterval(BaseModel):
    """
    A confidence interval for a prediction.
    
    Represents the uncertainty bounds around a point prediction.
    Used by the scheduler to make risk-aware decisions.
    """
    
    lower_bound: float = Field(
        ...,
        description="Lower bound of the confidence interval",
    )
    upper_bound: float = Field(
        ...,
        description="Upper bound of the confidence interval",
    )
    confidence_level: float = Field(
        default=0.95,
        description="Confidence level (e.g., 0.95 for 95%)",
        ge=0.0,
        le=1.0,
    )
    
    @property
    def width(self) -> float:
        """Calculate the width of the confidence interval."""
        return self.upper_bound - self.lower_bound
    
    @property
    def midpoint(self) -> float:
        """Calculate the midpoint of the confidence interval."""
        return (self.lower_bound + self.upper_bound) / 2
    
    def contains(self, value: float) -> bool:
        """Check if a value falls within the confidence interval."""
        return self.lower_bound <= value <= self.upper_bound
    
    @field_validator("upper_bound")
    @classmethod
    def validate_bounds_order(cls, v: float, info) -> float:
        """Ensure upper bound is >= lower bound."""
        lower = info.data.get("lower_bound")
        if lower is not None and v < lower:
            raise ValueError("upper_bound must be >= lower_bound")
        return v


class PredictionResult(BaseModel):
    """
    Result of a thermal forecast prediction.
    
    Contains both the point prediction and the confidence interval,
    along with metadata about when and for what region the prediction was made.
    """
    
    timestamp: datetime = Field(
        ...,
        description="Timestamp for which the prediction was made",
    )
    region: str = Field(
        ...,
        description="Region for which the prediction was made",
    )
    predicted_temperature_c: float = Field(
        ...,
        description="Point prediction of temperature in Celsius",
    )
    confidence_interval: ConfidenceInterval = Field(
        ...,
        description="Confidence interval around the prediction",
    )
    input_temperature_c: float = Field(
        ...,
        description="Input temperature used for the prediction",
    )
    hour: int = Field(
        ...,
        description="Hour of day (0-23)",
        ge=0,
        le=23,
    )
    
    @property
    def max_risk_temperature(self) -> float:
        """Get the maximum risk temperature (upper bound of CI)."""
        return self.confidence_interval.upper_bound
    
    def exceeds_threshold(self, threshold: float) -> bool:
        """
        Check if the upper bound exceeds a safety threshold.
        
        This is the key decision point for the scheduler.
        A conservative approach uses the upper bound, not the point prediction.
        
        Args:
            threshold: Temperature threshold in Celsius.
        
        Returns:
            bool: True if upper bound exceeds threshold.
        """
        return self.confidence_interval.upper_bound > threshold


@dataclass
class BatchPredictionResult:
    """
    Result of batch predictions for multiple timestamps.
    
    Uses a dataclass for efficiency with numpy arrays.
    """
    
    timestamps: List[datetime]
    region: str
    predictions: npt.NDArray[np.float64]
    lower_bounds: npt.NDArray[np.float64]
    upper_bounds: npt.NDArray[np.float64]
    confidence_level: float
    
    def __post_init__(self) -> None:
        """Validate array shapes match."""
        n = len(self.timestamps)
        if len(self.predictions) != n:
            raise ValueError("predictions length must match timestamps")
        if len(self.lower_bounds) != n:
            raise ValueError("lower_bounds length must match timestamps")
        if len(self.upper_bounds) != n:
            raise ValueError("upper_bounds length must match timestamps")
    
    @property
    def max_upper_bound(self) -> float:
        """Get the maximum upper bound across all predictions."""
        return float(np.max(self.upper_bounds))
    
    @property
    def max_prediction(self) -> float:
        """Get the maximum point prediction."""
        return float(np.max(self.predictions))
    
    def any_exceeds_threshold(self, threshold: float) -> bool:
        """Check if any prediction's upper bound exceeds threshold."""
        return bool(np.any(self.upper_bounds > threshold))
    
    def to_prediction_results(self, input_temperatures: List[float]) -> List[PredictionResult]:
        """
        Convert to a list of PredictionResult objects.
        
        Args:
            input_temperatures: List of input temperatures for each prediction.
        
        Returns:
            List of PredictionResult objects.
        """
        results = []
        for i, ts in enumerate(self.timestamps):
            results.append(PredictionResult(
                timestamp=ts,
                region=self.region,
                predicted_temperature_c=float(self.predictions[i]),
                confidence_interval=ConfidenceInterval(
                    lower_bound=float(self.lower_bounds[i]),
                    upper_bound=float(self.upper_bounds[i]),
                    confidence_level=self.confidence_level,
                ),
                input_temperature_c=input_temperatures[i],
                hour=ts.hour,
            ))
        return results
