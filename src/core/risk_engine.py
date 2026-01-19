"""
Risk Engine - Assesses thermal risk using conformal predictions.

The RiskEngine is responsible for:
1. Taking telemetry data and a model
2. Generating predictions with confidence intervals
3. Assessing whether the thermal risk exceeds safe thresholds

This follows the Dependency Injection pattern - the model is injected,
not loaded directly, making it easy to swap implementations for testing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Protocol, Tuple, Any
import numpy as np
import numpy.typing as npt
import pandas as pd

from src.config import settings
from src.models.prediction import BatchPredictionResult, ConfidenceInterval, PredictionResult
from src.logging_config import get_logger

logger = get_logger(__name__)


class RiskLevel(str, Enum):
    """Risk levels for thermal assessment."""
    
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskAssessment:
    """
    Result of a risk assessment for a region and time period.
    
    Attributes:
        region: The region assessed.
        risk_level: Categorical risk level.
        max_predicted_temp: Maximum predicted temperature.
        max_upper_bound: Maximum upper bound of confidence interval.
        threshold: The safety threshold used.
        exceeds_threshold: Whether the upper bound exceeds the threshold.
        margin: Degrees of margin (positive = safe, negative = breach).
        recommendations: List of recommended actions.
    """
    
    region: str
    risk_level: RiskLevel
    max_predicted_temp: float
    max_upper_bound: float
    threshold: float
    exceeds_threshold: bool
    margin: float
    recommendations: List[str]
    
    @property
    def is_safe(self) -> bool:
        """Check if the assessment indicates safe conditions."""
        return not self.exceeds_threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "region": self.region,
            "risk_level": self.risk_level.value,
            "max_predicted_temp": self.max_predicted_temp,
            "max_upper_bound": self.max_upper_bound,
            "threshold": self.threshold,
            "exceeds_threshold": self.exceeds_threshold,
            "margin": self.margin,
            "recommendations": self.recommendations,
        }


class ModelProtocol(Protocol):
    """Protocol defining the interface for prediction models."""
    
    def predict_interval(
        self, 
        X: pd.DataFrame
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Generate predictions with confidence intervals.
        
        Args:
            X: Feature DataFrame with columns [region_encoded, hour, temperature_c].
        
        Returns:
            Tuple of (predictions, prediction_intervals).
            prediction_intervals has shape (n_samples, 2, n_confidence_levels).
        """
        ...


class RiskEngine:
    """
    Engine for assessing thermal risk using conformal predictions.
    
    Uses dependency injection for the model, allowing easy testing
    and swapping between pickle and ONNX implementations.
    
    Example:
        model = ModelLoader().load()
        engine = RiskEngine(model)
        assessment = engine.assess_risk(telemetry_data, region="Arizona")
    """
    
    def __init__(
        self,
        model: ModelProtocol,
        region_map: Dict[str, int],
        thermal_limit: Optional[float] = None,
        confidence_level: Optional[float] = None,
    ) -> None:
        """
        Initialize the RiskEngine.
        
        Args:
            model: A model implementing the ModelProtocol.
            region_map: Mapping of region names to encoded values.
            thermal_limit: Temperature threshold in Celsius. Defaults to config.
            confidence_level: Confidence level for intervals. Defaults to config.
        """
        self._model = model
        self._region_map = region_map
        self._thermal_limit = thermal_limit or settings.THERMAL_LIMIT_C
        self._confidence_level = confidence_level or settings.CONFIDENCE_LEVEL
        
        logger.info(
            "RiskEngine initialized",
            extra={
                "thermal_limit": self._thermal_limit,
                "confidence_level": self._confidence_level,
                "regions": list(region_map.keys()),
            }
        )
    
    def predict(
        self,
        data: pd.DataFrame,
        region: str,
        temperature_adjustment: float = 0.0,
    ) -> BatchPredictionResult:
        """
        Generate predictions for a region.
        
        Args:
            data: DataFrame with telemetry data.
            region: Region to predict for.
            temperature_adjustment: Optional temperature offset (for optimization scenarios).
        
        Returns:
            BatchPredictionResult with predictions and confidence intervals.
        
        Raises:
            ValueError: If region is not in the region map.
        """
        if region not in self._region_map:
            raise ValueError(f"Unknown region: {region}. Valid: {list(self._region_map.keys())}")
        
        # Filter and prepare data
        region_data = data[data["region"] == region].copy()
        if region_data.empty:
            raise ValueError(f"No data found for region: {region}")
        
        region_data["region_encoded"] = self._region_map[region]
        region_data["hour"] = pd.to_datetime(region_data["timestamp"]).dt.hour
        
        # Apply temperature adjustment (e.g., from workload shift)
        region_data["temperature_c"] = region_data["temperature_c"] - temperature_adjustment
        
        # Prepare features
        features = region_data[["region_encoded", "hour", "temperature_c"]]
        
        # Generate predictions
        predictions, intervals = self._model.predict_interval(features)
        
        # Extract bounds (shape: n_samples, 2, n_confidence_levels)
        lower_bounds = intervals[:, 0, 0]
        upper_bounds = intervals[:, 1, 0]
        
        logger.debug(
            "Predictions generated",
            extra={
                "region": region,
                "n_samples": len(predictions),
                "temp_adjustment": temperature_adjustment,
                "max_upper_bound": float(np.max(upper_bounds)),
            }
        )
        
        return BatchPredictionResult(
            timestamps=pd.to_datetime(region_data["timestamp"]).tolist(),
            region=region,
            predictions=predictions,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            confidence_level=self._confidence_level,
        )
    
    def assess_risk(
        self,
        data: pd.DataFrame,
        region: str,
        temperature_adjustment: float = 0.0,
    ) -> RiskAssessment:
        """
        Assess thermal risk for a region.
        
        This is the primary method for making scheduling decisions.
        It generates predictions and evaluates them against the safety threshold.
        
        Args:
            data: DataFrame with telemetry data.
            region: Region to assess.
            temperature_adjustment: Optional temperature offset.
        
        Returns:
            RiskAssessment with risk level and recommendations.
        """
        # Generate predictions
        prediction_result = self.predict(data, region, temperature_adjustment)
        
        max_upper = prediction_result.max_upper_bound
        max_pred = prediction_result.max_prediction
        exceeds = max_upper > self._thermal_limit
        margin = self._thermal_limit - max_upper
        
        # Determine risk level
        if margin > 5.0:
            risk_level = RiskLevel.LOW
        elif margin > 2.0:
            risk_level = RiskLevel.MODERATE
        elif margin > 0:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            risk_level, margin, region, temperature_adjustment
        )
        
        assessment = RiskAssessment(
            region=region,
            risk_level=risk_level,
            max_predicted_temp=max_pred,
            max_upper_bound=max_upper,
            threshold=self._thermal_limit,
            exceeds_threshold=exceeds,
            margin=margin,
            recommendations=recommendations,
        )
        
        logger.info(
            "Risk assessment completed",
            extra=assessment.to_dict(),
        )
        
        return assessment
    
    def _generate_recommendations(
        self,
        risk_level: RiskLevel,
        margin: float,
        region: str,
        current_adjustment: float,
    ) -> List[str]:
        """Generate actionable recommendations based on risk assessment."""
        recommendations = []
        
        if risk_level == RiskLevel.CRITICAL:
            recommendations.append(f"URGENT: Block new workloads in {region}")
            shift_needed = abs(margin) / settings.COOLING_FACTOR
            recommendations.append(f"Shift {shift_needed:.0f}% of workload to cooler regions")
            recommendations.append("Consider emergency cooling measures")
        
        elif risk_level == RiskLevel.HIGH:
            recommendations.append(f"WARNING: Reduce workload in {region}")
            shift_needed = (2.0 - margin) / settings.COOLING_FACTOR
            recommendations.append(f"Consider shifting {shift_needed:.0f}% of workload")
        
        elif risk_level == RiskLevel.MODERATE:
            recommendations.append("Monitor closely for temperature increases")
            recommendations.append("Prepare contingency workload migration plan")
        
        else:
            recommendations.append("System operating within safe thermal envelope")
            if current_adjustment > 0:
                recommendations.append(
                    f"Current workload shift of {current_adjustment/settings.COOLING_FACTOR:.0f}% is effective"
                )
        
        return recommendations
    
    @property
    def thermal_limit(self) -> float:
        """Get the current thermal limit threshold."""
        return self._thermal_limit
    
    @property
    def confidence_level(self) -> float:
        """Get the confidence level for predictions."""
        return self._confidence_level
