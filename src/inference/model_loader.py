"""
Model Loader Interface and Implementations.

Defines the interface for loading models and provides implementations
for both pickle (.pkl) and ONNX formats.

Follows the Dependency Injection pattern - callers depend on the
interface, not the concrete implementation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple, Union
import pickle

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class ModelLoaderInterface(ABC):
    """
    Abstract interface for model loaders.
    
    Implementations must provide methods to load models and region maps.
    This allows swapping between pickle, ONNX, or mock implementations.
    """
    
    @abstractmethod
    def load_model(self) -> Any:
        """
        Load the prediction model.
        
        Returns:
            A model object with a predict_interval method.
        """
        pass
    
    @abstractmethod
    def load_region_map(self) -> Dict[str, int]:
        """
        Load the region encoding map.
        
        Returns:
            Dictionary mapping region names to encoded integers.
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the required model files are available.
        
        Returns:
            True if model can be loaded, False otherwise.
        """
        pass


class ModelLoader(ModelLoaderInterface):
    """
    Loader for pickle-based models (development/legacy).
    
    Loads the conformal model and region map from pickle files.
    This is the default implementation for backward compatibility.
    
    Example:
        loader = ModelLoader()
        if loader.is_available():
            model = loader.load_model()
            region_map = loader.load_region_map()
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        region_map_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the ModelLoader.
        
        Args:
            model_path: Path to model file. Defaults to config value.
            region_map_path: Path to region map file. Defaults to config value.
        """
        self._model_path = Path(model_path or settings.MODEL_PATH)
        self._region_map_path = Path(region_map_path or settings.REGION_MAP_PATH)
        self._model_cache: Optional[Any] = None
        self._region_map_cache: Optional[Dict[str, int]] = None
    
    def load_model(self) -> Any:
        """
        Load the conformal prediction model from pickle.
        
        Returns:
            The loaded model with predict_interval method.
        
        Raises:
            FileNotFoundError: If model file doesn't exist.
            RuntimeError: If model loading fails.
        """
        if self._model_cache is not None:
            logger.debug("Returning cached model")
            return self._model_cache
        
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model not found: {self._model_path}")
        
        try:
            logger.info(
                "Loading model",
                extra={"path": str(self._model_path)},
            )
            self._model_cache = joblib.load(self._model_path)
            logger.info("Model loaded successfully")
            return self._model_cache
        except Exception as e:
            logger.error(
                "Failed to load model",
                extra={"path": str(self._model_path), "error": str(e)},
            )
            raise RuntimeError(f"Failed to load model: {e}") from e
    
    def load_region_map(self) -> Dict[str, int]:
        """
        Load the region encoding map from pickle.
        
        Returns:
            Dictionary mapping region names to integers.
        
        Raises:
            FileNotFoundError: If region map file doesn't exist.
        """
        if self._region_map_cache is not None:
            logger.debug("Returning cached region map")
            return self._region_map_cache
        
        if not self._region_map_path.exists():
            raise FileNotFoundError(f"Region map not found: {self._region_map_path}")
        
        try:
            logger.info(
                "Loading region map",
                extra={"path": str(self._region_map_path)},
            )
            with open(self._region_map_path, "rb") as f:
                self._region_map_cache = pickle.load(f)
            logger.info(
                "Region map loaded",
                extra={"regions": list(self._region_map_cache.keys())},
            )
            return self._region_map_cache
        except Exception as e:
            logger.error(
                "Failed to load region map",
                extra={"path": str(self._region_map_path), "error": str(e)},
            )
            raise RuntimeError(f"Failed to load region map: {e}") from e
    
    def is_available(self) -> bool:
        """Check if model files exist."""
        return self._model_path.exists() and self._region_map_path.exists()
    
    def clear_cache(self) -> None:
        """Clear the model cache (useful for testing)."""
        self._model_cache = None
        self._region_map_cache = None


class MockModel:
    """
    Mock model for testing purposes.
    
    Returns predictable values for unit testing without
    needing actual trained models.
    """
    
    def __init__(
        self,
        base_prediction: float = 30.0,
        interval_width: float = 5.0,
    ) -> None:
        """
        Initialize the mock model.
        
        Args:
            base_prediction: Base prediction value.
            interval_width: Width of confidence interval.
        """
        self._base = base_prediction
        self._width = interval_width
    
    def predict_interval(
        self, 
        X: pd.DataFrame,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Generate mock predictions.
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Tuple of (predictions, intervals) matching real model output.
        """
        n = len(X)
        
        # Base predictions with some variation based on input temperature
        if "temperature_c" in X.columns:
            predictions = X["temperature_c"].values + 2.0  # Simple offset
        else:
            predictions = np.full(n, self._base)
        
        # Create intervals: shape (n_samples, 2, 1)
        lower = predictions - self._width / 2
        upper = predictions + self._width / 2
        intervals = np.stack([lower[:, np.newaxis], upper[:, np.newaxis]], axis=1)
        
        return predictions, intervals


class MockModelLoader(ModelLoaderInterface):
    """
    Mock model loader for testing.
    
    Provides predictable mock models without file system access.
    
    Example:
        loader = MockModelLoader(base_prediction=40.0)  # High temp
        model = loader.load_model()
        # Now scheduler will see high risk predictions
    """
    
    def __init__(
        self,
        base_prediction: float = 30.0,
        interval_width: float = 5.0,
        regions: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Initialize the mock loader.
        
        Args:
            base_prediction: Base prediction for mock model.
            interval_width: CI width for mock model.
            regions: Region map. Defaults to standard regions.
        """
        self._base = base_prediction
        self._width = interval_width
        self._regions = regions or {
            "Arizona": 0,
            "Ireland": 1,
            "Wyoming": 2,
        }
    
    def load_model(self) -> MockModel:
        """Return a mock model."""
        logger.info("Loading mock model for testing")
        return MockModel(self._base, self._width)
    
    def load_region_map(self) -> Dict[str, int]:
        """Return mock region map."""
        return self._regions.copy()
    
    def is_available(self) -> bool:
        """Mock is always available."""
        return True
