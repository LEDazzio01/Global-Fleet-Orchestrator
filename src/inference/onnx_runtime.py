"""
ONNX Runtime Model Loader.

Provides production-grade model loading using ONNX Runtime,
which is Microsoft's cross-platform inference engine.

Benefits of ONNX:
- Platform-independent (Windows, Linux, macOS)
- Optimized inference performance
- Smaller deployment footprint
- No pickle security concerns
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pickle

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.config import settings
from src.inference.model_loader import ModelLoaderInterface
from src.logging_config import get_logger

logger = get_logger(__name__)

# ONNX Runtime is an optional dependency
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available. Install with: pip install onnxruntime")


class ONNXConformalWrapper:
    """
    Wrapper around ONNX model that provides conformal prediction interface.
    
    This wraps an ONNX point-prediction model and adds conformal intervals
    using precomputed calibration data.
    """
    
    def __init__(
        self,
        session: "ort.InferenceSession",
        calibration_residuals: npt.NDArray[np.float64],
        confidence_level: float = 0.95,
    ) -> None:
        """
        Initialize the wrapper.
        
        Args:
            session: ONNX Runtime inference session.
            calibration_residuals: Absolute residuals from calibration set.
            confidence_level: Confidence level for intervals.
        """
        self._session = session
        self._residuals = calibration_residuals
        self._confidence_level = confidence_level
        
        # Compute quantile for conformal intervals
        n = len(calibration_residuals)
        q = np.ceil((n + 1) * confidence_level) / n
        self._quantile = np.quantile(calibration_residuals, min(q, 1.0))
        
        logger.info(
            "ONNX conformal wrapper initialized",
            extra={
                "calibration_samples": n,
                "confidence_level": confidence_level,
                "interval_width": float(self._quantile * 2),
            }
        )
    
    def predict_interval(
        self,
        X: pd.DataFrame,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Generate predictions with conformal intervals.
        
        Args:
            X: Feature DataFrame with columns [region_encoded, hour, temperature_c].
        
        Returns:
            Tuple of (predictions, intervals) where intervals has shape (n, 2, 1).
        """
        # Get input name from session
        input_name = self._session.get_inputs()[0].name
        
        # Convert to numpy array
        features = X.values.astype(np.float32)
        
        # Run inference
        outputs = self._session.run(None, {input_name: features})
        predictions = outputs[0].flatten().astype(np.float64)
        
        # Apply conformal intervals
        lower = predictions - self._quantile
        upper = predictions + self._quantile
        
        # Shape to match MAPIE output: (n_samples, 2, n_confidence_levels)
        intervals = np.stack([
            lower[:, np.newaxis],
            upper[:, np.newaxis],
        ], axis=1)
        
        return predictions, intervals


class ONNXModelLoader(ModelLoaderInterface):
    """
    Loader for ONNX-format models.
    
    Provides production-grade model loading using ONNX Runtime.
    Falls back to pickle loader if ONNX files are not available.
    
    Example:
        loader = ONNXModelLoader()
        if loader.is_available():
            model = loader.load_model()
            # Model is now using ONNX Runtime for inference
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
        region_map_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the ONNX loader.
        
        Args:
            model_path: Path to ONNX model file.
            calibration_path: Path to calibration residuals file.
            region_map_path: Path to region map file.
        """
        self._model_path = Path(model_path or settings.ONNX_MODEL_PATH)
        self._calibration_path = Path(
            calibration_path or 
            str(self._model_path).replace(".onnx", "_calibration.npy")
        )
        self._region_map_path = Path(region_map_path or settings.REGION_MAP_PATH)
        self._model_cache: Optional[ONNXConformalWrapper] = None
        self._region_map_cache: Optional[Dict[str, int]] = None
    
    def load_model(self) -> ONNXConformalWrapper:
        """
        Load the ONNX model with conformal wrapper.
        
        Returns:
            ONNXConformalWrapper providing predict_interval method.
        
        Raises:
            ImportError: If ONNX Runtime is not installed.
            FileNotFoundError: If model files don't exist.
        """
        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX Runtime not installed. Run: pip install onnxruntime"
            )
        
        if self._model_cache is not None:
            logger.debug("Returning cached ONNX model")
            return self._model_cache
        
        if not self._model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self._model_path}")
        
        if not self._calibration_path.exists():
            raise FileNotFoundError(
                f"Calibration data not found: {self._calibration_path}"
            )
        
        try:
            logger.info(
                "Loading ONNX model",
                extra={"path": str(self._model_path)},
            )
            
            # Create inference session
            session = ort.InferenceSession(
                str(self._model_path),
                providers=["CPUExecutionProvider"],
            )
            
            # Load calibration residuals
            residuals = np.load(self._calibration_path)
            
            # Create wrapper
            self._model_cache = ONNXConformalWrapper(
                session=session,
                calibration_residuals=residuals,
                confidence_level=settings.CONFIDENCE_LEVEL,
            )
            
            logger.info("ONNX model loaded successfully")
            return self._model_cache
            
        except Exception as e:
            logger.error(
                "Failed to load ONNX model",
                extra={"path": str(self._model_path), "error": str(e)},
            )
            raise RuntimeError(f"Failed to load ONNX model: {e}") from e
    
    def load_region_map(self) -> Dict[str, int]:
        """Load region map from pickle."""
        if self._region_map_cache is not None:
            return self._region_map_cache
        
        if not self._region_map_path.exists():
            raise FileNotFoundError(
                f"Region map not found: {self._region_map_path}"
            )
        
        with open(self._region_map_path, "rb") as f:
            self._region_map_cache = pickle.load(f)
        
        return self._region_map_cache
    
    def is_available(self) -> bool:
        """Check if ONNX model and dependencies are available."""
        return (
            ONNX_AVAILABLE and
            self._model_path.exists() and
            self._calibration_path.exists() and
            self._region_map_path.exists()
        )
    
    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._model_cache = None
        self._region_map_cache = None


def get_best_available_loader() -> ModelLoaderInterface:
    """
    Get the best available model loader.
    
    Prefers ONNX for production, falls back to pickle for development.
    
    Returns:
        The best available ModelLoaderInterface implementation.
    """
    from src.inference.model_loader import ModelLoader
    
    # Try ONNX first (production)
    onnx_loader = ONNXModelLoader()
    if onnx_loader.is_available():
        logger.info("Using ONNX model loader (production)")
        return onnx_loader
    
    # Fall back to pickle (development)
    pickle_loader = ModelLoader()
    if pickle_loader.is_available():
        logger.info("Using pickle model loader (development)")
        return pickle_loader
    
    raise FileNotFoundError(
        "No model files found. Run forecast_engine.py to train a model, "
        "or export to ONNX for production use."
    )
