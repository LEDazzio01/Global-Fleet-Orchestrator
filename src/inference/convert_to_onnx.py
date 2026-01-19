"""
ONNX Model Conversion Utilities.

Provides tools to convert scikit-learn models to ONNX format
for production deployment.

Usage:
    python -m src.inference.convert_to_onnx
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd
import joblib

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


def convert_model_to_onnx(
    input_model_path: Optional[str] = None,
    output_onnx_path: Optional[str] = None,
    output_calibration_path: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Convert a trained conformal model to ONNX format.
    
    This extracts the base estimator from the MAPIE conformal model
    and exports it to ONNX. The calibration residuals are saved
    separately for conformal interval calculation.
    
    Args:
        input_model_path: Path to the pickle model. Defaults to config.
        output_onnx_path: Path for ONNX output. Defaults to config.
        output_calibration_path: Path for calibration data. Defaults to derived.
    
    Returns:
        Tuple of (onnx_path, calibration_path).
    
    Raises:
        ImportError: If skl2onnx is not installed.
        FileNotFoundError: If input model doesn't exist.
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        raise ImportError(
            "skl2onnx not installed. Run: pip install skl2onnx"
        )
    
    # Resolve paths
    input_path = Path(input_model_path or settings.MODEL_PATH)
    output_onnx = Path(output_onnx_path or settings.ONNX_MODEL_PATH)
    output_calib = Path(
        output_calibration_path or 
        str(output_onnx).replace(".onnx", "_calibration.npy")
    )
    
    # Ensure output directory exists
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Model not found: {input_path}")
    
    logger.info(
        "Loading model for ONNX conversion",
        extra={"path": str(input_path)},
    )
    
    # Load the conformal model
    conformal_model = joblib.load(input_path)
    
    # Extract the base estimator (HistGradientBoostingRegressor)
    base_model = conformal_model.estimator_
    
    # Define input type (3 features: region_encoded, hour, temperature_c)
    initial_type = [("float_input", FloatTensorType([None, 3]))]
    
    logger.info("Converting base estimator to ONNX")
    
    # Convert to ONNX
    onnx_model = convert_sklearn(
        base_model,
        initial_types=initial_type,
        target_opset=12,
    )
    
    # Save ONNX model
    with open(output_onnx, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    logger.info(
        "ONNX model saved",
        extra={"path": str(output_onnx)},
    )
    
    # Extract and save calibration residuals
    # These are the conformity scores from the calibration set
    # In MAPIE, these are stored as conformity_scores_
    if hasattr(conformal_model, "conformity_scores_"):
        residuals = np.abs(conformal_model.conformity_scores_).flatten()
    else:
        # Fallback: compute from calibration data if available
        logger.warning("Conformity scores not found, using default interval")
        residuals = np.array([2.5])  # Default ±2.5°C
    
    np.save(output_calib, residuals)
    
    logger.info(
        "Calibration data saved",
        extra={
            "path": str(output_calib),
            "n_samples": len(residuals),
            "quantile_95": float(np.percentile(residuals, 95)),
        },
    )
    
    return str(output_onnx), str(output_calib)


def validate_onnx_model(
    onnx_path: Optional[str] = None,
    calibration_path: Optional[str] = None,
) -> bool:
    """
    Validate that an ONNX model produces correct outputs.
    
    Runs a simple inference test to ensure the model works.
    
    Args:
        onnx_path: Path to ONNX model.
        calibration_path: Path to calibration data.
    
    Returns:
        True if validation passes.
    
    Raises:
        ValueError: If validation fails.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "ONNX Runtime not installed. Run: pip install onnxruntime"
        )
    
    onnx_path = Path(onnx_path or settings.ONNX_MODEL_PATH)
    calib_path = Path(
        calibration_path or 
        str(onnx_path).replace(".onnx", "_calibration.npy")
    )
    
    logger.info("Validating ONNX model")
    
    # Load model
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    
    # Create test input
    test_input = np.array([
        [0, 14, 35.0],  # Arizona, 2 PM, 35°C
        [1, 10, 15.0],  # Ireland, 10 AM, 15°C
        [2, 18, 20.0],  # Wyoming, 6 PM, 20°C
    ], dtype=np.float32)
    
    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: test_input})
    predictions = outputs[0]
    
    # Validate output shape
    if predictions.shape[0] != 3:
        raise ValueError(f"Unexpected output shape: {predictions.shape}")
    
    # Load calibration and validate
    residuals = np.load(calib_path)
    if len(residuals) == 0:
        raise ValueError("Empty calibration data")
    
    logger.info(
        "ONNX validation passed",
        extra={
            "test_predictions": predictions.flatten().tolist(),
            "calibration_samples": len(residuals),
        },
    )
    
    return True


if __name__ == "__main__":
    """Command-line interface for ONNX conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert conformal model to ONNX format"
    )
    parser.add_argument(
        "--input",
        default=None,
        help=f"Input pickle model path (default: {settings.MODEL_PATH})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Output ONNX model path (default: {settings.ONNX_MODEL_PATH})",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the converted model",
    )
    
    args = parser.parse_args()
    
    # Convert
    onnx_path, calib_path = convert_model_to_onnx(
        input_model_path=args.input,
        output_onnx_path=args.output,
    )
    
    print(f"✅ ONNX model saved to: {onnx_path}")
    print(f"✅ Calibration data saved to: {calib_path}")
    
    # Optionally validate
    if args.validate:
        validate_onnx_model(onnx_path, calib_path)
        print("✅ Validation passed")
