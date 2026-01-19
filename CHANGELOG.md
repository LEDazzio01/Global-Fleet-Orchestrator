# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2026-01-19

### Added
- **New Modular Architecture** - Extracted business logic into `src/core/` modules
  - `src/core/scheduler.py` - Workload scheduling decisions (APPROVE/BLOCK)
  - `src/core/risk_engine.py` - Thermal risk assessment with conformal predictions
  - `src/core/optimizer.py` - Water and carbon efficiency calculations
- **Inference Layer** - New `src/inference/` module for model management
  - `src/inference/model_loader.py` - Model loading with dependency injection pattern
  - `src/inference/onnx_runtime.py` - ONNX Runtime inference support
  - `src/inference/convert_to_onnx.py` - Model export utility for production deployment
- **Typed Data Models** - Pydantic-based models in `src/models/`
  - `src/models/telemetry.py` - Telemetry data validation
  - `src/models/prediction.py` - Prediction result models
- **Configuration Management** - `src/config.py` with Pydantic settings
- **Structured Logging** - `src/logging_config.py` with JSON logging support
- **Refactored Application** - `app_refactored.py` using new architecture

### Testing
- Unit tests: `test_scheduler.py`, `test_risk_engine.py`, `test_optimizer.py`
- Integration tests: `test_integration.py`
- End-to-end tests: `tests/e2e/test_streamlit_app.py` (Playwright)
- Shared fixtures: `tests/conftest.py`

### CI/CD
- GitHub Actions workflow (`.github/workflows/main.yml`)
  - Linting (flake8, black, isort)
  - Type checking (mypy)
  - Unit and integration tests
  - E2E tests with Playwright
  - Security scanning (bandit, safety)
- Dedicated type check workflow (`.github/workflows/typecheck.yml`)

### Configuration Files
- `pyproject.toml` - Black, isort, pytest configuration
- `mypy.ini` - Strict type checking configuration
- `pytest.ini` - Test markers and settings
- `.flake8` - Linting configuration
- `.env.example` - Environment variable template

### Changed
- Implemented SOLID principles throughout codebase
- Added dependency injection via `ModelLoaderInterface`
- Updated `README.md` with v2.0 architecture documentation

---

## [1.1.0] - 2026-01-03

### Added
- **Streamlit Dashboard** (`app.py`) - Interactive web interface
  - Thermal Risk Monitor with real-time telemetry visualization
  - Workload Optimizer for scheduling decisions
  - Trade-off Analysis for water/carbon efficiency
- Trained model artifacts: `conformal_model.pkl`, `region_map.pkl`

### Changed
- Updated `forecast_engine.py` to MAPIE 1.2.0 API (`SplitConformalRegressor`)
- Pinned dependencies in `requirements.txt` for reproducibility
- Enhanced `README.md` with professional documentation

---

## [1.0.0] - 2026-01-03

### Added
- `forecast_engine.py` - Conformal prediction engine for thermal risk assessment
  - Feature engineering with 24-hour ahead temperature forecasting
  - Region encoding for multi-datacenter support
  - Hour-based features to capture daily heat cycles
  - Split conformal prediction with Train/Calibration/Test splits
  - HistGradientBoostingRegressor as base model
  - 95% confidence interval predictions using MAPIE's SplitConformalRegressor
  - Coverage score evaluation to validate prediction reliability
  - Risk-based workload scheduling decisions (block if upper bound > 35°C)
  - Model persistence with joblib and pickle

### Technical Details
- Uses MAPIE 1.2.0 API with `SplitConformalRegressor`
- Requires scikit-learn 1.5.x for compatibility
- Time-series aware splitting (`shuffle=False`) to prevent data leakage
- Outputs: `conformal_model.pkl`, `region_map.pkl`

### Dependencies
- pandas
- numpy
- scikit-learn (>=1.4, <1.6)
- mapie
- joblib
