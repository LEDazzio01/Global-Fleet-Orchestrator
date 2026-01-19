# Global Fleet Orchestrator

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://github.com/LEDazzio01/Global-Fleet-Orchestrator/actions/workflows/main.yml/badge.svg)](https://github.com/LEDazzio01/Global-Fleet-Orchestrator/actions)
[![Type Check](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy-lang.org/)

## Abstract

**Global Fleet Orchestrator** is a production-grade decision-support system for optimizing multi-region datacenter workload placement across three critical dimensions: **Carbon Sustainability**, **Water Efficiency**, and **Thermal Reliability**. 

The system employs **Conformal Prediction**—a distribution-free uncertainty quantification framework—to generate statistically valid prediction intervals for day-ahead thermal forecasting. Unlike traditional point forecasts, conformal prediction provides guaranteed coverage rates (e.g., 95% confidence intervals), enabling risk-aware scheduling decisions that account for worst-case thermal scenarios.

### Key Capabilities

- **Day-Ahead Thermal Risk Monitoring**: Forecasts datacenter temperatures 24 hours ahead with calibrated uncertainty bounds using MAPIE's Split Conformal Regression
- **What-If Workload Optimization**: Interactive simulation of workload migration between regions (Arizona → Wyoming) with real-time impact visualization
- **Multi-Objective Trade-off Analysis**: Animated scatter plots showing Carbon Intensity vs. Water Usage across regions and time
- **Risk-Based Scheduling**: Automated APPROVE/BLOCK decisions based on upper confidence bounds exceeding thermal safety thresholds (35°C)

### Engineering Highlights (v2.0)

- **SOLID Architecture**: Service layer pattern with dependency injection for testability
- **Pydantic Configuration**: All magic numbers externalized and environment-configurable
- **ONNX Runtime Support**: Production-grade model inference with cross-platform compatibility
- **Structured Logging**: JSON-formatted logs for observability (Azure Monitor compatible)
- **Comprehensive Testing**: Unit tests (pytest), integration tests, and E2E tests (Playwright)
- **CI/CD Pipeline**: GitHub Actions with linting, type checking, and security scanning

---

## Project Structure

```
Global-Fleet-Orchestrator/
├── src/                        # Core business logic (SOLID principles)
│   ├── config.py               # Pydantic BaseSettings (env-configurable)
│   ├── logging_config.py       # JSON structured logging
│   ├── core/                   # Business logic layer
│   │   ├── scheduler.py        # Workload scheduling decisions
│   │   ├── risk_engine.py      # Thermal risk assessment
│   │   └── optimizer.py        # Resource efficiency calculations
│   ├── models/                 # Data models with validation
│   │   ├── telemetry.py        # Pydantic-validated telemetry records
│   │   └── prediction.py       # Prediction result models
│   └── inference/              # Model loading & inference
│       ├── model_loader.py     # Dependency injection interface
│       ├── onnx_runtime.py     # ONNX Runtime loader (production)
│       └── convert_to_onnx.py  # Model export utility
├── tests/                      # Comprehensive test suite
│   ├── conftest.py             # Shared fixtures
│   ├── test_scheduler.py       # Scheduler unit tests
│   ├── test_risk_engine.py     # Risk engine unit tests
│   ├── test_optimizer.py       # Optimizer unit tests
│   ├── test_integration.py     # Pipeline integration tests
│   └── e2e/                    # Playwright E2E tests
│       └── test_streamlit_app.py
├── .github/workflows/          # CI/CD pipelines
│   ├── main.yml                # Full CI: lint, test, security
│   └── typecheck.yml           # MyPy type checking
├── app.py                      # Original Streamlit dashboard
├── app_refactored.py           # Refactored dashboard (uses src/)
├── forecast_engine.py          # Conformal prediction model training
├── generate_data.py            # Synthetic telemetry data generator
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Tool configuration (black, isort, pytest)
├── mypy.ini                    # Type checking configuration
└── .env.example                # Environment variable template
```

---

## Quick Start

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/LEDazzio01/Global-Fleet-Orchestrator.git
cd Global-Fleet-Orchestrator

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate sample data
python generate_data.py

# Train the conformal prediction model
python forecast_engine.py
```

### Run the Application

```bash
# Run the refactored application (recommended)
streamlit run app_refactored.py

# Or run the original application
streamlit run app.py
```

The application launches at `http://localhost:8501`

---

## Configuration

All configuration is managed via Pydantic BaseSettings and can be overridden with environment variables:

```bash
# Copy the example environment file
cp .env.example .env

# Or set environment variables directly
export GFO_THERMAL_LIMIT_C=35.0      # Safety threshold (°C)
export GFO_COOLING_FACTOR=0.05       # Temp reduction per % workload shift
export GFO_CONFIDENCE_LEVEL=0.95     # Prediction interval confidence
export GFO_LOG_LEVEL=INFO            # DEBUG, INFO, WARNING, ERROR
export GFO_LOG_FORMAT=json           # json or text
```

See [src/config.py](src/config.py) for all available settings.

---

## Testing

### Run All Tests

```bash
# Unit and integration tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_scheduler.py -v
```

### Test Categories

| Test File | Description | Count |
|-----------|-------------|-------|
| `test_scheduler.py` | Scheduler decision logic | 7 tests |
| `test_risk_engine.py` | Risk assessment logic | 10 tests |
| `test_optimizer.py` | Optimization calculations | 9 tests |
| `test_integration.py` | Data pipeline integration | 11 tests |
| `e2e/test_streamlit_app.py` | Playwright E2E tests | 8 tests |

### E2E Tests (Playwright)

```bash
# Install Playwright browsers
playwright install chromium

# Run E2E tests
pytest tests/e2e/ -v

# Run with visible browser
pytest tests/e2e/ --headed
```

---

## ONNX Export (Production)

For production deployment, export the model to ONNX format:

```bash
# Convert pickle model to ONNX
python -m src.inference.convert_to_onnx --validate

# Files created:
# - models/thermal_forecast.onnx
# - models/thermal_forecast_calibration.npy
```

Benefits of ONNX:
- Platform-independent inference (Windows, Linux, macOS)
- Optimized performance via ONNX Runtime
- No pickle security concerns
- Smaller deployment footprint

---

## Using the Dashboard

1. **Select Simulation Date**: Choose a date from the telemetry dataset to analyze
2. **Adjust Workload Shift Slider**: Simulate migrating 0-100% of Arizona workload to Wyoming
3. **Observe Impact**:
   - Thermal forecast shows baseline vs. optimized predictions
   - Scatter plot bubbles resize to reflect workload distribution
   - Metrics update with water savings and carbon impact
   - Scheduler decision changes from BLOCKED → APPROVED when risk is mitigated

---

## How It Works

### Conformal Prediction Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Training Set   │────▶│  Base Model      │────▶│  Point Predictions │
│  (60% of data)  │     │  (HistGradient)  │     │                    │
└─────────────────┘     └──────────────────┘     └────────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│ Calibration Set │────▶│  Conformalize    │────▶│  Residual Scores   │
│  (20% of data)  │     │  (Compute Errors)│     │  (Nonconformity)   │
└─────────────────┘     └──────────────────┘     └────────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│   Test/New Data │────▶│  Predict + Width │────▶│  [Lower, Upper]    │
│                 │     │  Based on Quantile│    │  Prediction Bounds │
└─────────────────┘     └──────────────────┘     └────────────────────┘
```

### Decision Logic

```python
if upper_confidence_bound > 35°C:
    decision = "BLOCKED"  # Risk of thermal runaway
else:
    decision = "APPROVED"  # Safe to schedule workload
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.0.0 | Data manipulation |
| numpy | ≥1.24.0 | Numerical operations |
| scikit-learn | ≥1.4.0, <1.6.0 | Base ML model (HistGradientBoostingRegressor) |
| mapie | ≥1.0.0 | Conformal prediction (SplitConformalRegressor) |
| streamlit | ≥1.30.0 | Interactive dashboard |
| plotly | ≥5.18.0 | Animated visualizations |
| pydantic | ≥2.0.0 | Configuration & data validation |
| pydantic-settings | ≥2.0.0 | Environment-based configuration |
| onnxruntime | ≥1.16.0 | Production model inference |
| skl2onnx | ≥1.16.0 | Model export to ONNX |
| pytest | ≥7.4.0 | Testing framework |
| playwright | ≥1.40.0 | E2E browser testing |
| mypy | ≥1.8.0 | Static type checking |
| flake8 | ≥6.1.0 | Code linting |
| black | ≥23.12.0 | Code formatting |

> ⚠️ **Note**: scikit-learn is pinned to <1.6.0 for compatibility with MAPIE 1.2.0

---

## CI/CD Pipeline

The project includes a comprehensive GitHub Actions workflow:

```yaml
# .github/workflows/main.yml
Jobs:
  ├── lint          # flake8, isort, black
  ├── type-check    # mypy static analysis
  ├── test          # pytest unit & integration tests
  ├── e2e-test      # Playwright browser tests (PRs only)
  ├── security-scan # bandit, safety
  └── build-check   # Verify imports and app startup
```

All checks run on every push and pull request to `main`.

---

## Architecture

### Service Layer Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                    │
│                    "Dumb" presentation layer                │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                     Scheduler                                │
│              Makes APPROVE/BLOCK decisions                   │
└─────────────────────────────┬───────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼───────┐    ┌────────▼────────┐   ┌───────▼───────┐
│  RiskEngine   │    │   Optimizer     │   │  ModelLoader  │
│ Risk assessment│   │ Water/Carbon    │   │ Dependency    │
│ with conformal │   │ efficiency      │   │ injection     │
└───────────────┘    └─────────────────┘   └───────────────┘
```

### Conformal Prediction Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Training Set   │────▶│  Base Model      │────▶│  Point Predictions │
│  (60% of data)  │     │  (HistGradient)  │     │                    │
└─────────────────┘     └──────────────────┘     └────────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│ Calibration Set │────▶│  Conformalize    │────▶│  Residual Scores   │
│  (20% of data)  │     │  (Compute Errors)│     │  (Nonconformity)   │
└─────────────────┘     └──────────────────┘     └────────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│   Test/New Data │────▶│  Predict + Width │────▶│  [Lower, Upper]    │
│                 │     │  Based on Quantile│    │  Prediction Bounds │
└─────────────────┘     └──────────────────┘     └────────────────────┘
```

### Decision Logic

```python
if upper_confidence_bound > settings.THERMAL_LIMIT_C:
    decision = "BLOCKED"  # Risk of thermal runaway
else:
    decision = "APPROVED"  # Safe to schedule workload
```

---

## Structured Logging

The application outputs JSON-structured logs for observability:

```json
{
  "timestamp": "2026-01-19T20:34:10.112766+00:00",
  "level": "INFO",
  "logger": "src.core.risk_engine",
  "message": "Risk assessment completed",
  "app": {"name": "Global Fleet Orchestrator", "version": "2.0.0"},
  "context": {
    "region": "Arizona",
    "risk_level": "CRITICAL",
    "max_upper_bound": 46.94,
    "threshold": 35.0,
    "exceeds_threshold": true
  }
}
```

Switch to human-readable format for development:
```bash
export GFO_LOG_FORMAT=text
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) - Model Agnostic Prediction Interval Estimator
- [Streamlit](https://streamlit.io/) - The fastest way to build data apps
- [ONNX Runtime](https://onnxruntime.ai/) - Cross-platform ML inference
- [Playwright](https://playwright.dev/) - Reliable E2E testing
- [Pydantic](https://pydantic.dev/) - Data validation using Python type hints
- Built for Project Forge Simulation