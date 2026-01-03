# Global Fleet Orchestrator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

**Global Fleet Orchestrator** is a decision-support system for optimizing multi-region datacenter workload placement across three critical dimensions: **Carbon Sustainability**, **Water Efficiency**, and **Thermal Reliability**. 

The system employs **Conformal Prediction**—a distribution-free uncertainty quantification framework—to generate statistically valid prediction intervals for day-ahead thermal forecasting. Unlike traditional point forecasts, conformal prediction provides guaranteed coverage rates (e.g., 95% confidence intervals), enabling risk-aware scheduling decisions that account for worst-case thermal scenarios.

### Key Capabilities

- **Day-Ahead Thermal Risk Monitoring**: Forecasts datacenter temperatures 24 hours ahead with calibrated uncertainty bounds using MAPIE's Split Conformal Regression
- **What-If Workload Optimization**: Interactive simulation of workload migration between regions (Arizona → Wyoming) with real-time impact visualization
- **Multi-Objective Trade-off Analysis**: Animated scatter plots showing Carbon Intensity vs. Water Usage across regions and time
- **Risk-Based Scheduling**: Automated APPROVE/BLOCK decisions based on upper confidence bounds exceeding thermal safety thresholds (35°C)

### Technical Highlights

- **Conformal Prediction Engine**: HistGradientBoostingRegressor wrapped with `SplitConformalRegressor` for calibrated prediction intervals
- **Time-Series Aware Splitting**: Prevents data leakage with chronological train/calibration/test splits
- **Coverage Validation**: Empirical coverage scores validate that prediction intervals achieve target confidence levels

---

## Project Structure

```
Global-Fleet-Orchestrator/
├── app.py                  # Streamlit dashboard application
├── forecast_engine.py      # Conformal prediction model training
├── generate_data.py        # Synthetic telemetry data generator
├── telemetry_data.csv      # Sample multi-region datacenter telemetry
├── conformal_model.pkl     # Trained conformal prediction model
├── region_map.pkl          # Region encoding mapping
├── requirements.txt        # Python dependencies
├── CHANGELOG.md            # Version history
└── README.md               # This file
```

---

## Setup

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/LEDazzio01/Global-Fleet-Orchestrator.git
   cd Global-Fleet-Orchestrator
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the conformal prediction model**
   ```bash
   python forecast_engine.py
   ```
   
   Expected output:
   ```
   Loading data...
   Data Splits - Train: 1252, Calib: 418, Test: 418
   Training Base Regressor...
   Calibrating Conformal Intervals...
   Evaluating on Test Set...
   
   RESULTS:
   Target Confidence: 95%
   Actual Coverage:   ~91-95% (Should be close to 95%)
   Model saved to conformal_model.pkl
   ```

---

## Running the Application

### Start the Dashboard

```bash
streamlit run app.py
```

The application will launch at `http://localhost:8501`

### Using the Dashboard

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
| joblib | ≥1.3.0 | Model serialization |

> ⚠️ **Note**: scikit-learn is pinned to <1.6.0 for compatibility with MAPIE 1.2.0

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) - Model Agnostic Prediction Interval Estimator
- [Streamlit](https://streamlit.io/) - The fastest way to build data apps
- Built for Project Forge Simulation