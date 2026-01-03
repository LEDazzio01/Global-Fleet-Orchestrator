# Changelog

All notable changes to this project will be documented in this file.

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
