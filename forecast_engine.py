import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from mapie.regression import MapieRegressor
from mapie.metrics import coverage_score
import joblib
import pickle

# CONFIGURATION
# ---------------------------------------------------------
INPUT_FILE = "telemetry_data.csv"
MODEL_FILE = "conformal_model.pkl"

def train_and_calibrate():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. FEATURE ENGINEERING
    # -----------------------------------------------------
    # Target: Temp 24h in the future (The "Day Ahead" view)
    # We shift the temperature column UP by 24 rows relative to the timestamp
    df['target_temp'] = df.groupby('region')['temperature_c'].shift(-24)
    
    # Encode Regions
    region_map = {region: i for i, region in enumerate(df['region'].unique())}
    df['region_encoded'] = df['region'].map(region_map)
    
    # Features: Hour is crucial for heat cycles
    df['hour'] = df['timestamp'].dt.hour
    features = ['region_encoded', 'hour', 'temperature_c']
    
    # Drop NaNs (The last 24h of data cannot be used for training as they have no future target)
    clean_df = df.dropna().copy()
    X = clean_df[features]
    y = clean_df['target_temp']
    
    # 2. SPLIT CONFORMAL (Crucial Step)
    # -----------------------------------------------------
    # We need THREE sets: Train, Calibration, Test.
    # Split 1: 60% Train, 40% Remaining
    # shuffle=False is CRITICAL for Time Series to prevent data leakage
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, shuffle=False)
    
    # Split 2: Divide Remaining into 50% Calib, 50% Test (so 20% each of total)
    X_calib, X_test, y_calib, y_test = train_test_split(X_rest, y_rest, test_size=0.5, shuffle=False)
    
    print(f"Data Splits - Train: {len(X_train)}, Calib: {len(X_calib)}, Test: {len(X_test)}")

    # 3. BASE MODEL TRAINING
    # -----------------------------------------------------
    print("Training Base Regressor...")
    # Using HistGradientBoosting (efficient, handles non-linearities like daily heat curves)
    base_model = HistGradientBoostingRegressor(random_state=42)
    base_model.fit(X_train, y_train)
    
    # 4. CONFORMAL CALIBRATION (The "Risk Layer")
    # -----------------------------------------------------
    print("Calibrating Conformal Intervals...")
    # MapieRegressor wraps the base model to add "Safety Bounds"
    # cv="prefit" means "we already fit the base model, just calibrate using X_calib"
    conformal_model = MapieRegressor(base_model, cv="prefit")
    conformal_model.fit(X_calib, y_calib)
    
    # 5. EVALUATION
    # -----------------------------------------------------
    # Predict with 95% Confidence (alpha=0.05)
    print("Evaluating on Test Set...")
    y_pred, y_pis = conformal_model.predict(X_test, alpha=0.05)
    
    # y_pis returns shape (n_samples, 2, 1) -> Lower Bound is [:, 0, 0], Upper is [:, 1, 0]
    lower_bounds = y_pis[:, 0, 0]
    upper_bounds = y_pis[:, 1, 0]
    
    # Calculate Coverage: How often did the TRUE temp fall INSIDE our bounds?
    # Target is ~95%. If it's 80%, our model is overconfident (dangerous).
    coverage = coverage_score(y_test, lower_bounds, upper_bounds)
    print(f"\nRESULTS:")
    print(f"Target Confidence: 95%")
    print(f"Actual Coverage:   {coverage:.1%} (Should be close to 95%)")
    
    # 6. SAVE ARTIFACTS
    # -----------------------------------------------------
    joblib.dump(conformal_model, MODEL_FILE)
    with open('region_map.pkl', 'wb') as f:
        pickle.dump(region_map, f)
    print(f"Model saved to {MODEL_FILE}")

    # 7. DEMO PREDICTION
    # -----------------------------------------------------
    # Demonstrate a "Risk Flag" scenario
    # Let's pretend we are in Arizona (Region 0) at 2 PM (Hour 14) with current temp 38C
    demo_features = pd.DataFrame({'region_encoded': [0], 'hour': [14], 'temperature_c': [38]})
    pred, pis = conformal_model.predict(demo_features, alpha=0.05)
    
    print("\nDEMO: Scheduling Decision")
    print(f"Input: Arizona, 2 PM, Current Temp 38C")
    print(f"Forecast: {pred[0]:.2f}C")
    print(f"Interval: [{pis[0,0,0]:.2f}C, {pis[0,1,0]:.2f}C]")
    
    if pis[0,1,0] > 35.0: # If Upper Bound > 35C [cite: 42]
        print("DECISION: BLOCK WORKLOAD (Risk of Thermal Runaway)")
    else:
        print("DECISION: APPROVE WORKLOAD")

if __name__ == "__main__":
    train_and_calibrate()