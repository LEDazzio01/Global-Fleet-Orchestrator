import pandas as pd
import numpy as np
import os

# CONFIGURATION
# ---------------------------------------------------------
# Operational Constraints (The "Truth")
REGIONS = {
    "Arizona": {
        "pue_base": 1.15,
        "water_factor": 150,  # Liters per degree above threshold
        "carbon_base": 400,   # gCO2/kWh (Coal/Gas heavy)
        "temp_mean": 30,
        "temp_std": 8,
        "cooling_type": "evaporative" 
    },
    "Ireland": {
        "pue_base": 1.08,     # Very efficient air cooling
        "water_factor": 0,    # Air cooling uses almost no water
        "carbon_base": 250,   # Variable (Wind)
        "temp_mean": 12,
        "temp_std": 4,
        "cooling_type": "air"
    },
    "Wyoming": {
        "pue_base": 1.10,
        "water_factor": 10,   # Minimal water
        "carbon_base": 300,
        "temp_mean": 15,
        "temp_std": 10,
        "cooling_type": "hybrid"
    }
}

def generate_telemetry(days=30):
    """Generates synthetic hourly telemetry for 3 regions."""
    # Create hourly timestamps
    dates = pd.date_range(start='2026-01-01', periods=days*24, freq='H')
    data_frames = []

    for region, specs in REGIONS.items():
        n = len(dates)
        
        # 1. Simulating Temperature (Sinusoidal pattern + Noise)
        # Peak temp usually around 2 PM (Hour 14)
        daily_pattern = np.sin((dates.hour - 8) * np.pi / 12) 
        temp_noise = np.random.normal(0, 2, n)
        temperature = specs["temp_mean"] + (specs["temp_std"] * daily_pattern) + temp_noise
        
        # 2. Simulating IT Load (Business hours peak)
        # Load follows people: Low at night, High 9-5
        load_pattern = np.sin((dates.hour - 6) * np.pi / 12)
        # Shift to make it positive and add baseline MW
        it_load = 50 + (20 * load_pattern) + np.random.normal(0, 2, n)
        
        # 3. Simulating Water Usage (The "Physics" Layer)
        # WUE Spikes in Arizona when Temp > 25C (Evaporative Cooling kicks in)
        water_usage = np.zeros(n)
        
        if specs["cooling_type"] == "evaporative":
            # Physics: If Temp > 25C, we spray water. 
            # Formula: Base usage + (Exceeded Temp * Factor)
            water_usage = np.where(temperature > 25, (temperature - 25) * specs["water_factor"], 0)
            # Add some randomness
            water_usage += np.abs(np.random.normal(0, 10, n))
            
        elif specs["cooling_type"] == "air":
            # Minimal water for humidity control only
            water_usage = np.random.uniform(0, 5, n)
            
        else: # Wyoming Hybrid
            water_usage = np.where(temperature > 30, (temperature - 30) * 50, 0)

        # 4. Simulating Carbon Intensity (gCO2/kWh)
        # In Ireland, wind often blows more at night or varies randomly
        if region == "Ireland":
            carbon_intensity = specs["carbon_base"] + np.random.normal(0, 50, n)
            # Clip to ensure no negative carbon (unless we have credits, but let's keep it simple)
            carbon_intensity = np.clip(carbon_intensity, 50, 600)
        else:
            # Solar regions (AZ) have lower carbon during the day
            solar_curve = -1 * daily_pattern # Invert sun pattern
            carbon_intensity = specs["carbon_base"] + (50 * solar_curve) + np.random.normal(0, 10, n)

        # Create DataFrame
        df = pd.DataFrame({
            "timestamp": dates,
            "region": region,
            "temperature_c": temperature.round(2),
            "it_load_mw": it_load.round(2),
            "carbon_intensity_gco2": carbon_intensity.round(2),
            "water_usage_l": water_usage.round(2)
        })
        
        data_frames.append(df)

    # Combine and Save
    full_df = pd.concat(data_frames)
    full_df.to_csv("telemetry_data.csv", index=False)
    print(f"Generate complete. Saved {len(full_df)} rows to telemetry_data.csv")
    return full_df

if __name__ == "__main__":
    generate_telemetry()