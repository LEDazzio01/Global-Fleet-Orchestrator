import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import pickle

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Global Fleet Orchestrator | Project Forge",
    page_icon="⚡",
    layout="wide"
)

# LOAD ARTIFACTS
@st.cache_resource
def load_artifacts():
    model = joblib.load("conformal_model.pkl")
    data = pd.read_csv("telemetry_data.csv")
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    with open('region_map.pkl', 'rb') as f:
        region_map = pickle.load(f)
    return model, data, region_map

model, df, region_map = load_artifacts()

# HEADER
st.title("⚡ Global Fleet Orchestrator")
st.markdown("""
**Objective:** Optimize multi-region workload placement based on **Carbon (Sustainability)**, **Water (Resource Efficiency)**, and **Thermal Risk (Reliability)**.
""")
st.markdown("---")

# SIDEBAR: CONTROLS
st.sidebar.header("🕹️ Scheduler Controls")
selected_date = st.sidebar.date_input("Select Simulation Date", pd.to_datetime("2026-01-15"))
workload_shift = st.sidebar.slider("Shift AZ Workload to Wyoming (%)", 0, 100, 0)

# Filter Data for Selected Date
day_data = df[df['timestamp'].dt.date == selected_date].copy()
day_data['hour'] = day_data['timestamp'].dt.hour

# --- VIEW A: THE RISK MONITOR (Conformal Prediction) ---
st.header("1. Day-Ahead Thermal Risk Monitor (Conformal Prediction)")
st.caption("Forecasting 24h ahead. Shaded area represents the 95% Confidence Interval (The 'Safety Net').")

col1, col2 = st.columns([3, 1])

with col1:
    # Prepare Prediction Data
    # We take the selected day's data as "Current Input" to predict "Tomorrow"
    # Note: In real life, we'd fetch live data. Here we simulate using historicals.
    input_data = day_data.copy()
    input_data['region_encoded'] = input_data['region'].map(region_map)
    input_data['hour'] = input_data['timestamp'].dt.hour
    
    # Predict
    # We only predict for Arizona (High Risk Region) for the chart
    az_data = input_data[input_data['region'] == "Arizona"].copy()
    
    if not az_data.empty:
        # BASELINE: Original Arizona temperature
        features = az_data[['region_encoded', 'hour', 'temperature_c']]
        pred, pis = model.predict_interval(features)
        lower_bound = pis[:, 0, 0]
        upper_bound = pis[:, 1, 0]
        
        # OPTIMIZED: Simulate reduced load effect on temperature
        # Shifting workload reduces cooling demand, lowering temps by ~0.1°C per 10% shift
        temp_reduction = workload_shift * 0.05  # Up to 5°C reduction at 100% shift
        az_data_optimized = az_data.copy()
        az_data_optimized['temperature_c'] = az_data['temperature_c'] - temp_reduction
        features_opt = az_data_optimized[['region_encoded', 'hour', 'temperature_c']]
        pred_opt, pis_opt = model.predict_interval(features_opt)
        upper_bound_opt = pis_opt[:, 1, 0]
        
        # Plotting
        fig_risk = go.Figure()
        
        # 1. Original Forecast Line (dimmed if shifted)
        fig_risk.add_trace(go.Scatter(
            x=az_data['timestamp'], y=pred,
            mode='lines', name='Baseline Forecast',
            line=dict(color='blue' if workload_shift == 0 else 'rgba(100,100,255,0.4)', width=2)
        ))
        
        # 2. Original Confidence Interval
        fig_risk.add_trace(go.Scatter(
            x=az_data['timestamp'], y=upper_bound,
            mode='lines', line=dict(width=0), showlegend=False
        ))
        fig_risk.add_trace(go.Scatter(
            x=az_data['timestamp'], y=lower_bound,
            mode='lines', line=dict(width=0), fill='tonexty',
            fillcolor='rgba(0, 0, 255, 0.15)' if workload_shift > 0 else 'rgba(0, 0, 255, 0.2)', 
            name='Baseline 95% CI'
        ))
        
        # 3. OPTIMIZED Forecast (only show if slider moved)
        if workload_shift > 0:
            fig_risk.add_trace(go.Scatter(
                x=az_data['timestamp'], y=pred_opt,
                mode='lines', name=f'Optimized Forecast (-{workload_shift}% load)',
                line=dict(color='#00D26A', width=3)
            ))
            # Optimized upper bound
            fig_risk.add_trace(go.Scatter(
                x=az_data['timestamp'], y=upper_bound_opt,
                mode='lines', name='Optimized Upper Bound',
                line=dict(color='#00D26A', width=1, dash='dot')
            ))
        
        # 4. Safety Threshold
        fig_risk.add_trace(go.Scatter(
            x=az_data['timestamp'], y=[35]*len(az_data),
            mode='lines', name='Thermal Breaker (35°C)',
            line=dict(color='red', dash='dash')
        ))
        
        fig_risk.update_layout(
            yaxis_title="Temperature (°C)",
            xaxis_title="Time"
        )
        
        st.plotly_chart(fig_risk, width='stretch')

with col2:
    # Decision Logic - compare baseline vs optimized
    st.subheader("Scheduler Decision")
    max_risk_baseline = np.max(upper_bound) if not az_data.empty else 0
    max_risk_optimized = np.max(upper_bound_opt) if workload_shift > 0 else max_risk_baseline
    
    if workload_shift > 0:
        st.caption("**Baseline → Optimized**")
        
    if max_risk_optimized > 35:
        st.error(f"🛑 **BLOCKED**")
        if workload_shift > 0:
            st.write(f"Baseline Risk: {max_risk_baseline:.1f}°C")
            st.write(f"Optimized Risk: {max_risk_optimized:.1f}°C")
            if max_risk_baseline > max_risk_optimized:
                st.write(f"💡 Reduced by {max_risk_baseline - max_risk_optimized:.1f}°C")
        else:
            st.write(f"Risk Breach: {max_risk_baseline:.1f}°C")
        st.write("Reason: Upper bound exceeds safety limit.")
    else:
        st.success(f"✅ **APPROVED**")
        if workload_shift > 0 and max_risk_baseline > 35:
            st.balloons()
            st.write(f"🎉 Workload shift RESOLVED the risk!")
            st.write(f"Baseline: {max_risk_baseline:.1f}°C → {max_risk_optimized:.1f}°C")
        else:
            st.write(f"Max Risk: {max_risk_optimized:.1f}°C")
        st.write("Reason: Within safety envelope.")

# --- VIEW B & C: TRADE-OFFS & OPTIMIZATION ---
st.markdown("---")
st.header("2. Global Resource Efficiency (Carbon vs. Water)")

# Calculate "What-If" Metrics based on Slider
# Logic: If we shift load from AZ to WY, AZ water drops, WY Carbon might rise (grid constraints)
current_az_load = 100 - workload_shift
wy_load_adder = workload_shift

# Aggregate Metrics for the Day
total_water = day_data['water_usage_l'].sum()
# Apply reduction to AZ water portion (Simulated)
az_water_idx = day_data['region'] == "Arizona"
total_water_optimized = total_water - (day_data.loc[az_water_idx, 'water_usage_l'].sum() * (workload_shift/100))

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Global Water Usage (L)", f"{total_water:,.0f}", delta=f"-{(total_water - total_water_optimized):,.0f} L Saved")
col_m2.metric("Workload in Arizona", f"{current_az_load}%", delta=f"-{workload_shift}%")
col_m3.metric("Workload in Wyoming", f"{100 + workload_shift}%", delta=f"+{workload_shift}%")

# Scatter Plot: Carbon vs Water
st.subheader("Live Trade-off Analysis")
# Create a vibrant scatter plot
# Add small offset to zero water values so they appear on log scale
# Also add slight jitter to separate overlapping points
plot_data = day_data.copy()

# Apply workload shift simulation to the scatter plot data
if workload_shift > 0:
    # Reduce Arizona IT load and water usage proportionally
    az_mask = plot_data['region'] == "Arizona"
    wy_mask = plot_data['region'] == "Wyoming"
    
    # Calculate shifted loads - use amplified effect for visual clarity
    shift_factor = workload_shift / 100
    
    # Apply to Arizona (reduce) - more aggressive reduction for visual impact
    plot_data.loc[az_mask, 'it_load_mw'] = plot_data.loc[az_mask, 'it_load_mw'] * (1 - shift_factor * 0.8)
    plot_data.loc[az_mask, 'water_usage_l'] = plot_data.loc[az_mask, 'water_usage_l'] * (1 - shift_factor)
    
    # Apply to Wyoming (increase) - double the visual effect
    base_wy_load = day_data.loc[day_data['region'] == 'Wyoming', 'it_load_mw'].values
    plot_data.loc[wy_mask, 'it_load_mw'] = base_wy_load * (1 + shift_factor * 1.5)
    # Wyoming uses no water (air-cooled), so water stays at 0

# Handle log scale zeros
plot_data.loc[plot_data['water_usage_l'] == 0, 'water_usage_l'] = 0.05  # Wyoming at bottom
plot_data.loc[plot_data['water_usage_l'] < 1, 'water_usage_l'] = plot_data.loc[plot_data['water_usage_l'] < 1, 'water_usage_l'] + 0.1

# Normalize bubble sizes for better visual differentiation
# Scale to 0-100 range for more dramatic size differences
min_load = plot_data['it_load_mw'].min()
max_load = plot_data['it_load_mw'].max()
plot_data['bubble_size'] = ((plot_data['it_load_mw'] - min_load) / (max_load - min_load) * 80) + 20

# Add scenario label for clarity
scenario_label = f" (After {workload_shift}% Shift)" if workload_shift > 0 else ""

fig_scatter = px.scatter(
    plot_data, 
    x="carbon_intensity_gco2", 
    y="water_usage_l", 
    color="region", 
    size="bubble_size",
    hover_data=["temperature_c", "hour", "it_load_mw"],
    animation_frame="hour",
    size_max=60,
    title=f"Carbon Intensity vs. Water Usage (Hourly Animation){scenario_label}",
    labels={
        "carbon_intensity_gco2": "Carbon Intensity (gCO2/kWh)", 
        "water_usage_l": "Water Usage (L)",
        "it_load_mw": "IT Load (MW)",
        "bubble_size": "Relative Load",
        "temperature_c": "Temperature (°C)",
        "hour": "Hour"
    },
    color_discrete_map={
        "Arizona": "#FF6B6B",
        "Ireland": "#4ECDC4", 
        "Wyoming": "#FFE66D"
    }
)
# Use log scale for y-axis to show detail across the full range
fig_scatter.update_yaxes(type="log", range=[-1.5, 3.5])  # Extend down to show Wyoming
fig_scatter.update_xaxes(range=[50, 500])
fig_scatter.update_traces(marker=dict(line=dict(width=2, color='white'), opacity=0.85, sizemin=8))
fig_scatter.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_scatter, width='stretch')

# --- VIEW C: OPTIMIZATION SUMMARY ---
if workload_shift > 0:
    st.markdown("---")
    st.header("3. Optimization Impact Summary")
    
    # Calculate before/after metrics
    baseline_water = day_data['water_usage_l'].sum()
    baseline_az_load = day_data[day_data['region'] == 'Arizona']['it_load_mw'].sum()
    baseline_wy_load = day_data[day_data['region'] == 'Wyoming']['it_load_mw'].sum()
    
    optimized_water = total_water_optimized
    water_saved = baseline_water - optimized_water
    water_saved_pct = (water_saved / baseline_water) * 100 if baseline_water > 0 else 0
    
    # Carbon impact (Wyoming has lower carbon intensity on average)
    az_carbon = day_data[day_data['region'] == 'Arizona']['carbon_intensity_gco2'].mean()
    wy_carbon = day_data[day_data['region'] == 'Wyoming']['carbon_intensity_gco2'].mean()
    carbon_delta = (wy_carbon - az_carbon) * (baseline_az_load * workload_shift / 100) / 1000  # kg CO2
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    col_s1.metric(
        "💧 Water Saved", 
        f"{water_saved:,.0f} L",
        delta=f"-{water_saved_pct:.1f}%",
        delta_color="inverse"
    )
    
    col_s2.metric(
        "🌡️ Thermal Risk Reduction",
        f"{max_risk_baseline - max_risk_optimized:.1f}°C",
        delta="Safer" if max_risk_optimized < max_risk_baseline else "Same"
    )
    
    col_s3.metric(
        "🔄 Workload Shifted",
        f"{workload_shift}%",
        delta=f"AZ→WY"
    )
    
    carbon_direction = "increase" if carbon_delta > 0 else "decrease"
    col_s4.metric(
        "🌿 Carbon Impact",
        f"{abs(carbon_delta):.1f} kg CO2",
        delta=f"{carbon_direction}",
        delta_color="inverse" if carbon_delta > 0 else "normal"
    )
    
    # Summary text
    if max_risk_optimized <= 35 and max_risk_baseline > 35:
        st.success(f"✅ **Shifting {workload_shift}% of Arizona workload to Wyoming resolves the thermal breach!** "
                   f"Water usage reduced by {water_saved_pct:.1f}% while maintaining reliability.")
    elif max_risk_optimized <= 35:
        st.info(f"ℹ️ System already within safe limits. Shifting workload saves {water_saved:,.0f}L of water.")
    else:
        remaining_shift = int((max_risk_optimized - 35) / 0.05) + workload_shift
        st.warning(f"⚠️ Need to shift approximately **{min(remaining_shift, 100)}%** to clear thermal risk.")

# Footer
st.caption("Built for Project Forge Simulation | Conformal Prediction Engine v1.0")