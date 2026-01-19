"""
Global Fleet Orchestrator - Streamlit Application

A Microsoft Senior Software Engineer level implementation of the
thermal-aware workload scheduler UI.

This UI layer is intentionally "dumb" - it only:
1. Accepts user input
2. Passes it to the core engine
3. Renders the output

All business logic lives in the src/core/ package.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple

# Core imports - business logic lives here
from src.config import settings
from src.core.scheduler import Scheduler, Decision
from src.core.optimizer import WorkloadOptimizer
from src.core.risk_engine import RiskEngine, RiskLevel
from src.inference.model_loader import ModelLoader
from src.logging_config import get_logger, event_logger

logger = get_logger(__name__)


# =============================================================================
# APPLICATION SETUP
# =============================================================================

st.set_page_config(
    page_title=f"{settings.APP_NAME} | Project Forge",
    page_icon="⚡",
    layout="wide"
)


@st.cache_resource
def load_core_components() -> Tuple[Scheduler, WorkloadOptimizer, pd.DataFrame]:
    """
    Load and cache all core components.
    
    Uses dependency injection pattern - components are assembled here
    and passed to the UI rendering functions.
    
    Returns:
        Tuple of (Scheduler, WorkloadOptimizer, DataFrame).
    """
    logger.info("Loading core components")
    event_logger.model_loaded("startup", "initializing", None)
    
    # Load model using the ModelLoader interface
    loader = ModelLoader()
    model = loader.load_model()
    region_map = loader.load_region_map()
    
    event_logger.model_loaded(
        model_type="conformal",
        path=settings.MODEL_PATH,
        load_time_ms=None,
    )
    
    # Create core components with dependency injection
    risk_engine = RiskEngine(
        model=model,
        region_map=region_map,
        thermal_limit=settings.THERMAL_LIMIT_C,
    )
    
    scheduler = Scheduler(risk_engine=risk_engine)
    optimizer = WorkloadOptimizer()
    
    # Load telemetry data
    data = pd.read_csv(settings.TELEMETRY_DATA_PATH)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    
    logger.info(
        "Core components loaded",
        extra={"regions": list(region_map.keys()), "data_rows": len(data)},
    )
    
    return scheduler, optimizer, data


# Load components
scheduler, optimizer, df = load_core_components()


# =============================================================================
# HEADER
# =============================================================================

st.title(f"⚡ {settings.APP_NAME}")
st.markdown("""
**Objective:** Optimize multi-region workload placement based on 
**Carbon (Sustainability)**, **Water (Resource Efficiency)**, and 
**Thermal Risk (Reliability)**.
""")
st.markdown("---")


# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================

st.sidebar.header("🕹️ Scheduler Controls")
selected_date = st.sidebar.date_input(
    "Select Simulation Date", 
    pd.to_datetime("2026-01-15")
)
workload_shift = st.sidebar.slider(
    "Shift AZ Workload to Wyoming (%)", 
    0, 100, 0
)

# Filter data for selected date
day_data = df[df["timestamp"].dt.date == selected_date].copy()
day_data["hour"] = day_data["timestamp"].dt.hour


# =============================================================================
# VIEW A: THERMAL RISK MONITOR
# =============================================================================

st.header("1. Day-Ahead Thermal Risk Monitor (Conformal Prediction)")
st.caption(
    "Forecasting 24h ahead. Shaded area represents the "
    f"{settings.CONFIDENCE_LEVEL:.0%} Confidence Interval (The 'Safety Net')."
)

col1, col2 = st.columns([3, 1])

# Get scheduling decision from core
decision = scheduler.evaluate_workload(
    data=day_data,
    region="Arizona",
    workload_shift_pct=float(workload_shift),
)

with col1:
    # Prepare prediction data for visualization
    az_data = day_data[day_data["region"] == "Arizona"].copy()
    
    if not az_data.empty:
        # Get predictions from risk engine
        baseline_result = scheduler._risk_engine.predict(day_data, "Arizona", 0.0)
        
        if workload_shift > 0:
            temp_adjustment = workload_shift * settings.COOLING_FACTOR
            optimized_result = scheduler._risk_engine.predict(
                day_data, "Arizona", temp_adjustment
            )
        else:
            optimized_result = baseline_result
        
        # Build visualization
        fig_risk = go.Figure()
        
        # 1. Baseline Forecast
        fig_risk.add_trace(go.Scatter(
            x=baseline_result.timestamps,
            y=baseline_result.predictions,
            mode="lines",
            name="Baseline Forecast",
            line=dict(
                color="blue" if workload_shift == 0 else "rgba(100,100,255,0.4)",
                width=2
            ),
        ))
        
        # 2. Confidence Interval
        fig_risk.add_trace(go.Scatter(
            x=baseline_result.timestamps,
            y=baseline_result.upper_bounds,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        ))
        fig_risk.add_trace(go.Scatter(
            x=baseline_result.timestamps,
            y=baseline_result.lower_bounds,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0, 0, 255, 0.15)" if workload_shift > 0 else "rgba(0, 0, 255, 0.2)",
            name=f"Baseline {settings.CONFIDENCE_LEVEL:.0%} CI",
        ))
        
        # 3. Optimized Forecast (if shift applied)
        if workload_shift > 0:
            fig_risk.add_trace(go.Scatter(
                x=optimized_result.timestamps,
                y=optimized_result.predictions,
                mode="lines",
                name=f"Optimized Forecast (-{workload_shift}% load)",
                line=dict(color="#00D26A", width=3),
            ))
            fig_risk.add_trace(go.Scatter(
                x=optimized_result.timestamps,
                y=optimized_result.upper_bounds,
                mode="lines",
                name="Optimized Upper Bound",
                line=dict(color="#00D26A", width=1, dash="dot"),
            ))
        
        # 4. Safety Threshold
        fig_risk.add_trace(go.Scatter(
            x=baseline_result.timestamps,
            y=[settings.THERMAL_LIMIT_C] * len(baseline_result.timestamps),
            mode="lines",
            name=f"Thermal Breaker ({settings.THERMAL_LIMIT_C}°C)",
            line=dict(color="red", dash="dash"),
        ))
        
        fig_risk.update_layout(
            yaxis_title="Temperature (°C)",
            xaxis_title="Time",
        )
        
        st.plotly_chart(fig_risk, use_container_width=True)

with col2:
    # Display scheduling decision from core engine
    st.subheader("Scheduler Decision")
    
    if workload_shift > 0:
        st.caption("**Baseline → Optimized**")
    
    if decision.is_blocked:
        st.error("🛑 **BLOCKED**")
        if decision.baseline_risk:
            st.write(f"Baseline Risk: {decision.baseline_risk.max_upper_bound:.1f}°C")
            st.write(f"Optimized Risk: {decision.risk_assessment.max_upper_bound:.1f}°C")
            if decision.risk_reduction and decision.risk_reduction > 0:
                st.write(f"💡 Reduced by {decision.risk_reduction:.1f}°C")
        else:
            st.write(f"Risk Breach: {decision.risk_assessment.max_upper_bound:.1f}°C")
        st.write("Reason: Upper bound exceeds safety limit.")
        
        # Log the breach
        event_logger.risk_breach_detected(
            region="Arizona",
            temperature=decision.risk_assessment.max_upper_bound,
            threshold=settings.THERMAL_LIMIT_C,
        )
    else:
        st.success("✅ **APPROVED**")
        if decision.resolved_breach:
            st.balloons()
            st.write("🎉 Workload shift RESOLVED the risk!")
            st.write(
                f"Baseline: {decision.baseline_risk.max_upper_bound:.1f}°C → "
                f"{decision.risk_assessment.max_upper_bound:.1f}°C"
            )
        else:
            st.write(f"Max Risk: {decision.risk_assessment.max_upper_bound:.1f}°C")
        st.write("Reason: Within safety envelope.")
    
    # Log the decision
    event_logger.scheduling_decision(
        region="Arizona",
        decision=decision.decision.value,
        reason=decision.reason,
        workload_shift_pct=workload_shift,
    )


# =============================================================================
# VIEW B: RESOURCE EFFICIENCY
# =============================================================================

st.markdown("---")
st.header("2. Global Resource Efficiency (Carbon vs. Water)")

# Calculate optimization metrics using core optimizer
if workload_shift > 0:
    opt_result = optimizer.calculate_optimization(
        data=day_data,
        source_region="Arizona",
        target_region="Wyoming",
        workload_shift_pct=float(workload_shift),
    )
    total_water_optimized = opt_result.optimized_water_l
else:
    total_water = day_data["water_usage_l"].sum()
    total_water_optimized = total_water

# Display metrics
current_az_load = 100 - workload_shift
total_water = day_data["water_usage_l"].sum()

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric(
    "Global Water Usage (L)",
    f"{total_water:,.0f}",
    delta=f"-{(total_water - total_water_optimized):,.0f} L Saved",
)
col_m2.metric(
    "Workload in Arizona",
    f"{current_az_load}%",
    delta=f"-{workload_shift}%",
)
col_m3.metric(
    "Workload in Wyoming",
    f"{100 + workload_shift}%",
    delta=f"+{workload_shift}%",
)

# Scatter Plot
st.subheader("Live Trade-off Analysis")

# Apply shift to visualization data
if workload_shift > 0:
    plot_data = optimizer.apply_shift_to_data(
        data=day_data,
        source_region="Arizona",
        target_region="Wyoming",
        workload_shift_pct=float(workload_shift),
    )
else:
    plot_data = day_data.copy()

# Handle log scale zeros
plot_data.loc[plot_data["water_usage_l"] == 0, "water_usage_l"] = 0.05
plot_data.loc[plot_data["water_usage_l"] < 1, "water_usage_l"] += 0.1

# Normalize bubble sizes
min_load = plot_data["it_load_mw"].min()
max_load = plot_data["it_load_mw"].max()
plot_data["bubble_size"] = (
    (plot_data["it_load_mw"] - min_load) / (max_load - min_load) * 80
) + 20

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
        "hour": "Hour",
    },
    color_discrete_map={
        "Arizona": "#FF6B6B",
        "Ireland": "#4ECDC4",
        "Wyoming": "#FFE66D",
    },
)
fig_scatter.update_yaxes(type="log", range=[-1.5, 3.5])
fig_scatter.update_xaxes(range=[50, 500])
fig_scatter.update_traces(
    marker=dict(line=dict(width=2, color="white"), opacity=0.85, sizemin=8)
)
fig_scatter.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_scatter, use_container_width=True)


# =============================================================================
# VIEW C: OPTIMIZATION SUMMARY
# =============================================================================

if workload_shift > 0:
    st.markdown("---")
    st.header("3. Optimization Impact Summary")
    
    opt_result = optimizer.calculate_optimization(
        data=day_data,
        source_region="Arizona",
        target_region="Wyoming",
        workload_shift_pct=float(workload_shift),
    )
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    col_s1.metric(
        "💧 Water Saved",
        f"{opt_result.water_saved_l:,.0f} L",
        delta=f"-{opt_result.water_saved_pct:.1f}%",
        delta_color="inverse",
    )
    
    risk_reduction = 0.0
    if decision.baseline_risk:
        risk_reduction = (
            decision.baseline_risk.max_upper_bound - 
            decision.risk_assessment.max_upper_bound
        )
    
    col_s2.metric(
        "🌡️ Thermal Risk Reduction",
        f"{risk_reduction:.1f}°C",
        delta="Safer" if risk_reduction > 0 else "Same",
    )
    
    col_s3.metric(
        "🔄 Workload Shifted",
        f"{workload_shift}%",
        delta="AZ→WY",
    )
    
    carbon_direction = "increase" if opt_result.carbon_delta_kg > 0 else "decrease"
    col_s4.metric(
        "🌿 Carbon Impact",
        f"{abs(opt_result.carbon_delta_kg):.1f} kg CO2",
        delta=carbon_direction,
        delta_color="inverse" if opt_result.carbon_delta_kg > 0 else "normal",
    )
    
    # Summary message
    if decision.resolved_breach:
        st.success(
            f"✅ **Shifting {workload_shift}% of Arizona workload to Wyoming "
            f"resolves the thermal breach!** Water usage reduced by "
            f"{opt_result.water_saved_pct:.1f}% while maintaining reliability."
        )
    elif decision.is_approved:
        st.info(
            f"ℹ️ System already within safe limits. "
            f"Shifting workload saves {opt_result.water_saved_l:,.0f}L of water."
        )
    else:
        min_shift = scheduler.find_minimum_shift(day_data, "Arizona")
        if min_shift:
            st.warning(
                f"⚠️ Need to shift approximately **{min_shift:.0f}%** to clear thermal risk."
            )
        else:
            st.error("❌ Unable to resolve thermal risk with workload shift alone.")


# =============================================================================
# FOOTER
# =============================================================================

st.caption(
    f"Built for Project Forge Simulation | {settings.APP_NAME} v{settings.APP_VERSION} | "
    f"Conformal Prediction Engine"
)
