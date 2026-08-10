from typing import Dict, Any, Callable, Optional
import streamlit as st

def render_scenario_view(
    payload: Dict[str, Any],
    on_run_simulation: Callable[[Dict[str, float]], None]
):
    """
    Render Interactive What-If Scenario Simulation Lab Section.
    """
    st.markdown("## 🧪 WHAT-IF SCENARIO SIMULATION LAB")
    st.caption("Simulate hypothetical microclimate shifts without modifying actual weather data.")

    # Controls Container
    with st.form("scenario_form"):
        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            delta_temp = st.slider("Temperature Shift (°C)", min_value=-10.0, max_value=10.0, value=2.0, step=0.5, help="Offset ambient temperature in °C (+/-)")
        with col_s2:
            override_rh = st.slider("Relative Humidity Override (%)", min_value=0.0, max_value=100.0, value=95.0, step=1.0, help="Set fixed canopy relative humidity %")
        with col_s3:
            delta_rain = st.slider("Rainfall Offset (mm)", min_value=-50.0, max_value=50.0, value=0.0, step=1.0, help="Offset rainfall in mm (+/-)")

        submit_sim = st.form_submit_button("🧪 Run What-If Simulation", type="primary", use_container_width=True)

    if submit_sim:
        scenario_params = {
            "delta_temperature_c": delta_temp,
            "override_relative_humidity_pct": override_rh,
            "delta_precipitation_mm": delta_rain
        }
        on_run_simulation(scenario_params)

    # Display What-If Results if available in payload
    what_if = payload.get("WHAT_IF")
    if what_if:
        st.markdown("---")
        st.markdown("### 📊 Hypothetical Scenario Risk Shift Results")

        base_level = what_if.get("baseline_risk_level", "UNKNOWN")
        base_index = what_if.get("baseline_risk_index", 0.0)
        scen_level = what_if.get("scenario_risk_level", "UNKNOWN")
        scen_index = what_if.get("scenario_risk_index", 0.0)
        index_delta = what_if.get("risk_index_delta", 0.0)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Baseline Risk Level", f"{base_level} ({base_index:.1f}%)")
        with col_m2:
            st.metric("Scenario Risk Level", f"{scen_level} ({scen_index:.1f}%)")
        with col_m3:
            st.metric("Risk Index Shift", f"{index_delta:+.1f}%")

        # Handle Risk Level Text cleanly to avoid contradictory text
        if base_level == scen_level:
            st.info(f"🟢 **Risk level remains {base_level}** (EVI changed by {index_delta:+.1f}%).")
        else:
            st.warning(f"⚠️ **Risk Level Shifted**: {base_level} ──► {scen_level} (EVI changed by {index_delta:+.1f}%).")

        # Factors & Narrative
        factors = what_if.get("scenario_factors", [])
        if factors:
            with st.expander("📋 View Scenario Factors Analysis", expanded=False):
                for f in factors:
                    st.markdown(f"• {f}")

        # Mandatory Hypothetical Disclaimer
        st.warning("⚠️ **HYPOTHETICAL SIMULATION**: These are simulated environmental conditions and do not represent actual weather forecasts.")
