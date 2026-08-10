from typing import Dict, Any
import streamlit as st

def render_weather_view(payload: Dict[str, Any]):
    """
    Render Microclimate Weather Metrics Section.
    """
    st.markdown("## 🌤️ Microclimate Weather Environment")

    env = payload.get("ENVIRONMENT", {})
    weather_avail = env.get("weather_available", True)
    warning_msg = env.get("warning")

    if not weather_avail or warning_msg:
        st.warning("⚠️ **Live Weather Service Unavailable**: Weather metrics could not be retrieved from the microclimate API.")
        return

    weather_sum = env.get("weather_summary", {})
    current = weather_sum.get("current", {})
    summary_24h = weather_sum.get("summary_24h", {})

    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    with col_w1:
        temp = current.get("temperature_c", summary_24h.get("avg_temp_c", 0.0))
        st.metric("Temperature", f"{temp} °C", help="Current field ambient temperature")
    with col_w2:
        rh = current.get("relative_humidity_pct", summary_24h.get("avg_humidity_pct", 0.0))
        st.metric("Relative Humidity", f"{rh} %", help="Current canopy relative humidity")
    with col_w3:
        rain = summary_24h.get("total_precipitation_mm", current.get("precipitation_mm", 0.0))
        st.metric("24h Total Rain", f"{rain} mm", help="Cumulative 24-hour rainfall")
    with col_w4:
        rh_hours = summary_24h.get("hours_high_humidity_gte_90pct", 0)
        st.metric("High RH Hours (≥90%)", f"{rh_hours} Hours", help="Hours in 24h window with RH >= 90%")

    # Optional 24h forecast metrics expander
    with st.expander("📈 View Detailed 24-Hour Microclimate Summary", expanded=False):
        st.markdown(f"• **24h Temperature Range**: Min {summary_24h.get('min_temp_c', 0.0)}°C | Max {summary_24h.get('max_temp_c', 0.0)}°C | Avg {summary_24h.get('avg_temp_c', 0.0)}°C")
        st.markdown(f"• **24h Peak Humidity**: {summary_24h.get('max_humidity_pct', 0.0)}%")
        st.markdown(f"• **Peak Wind Speed**: {summary_24h.get('max_wind_speed_kmh', 0.0)} km/h")
