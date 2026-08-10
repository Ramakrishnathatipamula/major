from typing import Dict, Any
import streamlit as st

def render_risk_view(payload: Dict[str, Any]):
    """
    Render Risk Assessment & Environmental Vulnerability Index Section.
    """
    st.markdown("## ⚠️ Environmental Vulnerability Assessment")

    risk_data = payload.get("RISK", {})
    risk_level = risk_data.get("risk_level", "UNKNOWN")
    risk_index = risk_data.get("risk_index", 0.0)
    risk_factors = risk_data.get("risk_factors", [])
    risk_explanation = risk_data.get("risk_explanation", "")

    col_r1, col_r2 = st.columns([1, 2])

    with col_r1:
        st.metric("Environmental Vulnerability Index", f"{risk_index:.1f}%")

        # Color-coded risk badge based strictly on backend risk_level
        if risk_level == "HIGH":
            st.error("🔴 **Risk Level: HIGH**")
        elif risk_level == "MODERATE":
            st.warning("🟡 **Risk Level: MODERATE**")
        elif risk_level == "LOW":
            st.success("🟢 **Risk Level: LOW**")
        else:
            st.info(f"⚪ **Risk Level: {risk_level}**")

    with col_r2:
        st.markdown("### 📋 Primary Risk Drivers & Epidemiological Factors")
        if risk_explanation:
            st.markdown(f"*{risk_explanation}*")

        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"• {factor}")
        else:
            st.markdown("• *No adverse environmental risk factors detected.*")

    # Standard Risk Disclaimer
    st.caption("ℹ️ *The Environmental Vulnerability Index is derived from microclimate rule models (Wallin DSV / P-Days) and is NOT a calibrated statistical probability.*")
