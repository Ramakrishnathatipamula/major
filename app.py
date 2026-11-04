import sys
import os
from pathlib import Path
from PIL import Image
import streamlit as st

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ui.components import (
    render_sidebar,
    validate_image,
    render_diagnosis_view,
    render_treatment_view,
    render_weather_view,
    render_risk_view,
    render_rag_view,
    render_scenario_view,
)
from modules.intelligence import CropDoctorIntelligenceLayer

# Page Config
st.set_page_config(
    page_title="AI Crop Doctor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
css_path = ROOT_DIR / "ui" / "styles" / "custom.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Application Header
st.title("🌿 AI Crop Doctor")
st.markdown("##### Integrated Crop Disease, Risk & Agricultural Knowledge System")
st.markdown("---")

# Initialize Session State
if "diagnostic_payload" not in st.session_state:
    st.session_state.diagnostic_payload = None
if "pil_image" not in st.session_state:
    st.session_state.pil_image = None
if "current_image_path" not in st.session_state:
    st.session_state.current_image_path = None
if "current_lat" not in st.session_state:
    st.session_state.current_lat = None
if "current_lon" not in st.session_state:
    st.session_state.current_lon = None


# Render Sidebar & Get User Inputs
sidebar_inputs = render_sidebar()

selected_crop = sidebar_inputs.get("crop", "Potato")
latitude = sidebar_inputs.get("latitude")
longitude = sidebar_inputs.get("longitude")
location_confirmed = sidebar_inputs.get("location_confirmed", False)
uploaded_file = sidebar_inputs.get("uploaded_file")
selected_image_path = sidebar_inputs.get("selected_image_path")
run_clicked = sidebar_inputs.get("run_clicked", False)

if "current_crop" not in st.session_state:
    st.session_state.current_crop = "Potato"

# Lazy-loaded Intelligence Layer singleton
@st.cache_resource
def get_intelligence_layer():
    return CropDoctorIntelligenceLayer()

intelligence_layer = get_intelligence_layer()

# Handle "Run Crop Doctor Diagnosis" Button Action
if run_clicked:
    if latitude is None or longitude is None or not location_confirmed:
        st.error("❌ **Location Unresolved**: Cannot retrieve microclimate weather or evaluate epidemiological disease risk without a confirmed field location. Please resolve and confirm a field location in the sidebar.")
    else:
        # Determine image source
        image_to_validate = uploaded_file if uploaded_file is not None else selected_image_path
        is_valid, err_msg, pil_img = validate_image(image_to_validate)

        if not is_valid:
            st.error(f"❌ **Image Error**: {err_msg}")
        else:
            # Save temp file if uploaded_file
            temp_img_path = None
            if uploaded_file is not None:
                temp_dir = ROOT_DIR / "data" / "processed" / "temp_uploads"
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_img_path = temp_dir / uploaded_file.name
                with open(temp_img_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                final_img_path = str(temp_img_path)
            else:
                final_img_path = str(selected_image_path)

            st.session_state.pil_image = pil_img
            st.session_state.current_image_path = final_img_path
            st.session_state.current_lat = latitude
            st.session_state.current_lon = longitude
            st.session_state.current_crop = selected_crop

            # Execute backend analysis pipeline
            with st.spinner(f"Running AI Crop Doctor Intelligence Pipeline for {selected_crop}..."):
                try:
                    intel_res = intelligence_layer.analyze(
                        image_path=final_img_path,
                        latitude=latitude,
                        longitude=longitude,
                        crop=selected_crop,
                        scenario_params=None
                    )
                    st.session_state.diagnostic_payload = intel_res.to_dict()
                    st.success(f"✅ **Crop Doctor Analysis Complete for {selected_crop}!**")
                except Exception as exc:
                    st.error(f"❌ **Backend Processing Error**: Failed to complete analysis ({exc}).")


# Callback handler for What-If Scenario Simulation
def run_scenario_callback(scenario_params: dict):
    if not st.session_state.current_image_path:
        st.error("Please run primary Crop Doctor diagnosis first.")
        return

    with st.spinner("Executing What-If Microclimate Scenario Simulation..."):
        try:
            intel_res = intelligence_layer.analyze(
                image_path=st.session_state.current_image_path,
                latitude=st.session_state.current_lat,
                longitude=st.session_state.current_lon,
                crop=st.session_state.get("current_crop", "Potato"),
                scenario_params=scenario_params
            )

            st.session_state.diagnostic_payload = intel_res.to_dict()
            st.success("✅ **What-If Scenario Simulation Complete!**")
        except Exception as exc:
            st.error(f"❌ **Scenario Simulation Error**: {exc}")

# Main Dashboard Workspace
payload = st.session_state.diagnostic_payload

if payload is None:
    st.info("👈 **Welcome to AI Crop Doctor!** Select an image and field location in the sidebar, then click **'Run Crop Doctor Diagnosis'** to start.")
else:
    # Render Dashboard Views
    render_diagnosis_view(payload, st.session_state.pil_image)
    st.markdown("---")
    render_treatment_view(payload)
    st.markdown("---")
    render_weather_view(payload)
    st.markdown("---")
    render_risk_view(payload)
    st.markdown("---")
    render_rag_view(payload)
    st.markdown("---")
    render_scenario_view(payload, run_scenario_callback)

    # Footer
    st.markdown("---")
    st.caption("🌿 **AI Crop Doctor Intelligence System** — Empowering smallholder farmers with multi-modal AI perception, microclimate physics, and grounded agricultural extension knowledge.")

