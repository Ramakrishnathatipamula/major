from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import streamlit as st

def render_diagnosis_view(payload: Dict[str, Any], pil_image: Optional[Image.Image]):
    """
    Render Diagnosis and Grad-CAM Explainability Section.
    """
    diag = payload.get("DIAGNOSIS", {})
    explain = payload.get("EXPLAINABILITY", {})

    st.markdown("## 🔍 Visual Diagnosis & Grad-CAM Explainability")

    # Disease Name Header & Confidence Metric
    disease_name = diag.get("disease", "Unknown")
    confidence = diag.get("confidence", 0.0)
    conf_pct = round(confidence * 100, 1) if confidence <= 1.0 else round(confidence, 1)
    is_low_conf = diag.get("is_low_confidence", False)

    col_meta1, col_meta2, col_meta3 = st.columns(3)
    with col_meta1:
        st.metric("Crop Type", diag.get("crop", "Potato"))
    with col_meta2:
        st.metric("Visual Diagnosis", disease_name)
    with col_meta3:
        st.metric("Classification Confidence", f"{conf_pct}%")

    # Low Confidence Warning Alert
    if is_low_conf or diag.get("warning"):
        st.warning("⚠️ **Low Diagnostic Certainty (< 60%)**: Classification confidence is low. Inspect secondary canopy leaves.")

    # Class Probabilities Breakdown
    with st.expander("📊 View Class Probabilities Breakdown", expanded=False):
        class_probs = diag.get("class_probabilities", {})
        if class_probs:
            # Clean class names for display
            clean_probs = {k.replace("Potato___", "").replace("_", " "): v for k, v in class_probs.items()}
            st.bar_chart(clean_probs)

    st.markdown("---")

    # Side-by-Side Original Leaf vs Grad-CAM Heatmap
    col_img1, col_img2 = st.columns(2)

    with col_img1:
        st.subheader("📷 Original Leaf Sample")
        if pil_image:
            st.image(pil_image, use_container_width=True, caption="Uploaded Leaf Input")
        else:
            st.info("Original image preview unavailable.")

    with col_img2:
        st.subheader("🔥 Grad-CAM Visual Attention Heatmap")
        heatmap_path = explain.get("gradcam_output_path")
        if heatmap_path and Path(heatmap_path).exists():
            st.image(heatmap_path, use_container_width=True, caption="CNN Lesion Feature Attention Overlay")
        else:
            st.info("Grad-CAM visual heatmap unavailable.")

    st.caption("ℹ️ *Grad-CAM highlights image regions that contributed to the model's classification decision.*")
