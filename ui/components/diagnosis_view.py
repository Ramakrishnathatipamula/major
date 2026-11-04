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
    # UI display formatting preserves internal confidence value without mutating it
    conf_pct = round(confidence, 1)
    is_low_conf = diag.get("is_low_confidence", False)
    is_rejected = diag.get("is_rejected", False)
    pred_status = diag.get("prediction_status", "SUPPORTED_CONFIDENT")

    col_meta1, col_meta2, col_meta3 = st.columns(3)
    with col_meta1:
        st.metric("Crop Type", diag.get("crop", "Potato"))
    with col_meta2:
        st.metric("Visual Diagnosis", disease_name)
    with col_meta3:
        if is_rejected:
            st.metric("Diagnostic Status", "Inconclusive / Rejected ❌", f"{conf_pct}%")
        elif is_low_conf:
            st.metric("Diagnostic Status", "Uncertain (Scout Field) ⚠️", f"{conf_pct}%")
        else:
            st.metric("Diagnostic Status", "Supported / Confident 🟢", f"{conf_pct}%")

    # Diagnostic Alerts and Scientific Uncertainty Disclaimers
    if "Model Pending" in disease_name or "specialist pending" in disease_name.lower() or "unsupported" in disease_name.lower():
        st.info(f"ℹ️ **Crop Specialist Status**: Disease specialist model for **{diag.get('crop')}** is not available for this request. No disease prediction or chemical treatment is being fabricated.")
    elif is_rejected:
        st.error(
            f"❌ **Prediction Inconclusive / Rejected (< 40.0% Confidence)**: "
            f"Classification certainty ({conf_pct}%) is near random chance or input validation failed. "
            "Disease-specific chemical fungicide recommendations are suppressed to prevent misapplication. "
            "Please inspect secondary canopy leaves and re-take the photo under even lighting."
        )
    elif is_low_conf:
        st.warning(
            f"⚠️ **Low Diagnostic Certainty (< 60.0% Confidence)**: "
            f"Tentative prediction ({conf_pct}%). "
            "Chemical treatments are withheld pending physical confirmation of foliar lesions in the field."
        )


    # Class Probabilities Breakdown
    with st.expander("📊 View Class Probabilities Breakdown", expanded=False):
        class_probs = diag.get("class_probabilities", {})
        if class_probs:
            # Clean class names for display
            clean_probs = {k.split("___", 1)[-1].replace("_", " "): v for k, v in class_probs.items()}
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

    st.caption("ℹ️ *Grad-CAM highlights image regions that contributed to the model's classification decision. Softmax confidence indicates the neural network's relative activation distribution over trained classes, not a biological guarantee of leaf pathology.*")
