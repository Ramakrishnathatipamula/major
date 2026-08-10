import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from PIL import Image
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Location Presets
LOCATION_PRESETS = {
    "Custom Coordinates": (44.9778, -93.2650),
    "US Midwest Potato Belt (Minneapolis, MN)": (44.9778, -93.2650),
    "Idaho Potato Belt (Boise, ID)": (43.6150, -116.2023),
    "Central Europe Potato Belt (Berlin, DE)": (52.5200, 13.4050),
}


def validate_image(image_input: Any) -> Tuple[bool, str, Optional[Image.Image]]:
    """
    Validate uploaded image or image path.

    Returns:
        Tuple of (is_valid: bool, error_message: str, pil_image: Optional[Image.Image])
    """
    if image_input is None:
        return False, "No image uploaded or selected.", None

    try:
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                return False, f"Image file does not exist: {path.name}", None
            img = Image.open(path)
            return True, "", img
        else:
            # Streamlit UploadedFile object
            if hasattr(image_input, "size") and image_input.size > 10 * 1024 * 1024:
                return False, "Image file size exceeds 10MB limit.", None
            img = Image.open(image_input)
            return True, "", img
    except Exception as exc:
        return False, f"Invalid or unreadable image file ({exc}).", None


def check_system_status() -> Dict[str, str]:
    """Check actual availability of system components."""
    status = {
        "weather_api": "Ready 🟢",
        "vector_store": "Ready 🟢",
        "gemini_llm": "Ready 🟢"
    }
    # Check Gemini API Key
    from config import settings
    if not settings.GEMINI_API_KEY:
        status["gemini_llm"] = "API Key Missing 🟡"

    # Check Vector Store
    try:
        from modules.rag import VectorStore
        vs = VectorStore(persist_dir=ROOT_DIR / "data" / "vector_store")
        count = vs.count()
        status["vector_store"] = f"{count} Chunks Indexed 🟢"
    except Exception:
        status["vector_store"] = "Offline 🔴"

    return status


def render_sidebar() -> Dict[str, Any]:
    """
    Render sidebar controls and return user inputs.
    """
    st.sidebar.markdown("## 📍 Location Settings")

    preset_name = st.sidebar.selectbox("Location Presets", list(LOCATION_PRESETS.keys()))
    default_lat, default_lon = LOCATION_PRESETS[preset_name]

    latitude = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, value=default_lat, format="%.4f")
    longitude = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=default_lon, format="%.4f")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📷 Crop Leaf Sample")

    input_mode = st.sidebar.radio("Select Image Source", ["Upload Image", "Sample Dataset Image"])

    selected_image_path = None
    uploaded_file = None

    if input_mode == "Upload Image":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Potato Leaf Image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of a potato leaf showing foliar symptoms."
        )
    else:
        sample_dir = ROOT_DIR / "data" / "raw" / "plant_village_subset"
        sample_files = []
        if sample_dir.exists():
            for p in sample_dir.glob("*/*.[jJ][pP][gG]"):
                sample_files.append(p)
            for p in sample_dir.glob("*/*.[pP][nN][gG]"):
                sample_files.append(p)

        if sample_files:
            file_map = {f"{f.parent.name} / {f.name}": str(f) for f in sample_files}
            selected_label = st.sidebar.selectbox("Select Sample Image", list(file_map.keys()))
            selected_image_path = file_map[selected_label]
        else:
            st.sidebar.warning("No sample images found in dataset directory.")

    st.sidebar.markdown("---")
    run_clicked = st.sidebar.button("🔍 Run Crop Doctor Diagnosis", type="primary", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🟢 System Status")
    sys_status = check_system_status()
    st.sidebar.markdown(f"• **Weather API**: {sys_status['weather_api']}")
    st.sidebar.markdown(f"• **Vector Store**: {sys_status['vector_store']}")
    st.sidebar.markdown(f"• **Gemini LLM**: {sys_status['gemini_llm']}")

    return {
        "latitude": latitude,
        "longitude": longitude,
        "uploaded_file": uploaded_file,
        "selected_image_path": selected_image_path,
        "run_clicked": run_clicked
    }
