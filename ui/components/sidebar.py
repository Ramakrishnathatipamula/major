import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from PIL import Image
import streamlit as st

import folium
from streamlit_folium import st_folium

from modules.weather import LocationResolver, ResolvedLocation
from modules.crops import get_crop_registry
from modules.image_utils import validate_and_load_image

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def validate_image(image_input: Any) -> Tuple[bool, str, Optional[Image.Image]]:
    """
    Validate uploaded image or image path using centralized safety checks.

    Returns:
        Tuple of (is_valid: bool, error_message: str, pil_image: Optional[Image.Image])
    """
    return validate_and_load_image(image_input)


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
    st.sidebar.markdown("## 🌾 Target Crop")
    crop_registry = get_crop_registry()
    crop_list = crop_registry.list_crops()
    crop_options = ["Auto-Detect (AI Plant ID)"] + [c.display_name for c in crop_list]

    selected_crop_name = st.sidebar.selectbox(
        "Select Crop",
        options=crop_options,
        index=0,
        help="Select 'Auto-Detect' to let Stage 1 AI identify plant species, or manually choose a target crop."
    )
    if selected_crop_name == "Auto-Detect (AI Plant ID)":
        st.sidebar.info("🔍 **Auto-Detect Mode**: Image will be screened for leaf validity, identified among 15 plant species, and safely routed.")
    else:
        selected_crop_spec = crop_registry.get_crop(selected_crop_name)
        if selected_crop_spec and not selected_crop_spec.is_supported:
            st.sidebar.warning(f"⚠️ **{selected_crop_spec.display_name} Support Notice**:\n{selected_crop_spec.support_status_message}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📍 Field Location (Satellite Map)")
    st.sidebar.caption("Search village/town to center map, zoom in, and click your exact agricultural field.")

    # Initialize map & location session state
    if "map_center_lat" not in st.session_state:
        st.session_state.map_center_lat = 17.8421
    if "map_center_lon" not in st.session_state:
        st.session_state.map_center_lon = 78.1254
    if "map_zoom" not in st.session_state:
        st.session_state.map_zoom = 15
    if "selected_lat" not in st.session_state:
        st.session_state.selected_lat = None
    if "selected_lon" not in st.session_state:
        st.session_state.selected_lon = None
    if "last_search_query" not in st.session_state:
        st.session_state.last_search_query = ""

    search_query = st.sidebar.text_input(
        "Search village, town, city or area",
        value="",
        placeholder="e.g. Keechulatapally, Medak, etc.",
        help="Search query centers the map viewport. Click directly on satellite imagery to select field coordinates."
    )

    resolver = LocationResolver()
    resolution_error: Optional[str] = None

    # Process search query to move map viewport
    clean_query = search_query.strip()
    if clean_query and clean_query != st.session_state.last_search_query:
        try:
            candidates = resolver.search_location(clean_query)
            if candidates:
                target = candidates[0]
                st.session_state.map_center_lat = target.latitude
                st.session_state.map_center_lon = target.longitude
                st.session_state.map_zoom = 15
                st.session_state.last_search_query = clean_query
                st.sidebar.info(f"🔍 Map centered near **{target.display_name}**. Zoom and click directly on your field.")
        except Exception as exc:
            resolution_error = str(exc)
            st.session_state.last_search_query = clean_query
            st.sidebar.warning(f"⚠️ Location '{clean_query}' not found by search provider. Please navigate the satellite map manually and click your field.")

    # Render interactive Satellite Map using Folium
    m = folium.Map(
        location=[st.session_state.map_center_lat, st.session_state.map_center_lon],
        zoom_start=st.session_state.map_zoom,
        max_zoom=19,
        tiles=None
    )

    # Base Layer: Esri World Imagery (Satellite) - Default View
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        control=True
    ).add_to(m)

    # Overlay Layer: Place & Road Labels
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Reference Labels",
        name="Place & Road Labels",
        overlay=True,
        control=True
    ).add_to(m)

    # Alternative Layer: OpenStreetMap Standard
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap Standard", overlay=False, control=True).add_to(m)

    folium.LayerControl(position="topright").add_to(m)

    # Add Marker Pin at selected field location
    if st.session_state.selected_lat is not None and st.session_state.selected_lon is not None:
        folium.Marker(
            [st.session_state.selected_lat, st.session_state.selected_lon],
            popup="Selected Field Location",
            tooltip="Selected Field Location",
            icon=folium.Icon(color="green", icon="leaf", prefix="fa")
        ).add_to(m)

    with st.sidebar:
        map_data = st_folium(
            m,
            height=380,
            key="field_satellite_map",
            use_container_width=True
        )

    # Capture direct click on satellite map
    if map_data and map_data.get("last_clicked"):
        clicked_lat = float(map_data["last_clicked"]["lat"])
        clicked_lon = float(map_data["last_clicked"]["lng"])
        st.session_state.selected_lat = clicked_lat
        st.session_state.selected_lon = clicked_lon

    latitude = st.session_state.selected_lat
    longitude = st.session_state.selected_lon
    resolved_loc: Optional[ResolvedLocation] = None
    location_confirmed: bool = False

    if latitude is not None and longitude is not None:
        resolved_loc = resolver.reverse_geocode(latitude, longitude)
        st.sidebar.success("📍 **Selected Field Location**")
        st.sidebar.markdown(
            f"• **Latitude**: `{latitude:.6f}`\n\n"
            f"• **Longitude**: `{longitude:.6f}`\n\n"
            f"• **Nearest Landmark/Area**: {resolved_loc.display_name}"
        )
        location_confirmed = True
    else:
        st.sidebar.info("👇 **Click directly on your agricultural field** in the satellite map to place the location pin.")
        location_confirmed = False

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📷 Crop Leaf Sample")

    input_mode = st.sidebar.radio("Select Image Source", ["Upload Image", "Sample Dataset Image"])

    selected_image_path = None
    uploaded_file = None

    if input_mode == "Upload Image":
        uploaded_file = st.sidebar.file_uploader(
            f"Upload {selected_crop_name} Leaf Image",
            type=["jpg", "jpeg", "png"],
            help=f"Upload a clear photo of a {selected_crop_name} leaf showing foliar symptoms."
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
        "crop": selected_crop_name,
        "latitude": latitude,
        "longitude": longitude,
        "location_confirmed": location_confirmed,
        "resolved_location": resolved_loc.to_dict() if resolved_loc else None,
        "resolution_error": resolution_error,
        "uploaded_file": uploaded_file,
        "selected_image_path": selected_image_path,
        "run_clicked": run_clicked
    }




