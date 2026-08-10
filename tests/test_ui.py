import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from ui.components import (
    render_sidebar,
    validate_image,
    render_diagnosis_view,
    render_weather_view,
    render_risk_view,
    render_rag_view,
    render_scenario_view,
)

class TestStreamlitUIComponents(unittest.TestCase):

    def setUp(self):
        # Mock payload representing a complete backend result
        self.mock_payload = {
            "DIAGNOSIS": {
                "crop": "Potato",
                "disease": "Potato___Late_blight",
                "confidence": 0.98,
                "class_probabilities": {"Potato___Late_blight": 0.98, "Potato___healthy": 0.02},
                "is_low_confidence": False,
                "warning": None
            },
            "EXPLAINABILITY": {
                "gradcam_output_path": "/tmp/mock_heatmap.jpg"
            },
            "ENVIRONMENT": {
                "weather_summary": {
                    "current": {"temperature_c": 18.5, "relative_humidity_pct": 92.0, "precipitation_mm": 2.0},
                    "summary_24h": {"avg_temp_c": 18.5, "avg_humidity_pct": 92.0, "total_precipitation_mm": 2.0, "hours_high_humidity_gte_90pct": 14}
                },
                "important_weather_factors": ["High humidity hours: 14"],
                "weather_available": True,
                "warning": None
            },
            "RISK": {
                "risk_level": "HIGH",
                "risk_index": 80.0,
                "risk_factors": ["Wallin DSV score 3 (High moisture)"],
                "risk_explanation": "High Late Blight environmental risk."
            },
            "WHAT_IF": {
                "is_hypothetical": True,
                "disclaimer": "HYPOTHETICAL SCENARIO — Simulated conditions only.",
                "scenario_parameters": {"override_relative_humidity_pct": 95.0},
                "baseline_risk_level": "LOW",
                "baseline_risk_index": 0.0,
                "scenario_risk_level": "LOW",
                "scenario_risk_index": 20.0,
                "risk_level_changed": False,
                "risk_index_delta": 20.0,
                "scenario_factors": ["Simulated RH = 95.0%"]
            },
            "KNOWLEDGE": {
                "generated_query": "What extension guidance applies to potato Late Blight under HIGH risk?",
                "grounded_answer": "Late blight is caused by Phytophthora infestans...",
                "grounding_status": True,
                "insufficient_context": False,
                "gemini_available": True,
                "retrieved_sources": [{
                    "source_organization": "University of Minnesota Extension",
                    "document_title": "Late blight of tomato and potato",
                    "url_reference": "https://extension.umn.edu/disease-management/late-blight",
                    "disease": "Late Blight",
                    "similarity_score": 0.65
                }]
            },
            "DISCLAIMERS": ["Environmental Vulnerability Index is a microclimate rule index."]
        }

    def test_1_image_validation_valid(self):
        """Test 1: Image validation with non-existent file returns invalid cleanly."""
        is_valid, msg, img = validate_image("/tmp/non_existent_image_xyz.jpg")
        self.assertFalse(is_valid)
        self.assertIn("does not exist", msg)

    def test_2_image_validation_none(self):
        """Test 2: Image validation with None returns invalid cleanly."""
        is_valid, msg, img = validate_image(None)
        self.assertFalse(is_valid)

    @patch("streamlit.metric")
    @patch("streamlit.warning")
    def test_3_render_diagnosis_view(self, mock_warn, mock_metric):
        """Test 3: render_diagnosis_view renders without crashing."""
        render_diagnosis_view(self.mock_payload, pil_image=None)
        mock_metric.assert_called()

    @patch("streamlit.warning")
    def test_4_render_weather_view_unavailable(self, mock_warn):
        """Test 4: render_weather_view handles missing weather cleanly."""
        bad_payload = dict(self.mock_payload, ENVIRONMENT={"weather_available": False, "warning": "API Offline"})
        render_weather_view(bad_payload)
        mock_warn.assert_called_once()

    @patch("streamlit.info")
    @patch("streamlit.markdown")
    def test_5_render_rag_view_insufficient_context(self, mock_markdown, mock_info):
        """Test 5: render_rag_view handles insufficient context cleanly."""
        bad_rag_payload = dict(self.mock_payload)
        bad_rag_payload["KNOWLEDGE"] = dict(self.mock_payload["KNOWLEDGE"], insufficient_context=True)
        render_rag_view(bad_rag_payload)
        mock_info.assert_called()

    @patch("streamlit.warning")
    def test_6_render_rag_view_gemini_unavailable(self, mock_warn):
        """Test 6: render_rag_view handles Gemini unavailable cleanly."""
        gemini_off_payload = dict(self.mock_payload)
        gemini_off_payload["KNOWLEDGE"] = dict(self.mock_payload["KNOWLEDGE"], gemini_available=False)
        render_rag_view(gemini_off_payload)
        mock_warn.assert_called()

    @patch("streamlit.info")
    @patch("streamlit.warning")
    def test_7_scenario_same_low_to_low_risk(self, mock_warning, mock_info):
        """Test 7: Scenario view with LOW -> LOW risk displays 'Risk level remains LOW' info banner."""
        dummy_cb = lambda p: None
        render_scenario_view(self.mock_payload, dummy_cb)
        # Should call st.info for same risk level (LOW -> LOW) instead of warning
        mock_info.assert_called()

if __name__ == "__main__":
    unittest.main()
