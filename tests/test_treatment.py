import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.treatment import (
    PotatoTreatmentRecommender,
    TreatmentOption,
    TreatmentRecommendationResult,
    TreatmentCategory,
)
from modules.intelligence import CropDoctorIntelligenceLayer
from ui.components.treatment_view import render_treatment_view


class TestPotatoTreatmentModule(unittest.TestCase):

    def setUp(self):
        self.recommender = PotatoTreatmentRecommender()
        self.allowed_active_ingredients = {
            "Mancozeb",
            "Chlorothalonil",
            "Copper Oxychloride",
            "Cymoxanil + Mancozeb",
            "Metalaxyl + Mancozeb",
            "Difenoconazole"
        }

    def test_1_early_blight_treatment_lookup(self):
        """Test 1: Potato Early Blight returns verified CIBRC/TNAU treatments and no hallucinations."""
        result = self.recommender.recommend_treatment("Potato", "Potato___Early_blight")
        self.assertTrue(result.is_supported)
        self.assertFalse(result.is_healthy)
        self.assertEqual(result.status, "VERIFIED_RECOMMENDATION_AVAILABLE")
        self.assertGreater(len(result.chemical_treatments), 0)

        # Check that Mancozeb 75% WP is included with verified 0.2% rate
        ai_names = [opt.active_ingredient for opt in result.chemical_treatments]
        self.assertIn("Mancozeb", ai_names)
        self.assertIn("Chlorothalonil", ai_names)

        # Verify rate and timing are documented
        mancozeb_opt = next(o for o in result.chemical_treatments if o.active_ingredient == "Mancozeb")
        self.assertIn("0.2%", mancozeb_opt.application_rate)
        self.assertIn("concentric", mancozeb_opt.timing_and_frequency)

        # Non-chemical control & safety
        self.assertGreater(len(result.non_chemical_control_measures), 0)
        self.assertGreater(len(result.safety_and_environmental_precautions), 0)
        self.assertGreater(len(result.evidence_sources), 0)

    def test_2_late_blight_treatment_lookup(self):
        """Test 2: Potato Late Blight returns both protectant and curative combination treatments."""
        result = self.recommender.recommend_treatment("Potato", "Potato___Late_blight")
        self.assertTrue(result.is_supported)
        self.assertFalse(result.is_healthy)
        self.assertEqual(result.status, "VERIFIED_RECOMMENDATION_AVAILABLE")

        categories = [opt.category for opt in result.chemical_treatments]
        self.assertIn(TreatmentCategory.PROPHYLACTIC_PROTECTANT.value, categories)
        self.assertIn(TreatmentCategory.CURATIVE_SYSTEMIC.value, categories)

        ai_names = [opt.active_ingredient for opt in result.chemical_treatments]
        self.assertIn("Cymoxanil + Mancozeb", ai_names)
        self.assertIn("Metalaxyl + Mancozeb", ai_names)

        # Cultural measures include dehaulming and destroying cull piles
        cultural_str = " ".join(result.non_chemical_control_measures).lower()
        self.assertIn("dehaulming", cultural_str)
        self.assertIn("cull", cultural_str)

    def test_3_healthy_prediction_returns_no_chemical_recommendations(self):
        """Test 3: Healthy prediction strictly returns zero chemical fungicides."""
        result = self.recommender.recommend_treatment("Potato", "Potato___healthy")
        self.assertTrue(result.is_supported)
        self.assertTrue(result.is_healthy)
        self.assertEqual(result.status, "HEALTHY_PREVENTIVE_ONLY")

        # STRICT VERIFICATION: Zero chemical treatments
        self.assertEqual(len(result.chemical_treatments), 0)
        self.assertIn("No chemical", result.summary_message)
        self.assertGreater(len(result.non_chemical_control_measures), 0)

    def test_4_unknown_disease_handling(self):
        """Test 4: Unknown/unsupported disease returns clear insufficient-evidence status."""
        result = self.recommender.recommend_treatment("Potato", "Potato___Unknown_Virus_XYZ")
        self.assertFalse(result.is_supported)
        self.assertFalse(result.is_healthy)
        self.assertEqual(result.status, "UNSUPPORTED_DISEASE")
        self.assertEqual(len(result.chemical_treatments), 0)
        self.assertIn("No verified treatment recommendation", result.summary_message)

    def test_5_unsupported_crop_handling(self):
        """Test 5: Supported crops without a verified treatment registry report VERIFIED_RECOMMENDATION_UNAVAILABLE with zero chemicals."""
        result = self.recommender.recommend_treatment("Rice", "Rice___Brown_spot")
        self.assertTrue(result.is_supported)
        self.assertEqual(result.status, "VERIFIED_RECOMMENDATION_UNAVAILABLE")
        self.assertEqual(len(result.chemical_treatments), 0)
        self.assertIn("no verified", result.summary_message.lower())

    def test_6_empty_or_insufficient_retrieved_evidence(self):
        """Test 6: Empty retriever response gracefully falls back to verified authority records without crashing."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        recommender = PotatoTreatmentRecommender(retriever=mock_retriever)

        result = recommender.recommend_treatment("Potato", "Late Blight")
        self.assertTrue(result.is_supported)
        self.assertEqual(result.status, "VERIFIED_RECOMMENDATION_AVAILABLE")
        self.assertGreater(len(result.chemical_treatments), 0)
        self.assertGreater(len(result.evidence_sources), 0)

    def test_7_prevention_of_unsupported_medicines_and_unverified_rates(self):
        """Test 7: Strictly enforces only verified CIBRC active ingredients; no invented drugs or blank rates."""
        for disease in ["Early Blight", "Late Blight"]:
            res = self.recommender.recommend_treatment("Potato", disease)
            for opt in res.chemical_treatments:
                # Active ingredient must be recognized
                self.assertIn(opt.active_ingredient, self.allowed_active_ingredients)
                # Rate must not be empty or unspecified
                self.assertTrue(bool(opt.application_rate and opt.application_rate.strip()))
                # PPE precautions must be provided
                self.assertGreater(len(opt.ppe_and_safety_precautions), 0)
                # Source must be indicated
                self.assertTrue(bool(opt.evidence_source_title))

    def test_8_integration_with_orchestrator(self):
        """Test 8: Integrated CropDoctorIntelligenceLayer incorporates TREATMENT payload cleanly."""
        mock_pipeline = MagicMock()
        mock_pipeline.diagnose.return_value = {
            "disease": "Late Blight",
            "predicted_disease": "Potato___Late_blight",
            "confidence": 95.5,
            "class_probabilities": {"Late Blight": 95.5, "Early Blight": 3.0, "Healthy": 1.5},
            "gradcam_output_path": "/tmp/gradcam.jpg",
            "weather_summary": {"summary_24h": {"avg_temp_c": 18.0, "avg_humidity_pct": 92.0}},
            "risk_level": "HIGH",
            "risk_index": 85.0,
            "risk_factors": ["High humidity"],
            "explanation": "High risk conditions"
        }

        mock_answerer = MagicMock()
        mock_ans_obj = MagicMock()
        mock_ans_obj.answer = "Verified late blight guidance."
        mock_ans_obj.is_grounded = True
        mock_ans_obj.insufficient_context = False
        mock_ans_obj.sources = []
        mock_answerer.answer_question.return_value = mock_ans_obj

        intel = CropDoctorIntelligenceLayer(
            pipeline=mock_pipeline,
            answerer=mock_answerer
        )

        result = intel.analyze(
            image_path="/tmp/fake_potato.jpg",
            latitude=17.84,
            longitude=78.12,
            crop="Potato"
        )

        res_dict = result.to_dict()
        self.assertIn("TREATMENT", res_dict)
        treatment = res_dict["TREATMENT"]
        self.assertIsNotNone(treatment)
        self.assertEqual(treatment["crop"], "Potato")
        self.assertEqual(treatment["predicted_disease"], "Late Blight")
        self.assertEqual(treatment["status"], "VERIFIED_RECOMMENDATION_AVAILABLE")
        self.assertGreater(len(treatment["chemical_treatments"]), 0)

    @patch("streamlit.caption")
    @patch("streamlit.info")
    @patch("streamlit.warning")
    @patch("streamlit.success")
    @patch("streamlit.markdown")
    @patch("streamlit.metric")
    def test_9_render_treatment_view_ui(self, mock_metric, mock_md, mock_succ, mock_warn, mock_info, mock_cap):
        """Test 9: render_treatment_view executes cleanly for Late Blight, Early Blight, Healthy, and None."""
        # 1. Late Blight Payload
        late_blight_res = self.recommender.recommend_treatment("Potato", "Late Blight")
        payload = {"TREATMENT": late_blight_res.to_dict()}
        render_treatment_view(payload)
        mock_metric.assert_called()

        # 2. Healthy Payload
        healthy_res = self.recommender.recommend_treatment("Potato", "Healthy")
        render_treatment_view({"TREATMENT": healthy_res.to_dict()})
        mock_succ.assert_called()

        # 3. Unsupported Disease Payload
        unsupported_res = self.recommender.recommend_treatment("Potato", "Unknown_Disease")
        render_treatment_view({"TREATMENT": unsupported_res.to_dict()})
        mock_warn.assert_called()

        # 4. Missing Treatment Key Payload
        render_treatment_view({})
        mock_info.assert_called()


if __name__ == "__main__":
    unittest.main()
