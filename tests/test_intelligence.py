import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from modules.intelligence import (
    ContextBuilder,
    DiagnosticContext,
    RAGQueryGenerator,
    CropDoctorIntelligenceLayer,
    IntegratedCropDoctorResult,
)
from modules.diagnosis import CropDoctor
from modules.simulation import ScenarioSimulator
from modules.rag import RAGAnswerer, RAGAnswer, SearchResult

class TestIntegratedIntelligenceLayer(unittest.TestCase):

    def setUp(self):
        self.mock_pipeline = MagicMock(spec=CropDoctor)
        self.mock_simulator = MagicMock(spec=ScenarioSimulator)
        self.mock_answerer = MagicMock(spec=RAGAnswerer)

        # Baseline mock diagnosis payload
        self.base_diag_res = {
            "disease": "Potato___Late_blight",
            "confidence": 0.98,
            "class_probabilities": {"Potato___Late_blight": 0.98, "Potato___Early_blight": 0.01, "Potato___healthy": 0.01},
            "gradcam_output_path": "/tmp/gradcam_sample.jpg",
            "weather_summary": {
                "avg_temperature_c": 18.5,
                "avg_relative_humidity_pct": 92.0,
                "total_precipitation_mm": 5.2,
                "high_humidity_hours": 15
            },
            "risk_level": "HIGH",
            "risk_index": 80.0,
            "risk_factors": ["Wallin DSV score 3 (High moisture)", "High humidity hours: 15"],
            "explanation": "High Late Blight environmental risk driven by 15 hours of relative humidity >= 90%."
        }
        self.mock_pipeline.diagnose.return_value = self.base_diag_res

        # Baseline mock scenario comparison payload
        self.base_scenario_comp = {
            "simulation_type": "Scenario-based environmental vulnerability simulation",
            "baseline_conditions": {},
            "baseline_risk_level": "LOW",
            "baseline_environmental_vulnerability_index": 0.0,
            "scenario_parameters": {"override_relative_humidity_pct": 95.0},
            "scenario_conditions": {},
            "scenario_risk_level": "HIGH",
            "scenario_environmental_vulnerability_index": 80.0,
            "risk_index_change": 80.0,
            "risk_level_change": "LOW -> HIGH",
            "changed_risk_factors": ["Simulated RH = 95.0%"],
            "explanation": "Scenario simulation causes risk to increase."
        }
        self.mock_simulator.simulate.return_value = self.base_scenario_comp

        # Baseline mock RAG answer payload
        self.mock_sources = [{
            "source_organization": "University of Minnesota Extension",
            "document_title": "Late blight of tomato and potato",
            "url_reference": "https://extension.umn.edu/disease-management/late-blight",
            "disease": "Late Blight",
            "similarity_score": 0.65
        }]
        self.base_rag_answer = RAGAnswer(
            query="What agricultural extension guidance applies to potato Late Blight under HIGH environmental risk?",
            answer="Late blight is caused by Phytophthora infestans. Under continuous high humidity, sporangia form on leaf undersides...",
            sources=self.mock_sources,
            is_grounded=True,
            insufficient_context=False,
            disclaimer="Educational Guidance"
        )
        self.mock_answerer.answer_question.return_value = self.base_rag_answer

    def test_1_healthy_diagnosis(self):
        """Test 1: Healthy crop diagnosis scenario handling."""
        healthy_diag = dict(self.base_diag_res, disease="Potato___healthy", confidence=0.99, risk_level="LOW", risk_index=0.0)
        self.mock_pipeline.diagnose.return_value = healthy_diag

        layer = CropDoctorIntelligenceLayer(pipeline=self.mock_pipeline, simulator=self.mock_simulator, answerer=self.mock_answerer)
        res = layer.analyze("/tmp/leaf.jpg", 44.97, -93.26)

        self.assertEqual(res.disease, "Potato___healthy")
        self.assertIn("healthy", res.rag_query.lower())

    def test_2_early_blight_diagnosis(self):
        """Test 2: Early Blight diagnosis scenario handling."""
        eb_diag = dict(self.base_diag_res, disease="Potato___Early_blight", risk_level="MODERATE")
        self.mock_pipeline.diagnose.return_value = eb_diag

        layer = CropDoctorIntelligenceLayer(pipeline=self.mock_pipeline, simulator=self.mock_simulator, answerer=self.mock_answerer)
        res = layer.analyze("/tmp/leaf.jpg", 44.97, -93.26)

        self.assertEqual(res.disease, "Potato___Early_blight")
        self.assertIn("Early Blight", res.rag_query)

    def test_3_late_blight_diagnosis(self):
        """Test 3: Late Blight diagnosis scenario handling."""
        layer = CropDoctorIntelligenceLayer(pipeline=self.mock_pipeline, simulator=self.mock_simulator, answerer=self.mock_answerer)
        res = layer.analyze("/tmp/leaf.jpg", 44.97, -93.26)

        self.assertEqual(res.disease, "Potato___Late_blight")
        self.assertEqual(res.risk_level, "HIGH")

    def test_4_low_confidence_diagnosis(self):
        """Test 4: Low-confidence diagnosis (<60%) sets warning flag."""
        low_conf_diag = dict(self.base_diag_res, confidence=0.45)
        self.mock_pipeline.diagnose.return_value = low_conf_diag

        layer = CropDoctorIntelligenceLayer(pipeline=self.mock_pipeline, simulator=self.mock_simulator, answerer=self.mock_answerer)
        res = layer.analyze("/tmp/leaf.jpg", 44.97, -93.26)

        self.assertTrue(res.is_low_confidence)
        payload = res.to_dict()
        self.assertIn("Low Diagnostic Certainty", payload["DIAGNOSIS"]["warning"])

    def test_5_rag_retrieval_integration(self):
        """Test 5: RAG retrieval integration passes generated query to RAGAnswerer."""
        layer = CropDoctorIntelligenceLayer(pipeline=self.mock_pipeline, simulator=self.mock_simulator, answerer=self.mock_answerer)
        res = layer.analyze("/tmp/leaf.jpg", 44.97, -93.26)

        self.mock_answerer.answer_question.assert_called_once_with(
            query=res.rag_query,
            crop="Potato",
            disease="Late Blight"
        )
        self.assertIn("Late Blight", res.rag_query)

    def test_6_grounded_answer_integration(self):
        """Test 6: Grounded answer payload is attached to final result."""
        layer = CropDoctorIntelligenceLayer(pipeline=self.mock_pipeline, simulator=self.mock_simulator, answerer=self.mock_answerer)
        res = layer.analyze("/tmp/leaf.jpg", 44.97, -93.26)

        self.assertTrue(res.grounding_status)
        self.assertIn("Phytophthora infestans", res.grounded_answer)

    def test_7_what_if_scenario_integration(self):
        """Test 7: What-If scenario integration marks result as hypothetical."""
        layer = CropDoctorIntelligenceLayer(pipeline=self.mock_pipeline, simulator=self.mock_simulator, answerer=self.mock_answerer)
        res = layer.analyze("/tmp/leaf.jpg", 44.97, -93.26, scenario_params={"override_relative_humidity_pct": 95.0})

        self.assertIsNotNone(res.what_if)
        self.assertTrue(res.what_if["is_hypothetical"])
        self.assertIn("HYPOTHETICAL SCENARIO", res.what_if["disclaimer"])

    def test_8_no_what_if_scenario(self):
        """Test 8: Omission of What-If scenario when scenario_params is None."""
        layer = CropDoctorIntelligenceLayer(pipeline=self.mock_pipeline, simulator=self.mock_simulator, answerer=self.mock_answerer)
        res = layer.analyze("/tmp/leaf.jpg", 44.97, -93.26, scenario_params=None)

        self.assertIsNone(res.what_if)

    def test_9_weather_failure(self):
        """Test 9: Weather failure returns weather_available=False cleanly."""
        weather_err_diag = dict(self.base_diag_res, weather_summary={"error": "Open-Meteo REST API offline"})
        self.mock_pipeline.diagnose.return_value = weather_err_diag

        layer = CropDoctorIntelligenceLayer(pipeline=self.mock_pipeline, simulator=self.mock_simulator, answerer=self.mock_answerer)
        res = layer.analyze("/tmp/leaf.jpg", 44.97, -93.26)

        self.assertFalse(res.weather_available)
        payload = res.to_dict()
        self.assertIn("unavailable", payload["ENVIRONMENT"]["warning"])

    def test_10_insufficient_rag_context(self):
        """Test 10: Insufficient RAG context returns refusal message without crashing."""
        refusal_rag_answer = RAGAnswer(
            query="Unknown query",
            answer="I don't have enough information in the verified knowledge base to answer that question.",
            sources=[],
            is_grounded=False,
            insufficient_context=True,
            disclaimer="Educational Guidance"
        )
        self.mock_answerer.answer_question.return_value = refusal_rag_answer

        layer = CropDoctorIntelligenceLayer(pipeline=self.mock_pipeline, simulator=self.mock_simulator, answerer=self.mock_answerer)
        res = layer.analyze("/tmp/leaf.jpg", 44.97, -93.26)

        self.assertTrue(res.insufficient_context)
        self.assertFalse(res.grounding_status)
        self.assertIn("don't have enough information", res.grounded_answer)

    def test_11_gemini_unavailable(self):
        """Test 11: Gemini service exception handled gracefully without crashing pipeline."""
        self.mock_answerer.answer_question.side_exception = RuntimeError("Gemini service offline")
        self.mock_answerer.answer_question.side_effect = RuntimeError("Gemini service offline")

        layer = CropDoctorIntelligenceLayer(pipeline=self.mock_pipeline, simulator=self.mock_simulator, answerer=self.mock_answerer)
        res = layer.analyze("/tmp/leaf.jpg", 44.97, -93.26)

        self.assertFalse(res.gemini_available)
        self.assertIn("unavailable", res.grounded_answer)

    def test_12_source_metadata_propagation(self):
        """Test 12: Source metadata propagates correctly into retrieved_sources list."""
        layer = CropDoctorIntelligenceLayer(pipeline=self.mock_pipeline, simulator=self.mock_simulator, answerer=self.mock_answerer)
        res = layer.analyze("/tmp/leaf.jpg", 44.97, -93.26)

        self.assertTrue(len(res.retrieved_sources) > 0)
        self.assertEqual(res.retrieved_sources[0]["source_organization"], "University of Minnesota Extension")
        self.assertEqual(res.retrieved_sources[0]["url_reference"], "https://extension.umn.edu/disease-management/late-blight")

    def test_13_final_structured_payload(self):
        """Test 13: to_dict() returns full structured payload with all 7 key sections."""
        layer = CropDoctorIntelligenceLayer(pipeline=self.mock_pipeline, simulator=self.mock_simulator, answerer=self.mock_answerer)
        res = layer.analyze("/tmp/leaf.jpg", 44.97, -93.26, scenario_params={"override_relative_humidity_pct": 95.0})

        payload = res.to_dict()
        required_keys = ["DIAGNOSIS", "EXPLAINABILITY", "ENVIRONMENT", "RISK", "WHAT_IF", "KNOWLEDGE", "DISCLAIMERS"]
        for key in required_keys:
            self.assertIn(key, payload)

        self.assertEqual(len(payload["DISCLAIMERS"]), 4)

if __name__ == "__main__":
    unittest.main()
