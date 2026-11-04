import unittest
from unittest.mock import MagicMock

from modules.diagnosis import CropDoctor
from modules.intelligence import ContextBuilder, CropDoctorIntelligenceLayer
from modules.intelligence.orchestrator import (
    _canonical_crop_name,
    _resolve_canonical_crop,
)
from modules.rag import RAGAnswerer, RAGAnswer
from modules.treatment import PotatoTreatmentRecommender


class _EmptyRetriever:
    """Minimal retriever stub so treatment tests never touch the vector store."""

    def retrieve(self, **kwargs):
        return []


class TestCanonicalCropResolution(unittest.TestCase):
    def test_detected_crop_wins_over_routing_and_requested(self):
        diag_res = {
            "detected_crop": "potato",
            "routing_decision": {"target_crop": "maize"},
        }
        self.assertEqual(_resolve_canonical_crop(diag_res, "Auto-Detect (AI Plant ID)"), "Potato")

    def test_routing_decision_is_used_when_detected_crop_missing(self):
        diag_res = {"routing_decision": {"target_crop": "tomato"}}
        self.assertEqual(_resolve_canonical_crop(diag_res, "Auto-Detect (AI Plant ID)"), "Tomato")

    def test_requested_crop_is_safe_fallback(self):
        self.assertEqual(_resolve_canonical_crop({}, "potato"), "Potato")
        self.assertEqual(_resolve_canonical_crop({}, "Tomato"), "Tomato")

    def test_unknown_requested_crop_is_preserved(self):
        self.assertEqual(
            _resolve_canonical_crop({}, "Auto-Detect (AI Plant ID)"),
            "Auto-Detect (AI Plant ID)",
        )

    def test_canonical_name_normalizes_registry_ids(self):
        self.assertEqual(_canonical_crop_name("potato"), "Potato")
        self.assertEqual(_canonical_crop_name("MAIZE"), "Maize")
        self.assertEqual(_canonical_crop_name("unknown-crop"), "unknown-crop")
        self.assertIsNone(_canonical_crop_name(None))
        self.assertIsNone(_canonical_crop_name("   "))


class TestCropIdentityPropagation(unittest.TestCase):
    def setUp(self):
        self.mock_pipeline = MagicMock(spec=CropDoctor)
        self.mock_answerer = MagicMock(spec=RAGAnswerer)
        self.mock_answerer.answer_question.return_value = RAGAnswer(
            query="q",
            answer="Advisory answer.",
            sources=[],
            is_grounded=True,
            insufficient_context=False,
            disclaimer="Educational Guidance",
        )

        self.late_blight_diag = {
            "disease": "Potato___Late_blight",
            "confidence": 0.98,
            "class_probabilities": {
                "Potato___Late_blight": 0.98,
                "Potato___Early_blight": 0.01,
                "Potato___healthy": 0.01,
            },
            "gradcam_output_path": "",
            "weather_summary": {"avg_temperature_c": 18.5},
            "risk_level": "HIGH",
            "risk_index": 80.0,
            "risk_factors": ["High humidity hours"],
            "explanation": "High Late Blight environmental risk.",
            "detected_crop": "potato",
        }

    def _build_layer(self, diag_res):
        self.mock_pipeline.diagnose.return_value = diag_res
        recommender = PotatoTreatmentRecommender(retriever=_EmptyRetriever())
        return CropDoctorIntelligenceLayer(
            pipeline=self.mock_pipeline,
            answerer=self.mock_answerer,
            treatment_recommender=recommender,
        )

    def test_auto_detect_potato_yields_potato_crop_and_treatment(self):
        layer = self._build_layer(self.late_blight_diag)

        result = layer.analyze(
            image_path="/tmp/leaf.jpg",
            latitude=44.97,
            longitude=-93.26,
            crop="Auto-Detect (AI Plant ID)",
        )

        self.assertEqual(result.crop, "Potato")
        self.assertEqual(result.requested_crop, "Auto-Detect (AI Plant ID)")
        self.assertEqual(result.disease, "Potato___Late_blight")

        # RAG query generation received the canonical crop.
        self.assertIn("potato", result.rag_query.lower())
        self.assertEqual(
            self.mock_answerer.answer_question.call_args.kwargs["crop"], "Potato"
        )
        self.assertEqual(
            self.mock_answerer.answer_question.call_args.kwargs["disease"], "Late Blight"
        )

        # Valid Potato treatment must not be suppressed by the requested label.
        treatment = result.treatment_recommendation
        self.assertIsNotNone(treatment)
        self.assertEqual(treatment["crop"], "Potato")
        self.assertEqual(treatment["status"], "VERIFIED_RECOMMENDATION_AVAILABLE")
        self.assertTrue(treatment["is_supported"])
        self.assertGreater(len(treatment["chemical_treatments"]), 0)

    def test_detected_maize_is_supported_but_gets_no_solanaceae_treatment(self):
        maize_diag = {
            "disease": "Gray Leaf Spot",
            "predicted_class_raw": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            "confidence": 0.91,
            "class_probabilities": {},
            "gradcam_output_path": "",
            "weather_summary": {"avg_temperature_c": 26.0},
            "risk_level": "LOW",
            "risk_index": 10.0,
            "risk_factors": [],
            "explanation": "Active Gray Leaf Spot detected.",
            "detected_crop": "maize",
            "routing_decision": {
                "target_crop": "maize",
                "routing_status": "ROUTED_TO_MAIZE_SPECIALIST",
            },
            "prediction_status": "SUPPORTED_CONFIDENT",
            "is_rejected": False,
        }
        layer = self._build_layer(maize_diag)

        result = layer.analyze(
            image_path="/tmp/leaf.jpg",
            latitude=44.97,
            longitude=-93.26,
            crop="Auto-Detect (AI Plant ID)",
        )

        self.assertEqual(result.crop, "Maize")
        self.assertEqual(result.requested_crop, "Auto-Detect (AI Plant ID)")
        self.assertIn("maize", result.rag_query.lower())
        self.assertEqual(
            self.mock_answerer.answer_question.call_args.kwargs["crop"], "Maize"
        )

        treatment = result.treatment_recommendation
        self.assertIsNotNone(treatment)
        self.assertEqual(treatment["crop"], "Maize")
        self.assertEqual(treatment["status"], "VERIFIED_RECOMMENDATION_UNAVAILABLE")
        self.assertTrue(treatment["is_supported"])
        self.assertEqual(treatment["chemical_treatments"], [])

        # No Potato/Tomato treatment content may leak into the Maize result.
        actionable_treatment = str(
            {
                "chemical_treatments": treatment["chemical_treatments"],
                "non_chemical_control_measures": treatment["non_chemical_control_measures"],
                "safety_and_environmental_precautions": treatment["safety_and_environmental_precautions"],
                "evidence_sources": treatment["evidence_sources"],
            }
        ).lower()
        for crop_name in ("potato", "tomato"):
            self.assertNotIn(crop_name, actionable_treatment)

    def test_to_dict_exposes_canonical_and_requested_crops(self):
        layer = self._build_layer(self.late_blight_diag)

        result = layer.analyze(
            image_path="/tmp/leaf.jpg",
            latitude=44.97,
            longitude=-93.26,
            crop="Auto-Detect (AI Plant ID)",
        )
        diag_section = result.to_dict()["DIAGNOSIS"]

        self.assertEqual(diag_section["crop"], "Potato")
        self.assertEqual(diag_section["requested_crop"], "Auto-Detect (AI Plant ID)")

    def test_context_builder_records_canonical_and_requested_crops(self):
        context = ContextBuilder.build_context(
            diagnostic_result=self.late_blight_diag,
            crop="Potato",
            requested_crop="Auto-Detect (AI Plant ID)",
        )

        self.assertEqual(context.crop, "Potato")
        self.assertEqual(context.requested_crop, "Auto-Detect (AI Plant ID)")
        self.assertEqual(context.to_dict()["crop"], "Potato")
        self.assertEqual(
            context.to_dict()["requested_crop"], "Auto-Detect (AI Plant ID)"
        )


if __name__ == "__main__":
    unittest.main()
