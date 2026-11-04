import sys
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from PIL import Image
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.image_utils import validate_and_load_image
from modules.intelligence.context_builder import (
    normalize_confidence_pct,
    ContextBuilder,
    CONFIDENCE_WARNING_THRESHOLD,
    CONFIDENCE_REJECTION_THRESHOLD,
)
from modules.treatment.recommender import PotatoTreatmentRecommender
from modules.treatment.models import TreatmentRecommendationResult
from modules.intelligence.orchestrator import (
    CropDoctorIntelligenceLayer,
    IntegratedCropDoctorResult,
)
from modules.disease_predictor import DiseasePredictor


class TestConfidenceUnitConsistency(unittest.TestCase):
    """Test confidence unit normalization and threshold evaluation."""

    def test_normalize_fractional_values(self):
        """Fractional inputs (0.0 < val <= 1.0] must be scaled to percentage [0.0, 100.0]."""
        self.assertEqual(normalize_confidence_pct(0.45), 45.0)
        self.assertEqual(normalize_confidence_pct(0.60), 60.0)
        self.assertEqual(normalize_confidence_pct(1.0), 100.0)
        self.assertEqual(normalize_confidence_pct(0.999), 99.9)

    def test_normalize_percentage_values(self):
        """Values already expressed as percentages should remain untouched."""
        self.assertEqual(normalize_confidence_pct(45.0), 45.0)
        self.assertEqual(normalize_confidence_pct(60.0), 60.0)
        self.assertEqual(normalize_confidence_pct(98.5), 98.5)
        self.assertEqual(normalize_confidence_pct(100.0), 100.0)

    def test_normalize_edge_cases(self):
        """Boundary and invalid values must be clamped or returned safely as 0.0."""
        self.assertEqual(normalize_confidence_pct(0.0), 0.0)
        self.assertEqual(normalize_confidence_pct(-10.0), 0.0)
        self.assertEqual(normalize_confidence_pct(105.0), 100.0)
        self.assertEqual(normalize_confidence_pct("invalid"), 0.0)
        self.assertEqual(normalize_confidence_pct(None), 0.0)

    def test_context_builder_preserves_percentage_units(self):
        """ContextBuilder must store confidence as percentage [0.0, 100.0]."""
        diag_fraction = {
            "disease": "Early Blight",
            "confidence": 0.85,
            "class_probabilities": {"Early Blight": 85.0, "Late Blight": 10.0, "Healthy": 5.0},
            "weather_summary": {"current": {"temperature_2m": 20.0}},
            "risk_level": "LOW",
            "risk_index": 25.0
        }
        ctx = ContextBuilder.build_context(diag_fraction)
        self.assertEqual(ctx.confidence, 85.0)
        self.assertFalse(ctx.is_low_confidence)
        self.assertFalse(ctx.is_rejected)
        self.assertEqual(ctx.prediction_status, "SUPPORTED_CONFIDENT")


class TestConfidenceThresholdBoundaries(unittest.TestCase):
    """Test configurable threshold boundaries (40.0% rejection, 60.0% warning)."""

    def test_rejection_boundary_below_40(self):
        """Predictions below 40.0% must be marked rejected and low confidence."""
        diag = {
            "disease": "Late Blight",
            "confidence": 39.5,
            "weather_summary": {"current": {"temperature_2m": 20.0}},
            "risk_level": "HIGH",
            "risk_index": 80.0
        }
        ctx = ContextBuilder.build_context(diag)
        self.assertTrue(ctx.is_rejected)
        self.assertTrue(ctx.is_low_confidence)
        self.assertEqual(ctx.prediction_status, "REJECTED_LOW_CONFIDENCE")
        self.assertIn("rejected", ctx.warning.lower())

    def test_warning_boundary_between_40_and_60(self):
        """Predictions between 40.0% and 59.9% must be uncertain but not rejected."""
        diag = {
            "disease": "Late Blight",
            "confidence": 45.0,
            "weather_summary": {"current": {"temperature_2m": 20.0}},
            "risk_level": "HIGH",
            "risk_index": 80.0
        }
        ctx = ContextBuilder.build_context(diag)
        self.assertFalse(ctx.is_rejected)
        self.assertTrue(ctx.is_low_confidence)
        self.assertEqual(ctx.prediction_status, "SUPPORTED_UNCERTAIN")
        self.assertIn("low diagnostic certainty", ctx.warning.lower())

    def test_confident_boundary_at_or_above_60(self):
        """Predictions at or above 60.0% must be confident."""
        diag_60 = {
            "disease": "Late Blight",
            "confidence": 60.0,
            "weather_summary": {"current": {"temperature_2m": 20.0}},
            "risk_level": "HIGH",
            "risk_index": 80.0
        }
        ctx_60 = ContextBuilder.build_context(diag_60)
        self.assertFalse(ctx_60.is_rejected)
        self.assertFalse(ctx_60.is_low_confidence)
        self.assertEqual(ctx_60.prediction_status, "SUPPORTED_CONFIDENT")
        self.assertIsNone(ctx_60.warning)

        diag_95 = {
            "disease": "Healthy",
            "confidence": 95.2,
            "weather_summary": {"current": {"temperature_2m": 20.0}},
            "risk_level": "LOW",
            "risk_index": 10.0
        }
        ctx_95 = ContextBuilder.build_context(diag_95)
        self.assertFalse(ctx_95.is_rejected)
        self.assertFalse(ctx_95.is_low_confidence)
        self.assertEqual(ctx_95.prediction_status, "SUPPORTED_CONFIDENT")


class TestImageInputValidationSafety(unittest.TestCase):
    """Test image validation edge cases: corrupt files, 0-byte files, tiny images, solid colors."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_nonexistent_file(self):
        """Nonexistent file path must be rejected with informative error."""
        non_existent = self.temp_path / "ghost.jpg"
        is_valid, msg, img = validate_and_load_image(non_existent)
        self.assertFalse(is_valid)
        self.assertIn("does not exist", msg)
        self.assertIsNone(img)

    def test_zero_byte_file(self):
        """0-byte empty file must be rejected."""
        empty_file = self.temp_path / "empty.jpg"
        empty_file.write_bytes(b"")
        is_valid, msg, img = validate_and_load_image(empty_file)
        self.assertFalse(is_valid)
        self.assertIn("0 bytes", msg)
        self.assertIsNone(img)

    def test_corrupt_image_file(self):
        """Files with corrupt / garbage data must be caught safely without unhandled crashes."""
        corrupt_file = self.temp_path / "corrupt.png"
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), color=(10, 120, 30)).save(buffer, "PNG")
        # Deterministically truncate a valid PNG so decoding fails every run.
        corrupt_file.write_bytes(buffer.getvalue()[:50])
        is_valid, msg, img = validate_and_load_image(corrupt_file)
        self.assertFalse(is_valid)
        self.assertTrue(
            any(token in msg.lower() for token in ("corrupt", "failed", "truncated")),
            msg,
        )
        self.assertIsNone(img)

    def test_unsupported_format(self):
        """Non-image formats and unsupported image formats (e.g. GIF) must be rejected."""
        # Case A: Plain text file
        text_file = self.temp_path / "fake.txt"
        text_file.write_text("Hello, I am not an image")
        is_valid, msg, img = validate_and_load_image(text_file)
        self.assertFalse(is_valid)
        self.assertIn("not a recognized image", msg.lower())

        # Case B: Unsupported image format (GIF)
        gif_file = self.temp_path / "anim.gif"
        gif_img = Image.new("RGB", (64, 64), color="blue")
        gif_img.save(gif_file, "GIF")
        is_valid_gif, msg_gif, img_gif = validate_and_load_image(gif_file)
        self.assertFalse(is_valid_gif)
        self.assertIn("unsupported image format", msg_gif.lower())

    def test_tiny_dimensions(self):
        """Tiny images below 32x32 must be rejected."""
        tiny_path = self.temp_path / "tiny.png"
        tiny_img = Image.new("RGB", (10, 10), color="green")
        tiny_img.save(tiny_path, "PNG")

        is_valid, msg, img = validate_and_load_image(tiny_path)
        self.assertFalse(is_valid)
        self.assertIn("too small", msg.lower())

    def test_solid_color_image(self):
        """Completely uniform blank/solid images must be flagged for lacking foliar content."""
        blank_path = self.temp_path / "solid_black.png"
        blank_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        blank_img.save(blank_path, "PNG")

        is_valid, msg, img = validate_and_load_image(blank_path)
        self.assertFalse(is_valid)
        self.assertIn("uniform", msg.lower())

    def test_valid_leaf_image(self):
        """Valid RGB image with variance should load cleanly."""
        valid_path = self.temp_path / "valid_leaf.png"
        arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(valid_path, "PNG")

        is_valid, msg, loaded = validate_and_load_image(valid_path)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.size, (224, 224))


class TestMedicineSuppressionPolicy(unittest.TestCase):
    """Test suppression of disease-specific chemical recommendations when rejected or uncertain."""

    def setUp(self):
        self.recommender = PotatoTreatmentRecommender(retriever=None)

    def test_chemical_suppression_on_rejection(self):
        """When is_rejected=True, chemical_treatments must be empty list."""
        result: TreatmentRecommendationResult = self.recommender.recommend_treatment(
            crop="Potato",
            disease="Late Blight",
            is_rejected=True
        )
        self.assertEqual(result.status, "PREDICTION_REJECTED")
        self.assertEqual(len(result.chemical_treatments), 0)
        self.assertFalse(result.is_supported)
        self.assertIn("rejected", result.summary_message.lower())
        self.assertTrue(len(result.non_chemical_control_measures) > 0)

    def test_chemical_suppression_on_low_confidence(self):
        """When is_low_confidence=True (<60%), chemical fungicides must be withheld."""
        result: TreatmentRecommendationResult = self.recommender.recommend_treatment(
            crop="Potato",
            disease="Early Blight",
            is_low_confidence=True,
            is_rejected=False
        )
        self.assertEqual(result.status, "UNCERTAIN_DIAGNOSIS_WARNING")
        self.assertEqual(len(result.chemical_treatments), 0)
        self.assertIn("withheld", result.summary_message.lower())
        self.assertTrue(len(result.non_chemical_control_measures) > 0)

    def test_chemical_suppression_on_healthy(self):
        """Healthy foliage must never recommend chemical treatments."""
        result: TreatmentRecommendationResult = self.recommender.recommend_treatment(
            crop="Potato",
            disease="Healthy",
            is_low_confidence=False,
            is_rejected=False
        )
        self.assertEqual(result.status, "HEALTHY_PREVENTIVE_ONLY")
        self.assertEqual(len(result.chemical_treatments), 0)
        self.assertTrue(result.is_healthy)

    def test_chemical_provided_on_confident_disease(self):
        """When diagnosis is confident and supported, verified chemical options must be provided."""
        result: TreatmentRecommendationResult = self.recommender.recommend_treatment(
            crop="Potato",
            disease="Late Blight",
            is_low_confidence=False,
            is_rejected=False
        )
        self.assertEqual(result.status, "VERIFIED_RECOMMENDATION_AVAILABLE")
        self.assertTrue(len(result.chemical_treatments) >= 3)
        self.assertFalse(result.is_healthy)


class TestOrchestratorPipelineIntegration(unittest.TestCase):
    """Test full orchestrator pipeline propagation of confidence, rejection, and medicine suppression."""

    def test_orchestrator_handles_uncertain_prediction(self):
        """Orchestrator must forward low confidence status and suppress chemicals in integrated result."""
        mock_pipeline = MagicMock()
        mock_pipeline.diagnose.return_value = {
            "disease": "Early Blight",
            "predicted_class_raw": "Potato___Early_blight",
            "confidence": 48.0,
            "class_probabilities": {"Early Blight": 48.0, "Late Blight": 32.0, "Healthy": 20.0},
            "gradcam_output_path": "",
            "weather_summary": {"current": {"temperature_2m": 22.0}},
            "risk_level": "MODERATE",
            "risk_index": 50.0,
            "risk_factors": ["Moderate humidity"],
            "explanation": "Moderate risk",
            "is_low_confidence": True,
            "is_rejected": False,
            "prediction_status": "SUPPORTED_UNCERTAIN"
        }

        mock_answerer = MagicMock()
        mock_answerer.answer_question.return_value = MagicMock(
            answer="Early blight management advisory.",
            sources=[],
            is_grounded=True,
            insufficient_context=False
        )

        orchestrator = CropDoctorIntelligenceLayer(
            pipeline=mock_pipeline,
            answerer=mock_answerer
        )

        result: IntegratedCropDoctorResult = orchestrator.analyze(
            image_path="dummy.jpg",
            latitude=17.8,
            longitude=78.1,
            crop="Potato"
        )

        self.assertTrue(result.is_low_confidence)
        self.assertFalse(result.is_rejected)
        self.assertEqual(result.prediction_status, "SUPPORTED_UNCERTAIN")
        self.assertIsNotNone(result.treatment_recommendation)
        self.assertEqual(result.treatment_recommendation["status"], "UNCERTAIN_DIAGNOSIS_WARNING")
        self.assertEqual(len(result.treatment_recommendation["chemical_treatments"]), 0)

        # Ensure dictionary representation carries prediction status and rejection flag
        result_dict = result.to_dict()
        diag_sec = result_dict["DIAGNOSIS"]
        self.assertEqual(diag_sec["is_low_confidence"], True)
        self.assertEqual(diag_sec["is_rejected"], False)
        self.assertEqual(diag_sec["prediction_status"], "SUPPORTED_UNCERTAIN")


if __name__ == "__main__":
    unittest.main()
