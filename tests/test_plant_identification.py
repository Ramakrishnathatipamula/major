import sys
import unittest
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.identification import (
    LeafVerifier,
    LeafVerificationResult,
    PlantIdentifier,
    PlantIdentificationResult,
    PlantRouter,
    RoutingDecision,
    PLANT_SPECIES_CLASSES,
    PLANT_SPECIES_DISPLAY_NAMES,
)
from modules.crops import get_crop_registry
from modules.diagnosis import CropDoctor, DiagnosticResult
from modules.treatment import PotatoTreatmentRecommender


class TestLeafVerifier(unittest.TestCase):
    """Test LeafVerifier heuristic and visual screening."""

    def setUp(self):
        self.verifier = LeafVerifier()
        self.real_leaf_path = ROOT_DIR / "data" / "raw" / "plant_village_subset" / "Potato___healthy" / "real_001.JPG"

    def test_real_leaf_verification(self):
        """Genuine leaf image passes leaf verification with confirmed foliar traits."""
        if not self.real_leaf_path.exists():
            self.skipTest(f"Sample image not found: {self.real_leaf_path}")

        res = self.verifier.verify_image(self.real_leaf_path)
        self.assertIsInstance(res, LeafVerificationResult)
        self.assertTrue(res.is_leaf)
        self.assertEqual(res.status, "VERIFIED_LEAF")
        self.assertGreaterEqual(res.confidence, 60.0)
        self.assertGreater(res.metrics["foliar_ratio"], 0.08)

    def test_blank_solid_images_rejected(self):
        """Uniform or blank images are rejected as REJECTED_BLANK."""
        # Solid white
        white_img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        res_white = self.verifier.verify_image(white_img)
        self.assertFalse(res_white.is_leaf)
        self.assertEqual(res_white.status, "REJECTED_BLANK")

        # Solid black
        black_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        res_black = self.verifier.verify_image(black_img)
        self.assertFalse(res_black.is_leaf)
        self.assertEqual(res_black.status, "REJECTED_BLANK")

    def test_non_foliar_synthetic_colors_rejected(self):
        """Synthetic blocks lacking foliar chlorophyll pigment spectrum are rejected as REJECTED_NON_LEAF."""
        # Pure blue/cyan pattern with noise
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        arr[:, :, 2] = 220  # Strong blue channel
        arr[:, :, 1] = 50   # Weak green
        arr[:, :, 0] = 30   # Weak red
        # Add texture noise so standard deviation is non-zero
        noise = np.random.randint(-15, 15, (224, 224, 3), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        non_foliar_img = Image.fromarray(arr)
        res = self.verifier.verify_image(non_foliar_img)
        self.assertFalse(res.is_leaf)
        self.assertEqual(res.status, "REJECTED_NON_LEAF")

    def test_corrupt_and_nonexistent_files_rejected(self):
        """Corrupt or missing files fail safely with REJECTED_CORRUPT."""
        res = self.verifier.verify_image("/tmp/does_not_exist_xyz.jpg")
        self.assertFalse(res.is_leaf)
        self.assertEqual(res.status, "REJECTED_CORRUPT")

        res_none = self.verifier.verify_image(None)
        self.assertFalse(res_none.is_leaf)
        self.assertEqual(res_none.status, "REJECTED_CORRUPT")


class TestPlantIdentifier(unittest.TestCase):
    """Test Stage 1 PlantIdentifier model loading and inference."""

    def setUp(self):
        self.identifier = PlantIdentifier()
        self.potato_leaf = ROOT_DIR / "data" / "raw" / "plant_village_subset" / "Potato___healthy" / "real_001.JPG"

    def test_model_checkpoint_available(self):
        """Verify plant identification checkpoint exists on disk."""
        self.assertTrue(self.identifier.is_model_available(), "Plant identification checkpoint must exist")
        self.assertEqual(len(self.identifier.classes), 15)
        self.assertIn("potato", self.identifier.classes)
        self.assertIn("tomato", self.identifier.classes)
        self.assertIn("maize", self.identifier.classes)
        self.assertIn("wheat", self.identifier.classes)

    def test_potato_identification(self):
        """Genuine potato leaf is recognized as Potato with high certainty."""
        if not self.potato_leaf.exists():
            self.skipTest(f"Sample image not found: {self.potato_leaf}")

        res = self.identifier.identify(self.potato_leaf)
        self.assertIsInstance(res, PlantIdentificationResult)
        self.assertEqual(res.crop_id, "potato")
        self.assertEqual(res.display_name, "Potato")
        self.assertTrue(res.is_supported)
        self.assertFalse(res.is_rejected)
        self.assertGreaterEqual(res.confidence, 60.0)
        self.assertEqual(res.status, "CONFIDENT_IDENTIFICATION")

    def test_invalid_input_handling(self):
        """Invalid or corrupt image returns INVALID_INPUT status."""
        res = self.identifier.identify("/tmp/does_not_exist.jpg")
        self.assertEqual(res.status, "INVALID_INPUT")
        self.assertTrue(res.is_rejected)
        self.assertEqual(res.confidence, 0.0)


class TestPlantRouter(unittest.TestCase):
    """Test confidence-based routing to disease specialists and safe fallback."""

    def setUp(self):
        self.router = PlantRouter()
        self.potato_leaf = ROOT_DIR / "data" / "raw" / "plant_village_subset" / "Potato___healthy" / "real_001.JPG"

    def test_auto_detect_potato_routing(self):
        """Auto-detect mode with Potato leaf correctly routes to Potato specialist."""
        if not self.potato_leaf.exists():
            self.skipTest(f"Sample image not found: {self.potato_leaf}")

        decision = self.router.route(self.potato_leaf, manual_crop="Auto-Detect (AI Plant ID)")
        self.assertIsInstance(decision, RoutingDecision)
        self.assertTrue(decision.route_to_specialist)
        self.assertEqual(decision.target_crop, "potato")
        self.assertEqual(decision.routing_status, "ROUTED_TO_POTATO_SPECIALIST")
        self.assertGreaterEqual(decision.confidence, 60.0)
        self.assertIn("Potato", decision.message)

    def test_manual_potato_selection_preserved(self):
        """Manual Potato selection preserves specialist routing."""
        if not self.potato_leaf.exists():
            self.skipTest(f"Sample image not found: {self.potato_leaf}")

        decision = self.router.route(self.potato_leaf, manual_crop="Potato")
        self.assertTrue(decision.route_to_specialist)
        self.assertEqual(decision.target_crop, "potato")
        self.assertEqual(decision.routing_status, "ROUTED_TO_POTATO_SPECIALIST")

    def test_unsupported_crop_routing_suppresses_specialist(self):
        """Manual or auto-detected unsupported crop reports UNSUPPORTED_CROP and bypasses specialist."""
        if not self.potato_leaf.exists():
            self.skipTest(f"Sample image not found: {self.potato_leaf}")

        decision = self.router.route(self.potato_leaf, manual_crop="Cotton")
        self.assertFalse(decision.route_to_specialist)
        self.assertEqual(decision.routing_status, "UNSUPPORTED_CROP")
        self.assertEqual(decision.target_crop, "cotton")
        self.assertIn("Cotton", decision.message)
        # No specialist assessed this crop, so confidence must not be fabricated.
        self.assertEqual(decision.confidence, 0.0)
        self.assertIsNotNone(decision.leaf_verification)
        self.assertTrue(decision.leaf_verification["is_leaf"])

    def test_unsupported_manual_crop_runs_leaf_verification(self):
        """Blank, non-leaf, and corrupt inputs are rejected even when a manual unsupported crop is selected."""
        blank_img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        blank_decision = self.router.route(blank_img, manual_crop="Cotton")
        self.assertFalse(blank_decision.route_to_specialist)
        self.assertEqual(blank_decision.routing_status, "REJECTED_BLANK")
        self.assertNotEqual(blank_decision.confidence, 100.0)
        self.assertIsNotNone(blank_decision.leaf_verification)
        self.assertFalse(blank_decision.leaf_verification["is_leaf"])

        non_leaf_img = Image.new("RGB", (224, 224), color=(245, 245, 245))
        draw = ImageDraw.Draw(non_leaf_img)
        draw.rectangle([40, 40, 180, 180], fill=(20, 20, 200))
        non_leaf_decision = self.router.route(non_leaf_img, manual_crop="Cotton")
        self.assertFalse(non_leaf_decision.route_to_specialist)
        self.assertEqual(non_leaf_decision.routing_status, "REJECTED_NON_LEAF")
        self.assertNotEqual(non_leaf_decision.confidence, 100.0)

        corrupt_decision = self.router.route("/path/does/not/exist/image.jpg", manual_crop="Cotton")
        self.assertFalse(corrupt_decision.route_to_specialist)
        self.assertEqual(corrupt_decision.routing_status, "INVALID_INPUT")
        self.assertEqual(corrupt_decision.confidence, 0.0)
        self.assertIsNotNone(corrupt_decision.leaf_verification)
        self.assertFalse(corrupt_decision.leaf_verification["is_leaf"])

    def test_non_leaf_image_rejected(self):
        """Non-leaf blank or corrupt input is rejected before plant identification."""
        white_img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        decision = self.router.route(white_img, manual_crop="Auto-Detect (AI Plant ID)")
        self.assertFalse(decision.route_to_specialist)
        self.assertIn("REJECTED", decision.routing_status)


class TestTreatmentSuppressionForUnsupportedCases(unittest.TestCase):
    """Verify chemical treatments are strictly withheld when plant/specialist is unsupported."""

    def setUp(self):
        self.recommender = PotatoTreatmentRecommender()

    def test_chemical_suppression_on_unsupported_crop(self):
        """Supported crops without a verified treatment registry return VERIFIED_RECOMMENDATION_UNAVAILABLE with zero chemicals."""
        res = self.recommender.recommend_treatment(crop="Rice", disease="Rice___Brown_spot")
        self.assertTrue(res.is_supported)
        self.assertEqual(res.status, "VERIFIED_RECOMMENDATION_UNAVAILABLE")
        self.assertEqual(len(res.chemical_treatments), 0)

    def test_chemical_suppression_on_rejected_status(self):
        """Rejected or inconclusive status strictly returns 0 chemical options."""
        res = self.recommender.recommend_treatment(crop="Potato", disease="Late Blight", is_rejected=True)
        self.assertEqual(len(res.chemical_treatments), 0)

    def test_verified_chemicals_available_only_for_supported_potato(self):
        """Supported Potato disease diagnosis returns verified chemical options."""
        res = self.recommender.recommend_treatment(crop="Potato", disease="Potato___Late_blight")
        self.assertTrue(res.is_supported)
        self.assertEqual(res.status, "VERIFIED_RECOMMENDATION_AVAILABLE")
        self.assertGreater(len(res.chemical_treatments), 0)


class TestCropDoctorEndToEndIntegration(unittest.TestCase):
    """Test full end-to-end CropDoctor pipeline with Stage 1 PlantRouter."""

    def setUp(self):
        self.doctor = CropDoctor()
        self.potato_leaf = ROOT_DIR / "data" / "raw" / "plant_village_subset" / "Potato___healthy" / "real_001.JPG"
        self.lat = 17.8421
        self.lon = 78.1254

    def test_auto_detect_end_to_end_potato_diagnosis(self):
        """Auto-detect mode with Potato leaf diagnoses disease and returns full payload."""
        if not self.potato_leaf.exists():
            self.skipTest(f"Sample image not found: {self.potato_leaf}")

        diag_res = self.doctor.diagnose(
            image_input=self.potato_leaf,
            latitude=self.lat,
            longitude=self.lon,
            crop="Auto-Detect (AI Plant ID)"
        )

        self.assertIsInstance(diag_res, DiagnosticResult)
        self.assertEqual(diag_res.detected_crop, "potato")
        self.assertIn(diag_res.disease, ["Healthy", "Early Blight", "Late Blight"])
        self.assertTrue(0.0 <= diag_res.confidence <= 100.0)
        self.assertIsNotNone(diag_res.routing_decision)
        self.assertEqual(diag_res.routing_decision["routing_status"], "ROUTED_TO_POTATO_SPECIALIST")

    def test_manual_crop_backward_compatibility(self):
        """Calling diagnose(crop='Potato') remains 100% backward compatible."""
        if not self.potato_leaf.exists():
            self.skipTest(f"Sample image not found: {self.potato_leaf}")

        diag_res = self.doctor.diagnose(
            image_input=self.potato_leaf,
            latitude=self.lat,
            longitude=self.lon,
            crop="Potato"
        )

        self.assertIsInstance(diag_res, DiagnosticResult)
        self.assertEqual(diag_res.detected_crop, "potato")
        self.assertTrue(len(diag_res.disease) > 0)

    def test_unsupported_crop_bypasses_disease_specialist(self):
        """Unsupported crop returns safe status and does not force into Potato specialist."""
        if not self.potato_leaf.exists():
            self.skipTest(f"Sample image not found: {self.potato_leaf}")

        diag_res = self.doctor.diagnose(
            image_input=self.potato_leaf,
            latitude=self.lat,
            longitude=self.lon,
            crop="Cotton"
        )

        self.assertIsInstance(diag_res, DiagnosticResult)
        self.assertTrue(diag_res.is_rejected)
        self.assertEqual(diag_res.prediction_status, "UNSUPPORTED_CROP")
        self.assertIn("Cotton", diag_res.disease)


if __name__ == "__main__":
    unittest.main()
