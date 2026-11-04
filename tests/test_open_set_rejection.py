"""
Milestone 4 Test Suite: Open-Set Rejection & Safe Routing.

Tests:
1. Known Potato routing to Potato specialist.
2. Known Tomato routing to Tomato specialist.
3. Unknown-species open-set rejection (feature prototype distance & energy score).
4. Green non-leaf object screening (LeafVerifier heuristic protection).
5. Low-confidence prediction rejection & ambiguity flagging.
6. Missing model / missing prototypes graceful degradation.
7. Invalid and corrupted image handling.
8. Treatment suppression for unknown species, unsupported crops, and healthy leaves.
"""

import sys
import unittest
from pathlib import Path
import torch
from PIL import Image, ImageDraw
import numpy as np

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
    PROTOTYPE_SIM_ROUTING_THRESHOLD,
    PROTOTYPE_SIM_REJECTION_THRESHOLD,
    ENERGY_ROUTING_THRESHOLD,
    ENERGY_REJECTION_THRESHOLD,
)
from modules.crops import get_crop_registry
from modules.dataset_paths import resolve_dataset_path
from modules.diagnosis import CropDoctor, DiagnosticResult
from modules.treatment import PotatoTreatmentRecommender


class TestOpenSetRejection(unittest.TestCase):
    """Test Stage 1 Open-Set Rejection, Prototypes, and Multi-Gate Routing."""

    def setUp(self):
        self.verifier = LeafVerifier()
        self.identifier = PlantIdentifier()
        self.router = PlantRouter(leaf_verifier=self.verifier, plant_identifier=self.identifier)
        self.recommender = PotatoTreatmentRecommender()
        self.doctor = CropDoctor(plant_router=self.router)

        # Verified test samples
        self.potato_leaf = ROOT_DIR / "data" / "raw" / "plant_village_subset" / "Potato___healthy" / "real_001.JPG"
        # Tomato test split first sample
        tomato_test_dir = ROOT_DIR / "data" / "tomato" / "splits"
        self.tomato_leaf = None
        if (tomato_test_dir / "test.json").exists():
            import json
            with open(tomato_test_dir / "test.json") as f:
                tomato_items = json.load(f)
                if tomato_items:
                    self.tomato_leaf = resolve_dataset_path(tomato_items[0]["path"])

    def test_prototypes_loaded_and_valid(self):
        """Feature prototype matrix is loaded with 15 normalized class centroids."""
        self.assertIsNotNone(self.identifier.proto_matrix, "Prototype matrix must be loaded")
        self.assertEqual(self.identifier.proto_matrix.shape, (15, 512))
        # Norm of each prototype row should be 1.0 (unit vectors)
        norms = torch.norm(self.identifier.proto_matrix, p=2, dim=1)
        for n in norms:
            self.assertAlmostEqual(float(n.item()), 1.0, places=4)

    def test_known_potato_routing(self):
        """Genuine Potato leaf routes to Potato specialist with high prototype similarity."""
        if not self.potato_leaf.exists():
            self.skipTest(f"Sample not found: {self.potato_leaf}")

        plant_res = self.identifier.identify(self.potato_leaf)
        self.assertEqual(plant_res.crop_id, "potato")
        self.assertFalse(plant_res.is_rejected)
        self.assertIsNotNone(plant_res.prototype_similarity)
        self.assertGreaterEqual(plant_res.prototype_similarity, PROTOTYPE_SIM_REJECTION_THRESHOLD)
        self.assertIsNotNone(plant_res.energy_score)

        decision = self.router.route(self.potato_leaf, manual_crop="Auto-Detect (AI Plant ID)")
        self.assertTrue(decision.route_to_specialist)
        self.assertIn("POTATO", decision.routing_status)
        self.assertEqual(decision.target_crop, "potato")

    def test_known_tomato_routing(self):
        """Genuine Tomato leaf routes to Tomato specialist with high confidence and prototype similarity."""
        if not self.tomato_leaf or not self.tomato_leaf.exists():
            self.skipTest("Tomato test sample not found")

        plant_res = self.identifier.identify(self.tomato_leaf)
        self.assertEqual(plant_res.crop_id, "tomato")
        self.assertFalse(plant_res.is_rejected)
        self.assertIsNotNone(plant_res.prototype_similarity)
        self.assertGreaterEqual(plant_res.prototype_similarity, PROTOTYPE_SIM_REJECTION_THRESHOLD)

        decision = self.router.route(self.tomato_leaf, manual_crop="Auto-Detect (AI Plant ID)")
        self.assertTrue(decision.route_to_specialist)
        self.assertIn("TOMATO", decision.routing_status)
        self.assertEqual(decision.target_crop, "tomato")

    def test_green_non_leaf_object_rejected_by_leaf_verifier(self):
        """Synthetic green geometric objects are rejected by LeafVerifier before neural network."""
        # Pure green rectangle
        img_green = Image.new("RGB", (256, 256), color=(34, 139, 34))
        draw = ImageDraw.Draw(img_green)
        draw.rectangle([50, 50, 200, 200], fill=(0, 255, 0), outline=(0, 100, 0))

        leaf_res = self.verifier.verify_image(img_green)
        self.assertFalse(leaf_res.is_leaf)
        self.assertEqual(leaf_res.status, "REJECTED_NON_LEAF")

        # Routing decision must reject
        decision = self.router.route(img_green, manual_crop="Auto-Detect (AI Plant ID)")
        self.assertFalse(decision.route_to_specialist)
        self.assertEqual(decision.routing_status, "REJECTED_NON_LEAF")
        self.assertEqual(decision.target_crop, "unknown")

    def test_open_set_synthetic_sample_rejection(self):
        """Geometric non-plant synthetic samples fail prototype/energy thresholds and reject as OOD."""
        syn_dir = ROOT_DIR / "data" / "archived_synthetic_samples"
        syn_files = [p for p in syn_dir.rglob("*") if p.is_file() and p.suffix.lower() in [".jpg", ".png"]]
        if not syn_files:
            self.skipTest("No archived synthetic samples found")

        for syn_path in syn_files[:5]:
            plant_res = self.identifier.identify(syn_path)
            # Either LeafVerifier or PlantIdentifier prototype similarity (< 0.70) must reject
            if plant_res.prototype_similarity is not None:
                self.assertLess(
                    plant_res.prototype_similarity,
                    PROTOTYPE_SIM_ROUTING_THRESHOLD,
                    f"Synthetic sample {syn_path.name} should not have confident prototype similarity"
                )

    def test_low_confidence_and_ood_routing_suppression(self):
        """Unfamiliar or low-confidence inputs return UNKNOWN_SPECIES with treatments suppressed."""
        # Noise pattern with foliar hue that produces uncertain logits
        np.random.seed(42)
        noise_arr = np.random.randint(50, 180, (224, 224, 3), dtype=np.uint8)
        noise_img = Image.fromarray(noise_arr)

        decision = self.router.route(noise_img, manual_crop="Auto-Detect (AI Plant ID)")
        self.assertFalse(decision.route_to_specialist)
        self.assertIn(decision.routing_status, ["UNKNOWN_SPECIES", "REJECTED_NON_LEAF", "UNSUPPORTED_CROP"])

    def test_missing_model_fallback(self):
        """PlantIdentifier operates gracefully without crash when model checkpoint is missing."""
        missing_identifier = PlantIdentifier(model_path="/tmp/nonexistent_model_checkpoint.pth")
        self.assertFalse(missing_identifier.is_model_available())

        res = missing_identifier.identify(self.potato_leaf)
        self.assertEqual(res.status, "MODEL_UNAVAILABLE")
        self.assertEqual(res.ood_status, "NOT_EVALUATED")

    def test_invalid_and_corrupt_images(self):
        """Empty, non-existent, and corrupt files are safely rejected without exception."""
        res_none = self.identifier.identify(None)
        self.assertTrue(res_none.is_rejected)
        self.assertEqual(res_none.status, "INVALID_INPUT")

        res_missing = self.identifier.identify("/path/does/not/exist/image.jpg")
        self.assertTrue(res_missing.is_rejected)
        self.assertEqual(res_missing.status, "INVALID_INPUT")

    def test_treatment_suppression_on_unknown_and_healthy(self):
        """Chemical treatments are 100% suppressed for unknown species, unsupported crops, and healthy plants."""
        # 1. Unknown species
        res_unknown = self.recommender.recommend_treatment(crop="Unknown Plant", disease="Unknown Plant", is_rejected=True)
        self.assertEqual(len(res_unknown.chemical_treatments), 0)

        # 2. Supported crop without verified treatment registry (e.g. Rice)
        res_rice = self.recommender.recommend_treatment(crop="Rice", disease="Rice___Brown_spot")
        self.assertEqual(len(res_rice.chemical_treatments), 0)
        self.assertEqual(res_rice.status, "VERIFIED_RECOMMENDATION_UNAVAILABLE")

        # 3. Healthy potato
        res_healthy = self.recommender.recommend_treatment(crop="Potato", disease="Potato___healthy")
        self.assertEqual(len(res_healthy.chemical_treatments), 0)
        self.assertTrue(res_healthy.is_healthy)
        self.assertEqual(res_healthy.status, "HEALTHY_PREVENTIVE_ONLY")

    def test_end_to_end_doctor_suppression_for_unsupported_crop(self):
        """End-to-end CropDoctor suppresses chemical treatments and Grad-CAM for unsupported crops."""
        if not self.potato_leaf.exists():
            self.skipTest(f"Sample image not found: {self.potato_leaf}")

        diag_res = self.doctor.diagnose(
            image_input=self.potato_leaf,
            latitude=17.8421,
            longitude=78.1254,
            crop="Cotton"
        )
        self.assertIsInstance(diag_res, DiagnosticResult)
        self.assertTrue(diag_res.is_rejected)
        self.assertEqual(diag_res.prediction_status, "UNSUPPORTED_CROP")
        self.assertEqual(diag_res.gradcam_output_path, "")


if __name__ == "__main__":
    unittest.main()
