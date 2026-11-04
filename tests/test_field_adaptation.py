"""
Unit and regression tests for Milestone 6C: Field-Domain Adaptation Feasibility.
Validates:
1. Baseline model immutability (cryptographic hashes match exactly).
2. Dynamic switchability between baseline and experimental field-adapted models.
3. DiseasePredictor inference integrity under both configurations.
4. Treatment suppression compliance under both configurations.
"""

import unittest
import hashlib
from pathlib import Path
import torch
from PIL import Image

from config import settings
from modules.disease_predictor import DiseasePredictor
from modules.treatment import PotatoTreatmentRecommender

# Baseline immutable SHA-256 hashes recorded at start of Milestone 6C
BASELINE_POTATO_SHA256 = "305fff18457429a3b5a0f0c7f02d2cdde894cf4db296e2964b80b7cc8f1d5d14"
BASELINE_TOMATO_SHA256 = "ba5d092ef0294a25c558dad2508bf420219c5da89910886a1dde1457d894f154"


class TestFieldAdaptationFeasibility(unittest.TestCase):
    """Regression and safety tests for Milestone 6C field adaptation."""

    def test_baseline_checkpoints_immutability(self):
        """Baseline model checkpoints must remain bit-for-bit identical to original hashes."""
        potato_path = settings.BASE_DIR / "models" / "disease" / "potato_disease_model.pth"
        tomato_path = settings.BASE_DIR / "models" / "disease" / "tomato_disease_model.pth"

        self.assertTrue(potato_path.exists(), "Baseline potato checkpoint missing.")
        self.assertTrue(tomato_path.exists(), "Baseline tomato checkpoint missing.")

        pot_sha256 = hashlib.sha256(potato_path.read_bytes()).hexdigest()
        tom_sha256 = hashlib.sha256(tomato_path.read_bytes()).hexdigest()

        self.assertEqual(pot_sha256, BASELINE_POTATO_SHA256, "Baseline potato checkpoint was unexpectedly modified!")
        self.assertEqual(tom_sha256, BASELINE_TOMATO_SHA256, "Baseline tomato checkpoint was unexpectedly modified!")

    def test_experimental_checkpoints_exist_and_distinct(self):
        """Experimental field-adapted models must exist under separate filenames and have distinct weights."""
        exp_potato = settings.BASE_DIR / "models" / "disease" / "potato_field_adapted_model.pth"
        exp_tomato = settings.BASE_DIR / "models" / "disease" / "tomato_field_adapted_model.pth"

        self.assertTrue(exp_potato.exists(), "Experimental potato checkpoint missing.")
        self.assertTrue(exp_tomato.exists(), "Experimental tomato checkpoint missing.")

        pot_exp_sha256 = hashlib.sha256(exp_potato.read_bytes()).hexdigest()
        tom_exp_sha256 = hashlib.sha256(exp_tomato.read_bytes()).hexdigest()

        self.assertNotEqual(pot_exp_sha256, BASELINE_POTATO_SHA256, "Experimental potato matches baseline unexpectedly.")
        self.assertNotEqual(tom_exp_sha256, BASELINE_TOMATO_SHA256, "Experimental tomato matches baseline unexpectedly.")

    def test_predictor_model_switching(self):
        """DiseasePredictor correctly switches between baseline and experimental models via argument."""
        # Baseline mode
        pred_base = DiseasePredictor(device=torch.device("cpu"), use_field_adapted_models=False)
        self.assertFalse(pred_base.use_field_adapted_models)
        self.assertEqual(pred_base.model_path.name, "potato_disease_model.pth")
        base_pot = pred_base.get_model("Potato")
        base_tom = pred_base.get_model("Tomato")
        self.assertIsNotNone(base_pot)
        self.assertIsNotNone(base_tom)

        # Experimental field-adapted mode
        pred_exp = DiseasePredictor(device=torch.device("cpu"), use_field_adapted_models=True)
        self.assertTrue(pred_exp.use_field_adapted_models)
        self.assertEqual(pred_exp.model_path.name, "potato_field_adapted_model.pth")
        exp_pot = pred_exp.get_model("Potato")
        exp_tom = pred_exp.get_model("Tomato")
        self.assertIsNotNone(exp_pot)
        self.assertIsNotNone(exp_tom)

        # Verify weights in memory are distinct
        base_pot_w = list(base_pot.parameters())[0]
        exp_pot_w = list(exp_pot.parameters())[0]
        # At layer4/fc they must differ
        base_pot_fc = list(base_pot.fc.parameters())[0]
        exp_pot_fc = list(exp_pot.fc.parameters())[0]
        self.assertFalse(torch.allclose(base_pot_fc, exp_pot_fc))

    def test_both_configurations_inference_and_safety(self):
        """Inference and treatment safety run cleanly under both configurations."""
        sample_path = next(Path("data/raw/plant_village_subset/Potato___Early_blight").glob("*.JPG"), None)
        if not sample_path or not sample_path.exists():
            return

        img = Image.open(sample_path).convert("RGB")
        recommender = PotatoTreatmentRecommender()

        for use_adapted in [False, True]:
            predictor = DiseasePredictor(device=torch.device("cpu"), use_field_adapted_models=use_adapted)
            diag = predictor.predict(img, crop="Potato")

            self.assertIn("predicted_disease", diag)
            self.assertIn("confidence", diag)
            self.assertTrue(diag["is_supported"])

            # Treatment suppression test: low confidence must suppress chemicals
            treat_res = recommender.recommend_treatment(
                crop="Potato",
                disease=diag["predicted_disease"],
                is_low_confidence=True,
                risk_level="LOW"
            )
            self.assertEqual(len(treat_res.chemical_treatments), 0, "Chemical treatments not suppressed!")


if __name__ == "__main__":
    unittest.main()
