import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
import json
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.disease_predictor import DiseasePredictor, TOMATO_CLASSES, TOMATO_FRIENDLY_NAMES
from modules.crops import get_crop_registry
from modules.dataset_paths import resolve_dataset_path
from modules.identification.router import PlantRouter, RoutingDecision
from modules.treatment import PotatoTreatmentRecommender, TreatmentRegistry
from modules.risk import RiskEngine, RiskAssessment
from modules.diagnosis import CropDoctor, DiagnosticResult


class TestTomatoSpecialist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tomato_model_path = ROOT_DIR / "models" / "disease" / "tomato_disease_model.pth"
        cls.splits_test_path = ROOT_DIR / "data" / "tomato" / "splits" / "test.json"
        
        # Load a verified sample path from test split if available
        cls.sample_image_path = None
        if cls.splits_test_path.exists():
            try:
                with open(cls.splits_test_path, "r") as f:
                    test_samples = json.load(f)
                    if test_samples and len(test_samples) > 0:
                        candidate_path = resolve_dataset_path(test_samples[0]["path"])
                        if candidate_path.exists():
                            cls.sample_image_path = str(candidate_path)
            except Exception:
                cls.sample_image_path = None

        if cls.sample_image_path is None:
            # Fallback to synthetic test PIL image
            cls.test_img = Image.new("RGB", (256, 256), color=(40, 150, 40))
        else:
            cls.test_img = Image.open(cls.sample_image_path).convert("RGB")

    def test_1_tomato_model_checkpoint_and_classes(self):
        """Test 1: Verify Tomato disease specialist checkpoint on disk, weight shapes, and 10 classes."""
        self.assertTrue(self.tomato_model_path.exists(), f"Checkpoint missing: {self.tomato_model_path}")
        checkpoint = torch.load(self.tomato_model_path, map_location="cpu")

        # Checkpoint is saved as a raw state_dict (OrderedDict / dict of tensors)
        self.assertIsInstance(checkpoint, dict)
        self.assertIn("fc.weight", checkpoint)
        self.assertIn("fc.bias", checkpoint)

        # Confirm final layer output dimension matches 10 classes
        fc_weight = checkpoint["fc.weight"]
        self.assertEqual(fc_weight.shape[0], 10, f"Expected 10 output classes, got {fc_weight.shape[0]}")
        self.assertEqual(fc_weight.shape[1], 512, f"Expected ResNet-18 feature dim 512, got {fc_weight.shape[1]}")

        # Confirm layer4 is present (ResNet-18 architecture)
        layer4_keys = [k for k in checkpoint if k.startswith("layer4")]
        self.assertGreater(len(layer4_keys), 0, "ResNet-18 layer4 keys missing from checkpoint")

        # Confirm 10 verified classes are registered in CropSpec registry
        from modules.crops import get_crop_registry
        registry = get_crop_registry()
        tomato_spec = registry.get_crop("Tomato")
        self.assertIsNotNone(tomato_spec)
        self.assertEqual(len(tomato_spec.supported_diseases), 10)

        expected_classes = [
            "Tomato___Bacterial_spot",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite",
            "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato___Tomato_mosaic_virus",
            "Tomato___healthy"
        ]
        self.assertEqual(tomato_spec.supported_diseases, expected_classes)
        self.assertEqual(TOMATO_CLASSES, expected_classes)

    def test_2_tomato_disease_predictor_inference(self):
        """Test 2: DiseasePredictor loads Tomato model and produces valid predictions."""
        predictor = DiseasePredictor()
        tomato_model = predictor.get_model("Tomato")
        self.assertIsNotNone(tomato_model)

        res = predictor.predict(self.test_img, crop="Tomato")
        self.assertEqual(res["crop"], "Tomato")
        self.assertTrue(res["is_supported"])
        self.assertIn(res["predicted_disease"], TOMATO_CLASSES)
        self.assertIn(res["friendly_name"], TOMATO_FRIENDLY_NAMES.values())
        self.assertGreater(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 100.0)
        self.assertEqual(len(res["probabilities"]), 10)
        self.assertAlmostEqual(sum(res["probabilities"].values()), 100.0, places=1)

    def test_3_missing_checkpoint_safe_failure(self):
        """Test 3: Missing specialist checkpoint returns MISSING_CHECKPOINT and does not crash."""
        predictor = DiseasePredictor()
        
        # Test nonexistent crop model path
        with patch.object(predictor, "get_model", return_value=None):
            res = predictor.predict(self.test_img, crop="Tomato")
            self.assertFalse(res["is_supported"])
            self.assertTrue(res["is_rejected"])
            self.assertEqual(res["prediction_status"], "MISSING_CHECKPOINT")
            self.assertIn("Checkpoint Missing", res["friendly_name"])

    def test_4_plant_router_tomato_manual_and_auto(self):
        """Test 4: PlantRouter correctly routes Tomato manually and via auto-detection."""
        router = PlantRouter()

        # Manual routing to Tomato specialist
        decision_manual = router.route(self.test_img, manual_crop="Tomato")
        self.assertTrue(decision_manual.route_to_specialist)
        self.assertEqual(decision_manual.target_crop, "tomato")
        self.assertEqual(decision_manual.routing_status, "ROUTED_TO_TOMATO_SPECIALIST")

        from modules.identification.plant_identifier import PlantIdentificationResult

        # Auto-detect confident Tomato
        with patch.object(router.plant_identifier, "identify") as mock_identify:
            mock_identify.return_value = PlantIdentificationResult(
                crop_id="tomato",
                display_name="Tomato",
                confidence=92.5,
                probabilities={"tomato": 92.5},
                is_supported=True,
                is_ambiguous=False,
                is_rejected=False,
                status="CONFIDENT_IDENTIFICATION",
                warning=None
            )
            decision_auto_confident = router.route(self.test_img, manual_crop="Auto-Detect (AI Plant ID)")
            self.assertTrue(decision_auto_confident.route_to_specialist)
            self.assertEqual(decision_auto_confident.routing_status, "ROUTED_TO_TOMATO_SPECIALIST")
            self.assertIsNone(decision_auto_confident.warning)

        # Auto-detect marginal/uncertain Tomato
        with patch.object(router.plant_identifier, "identify") as mock_identify:
            mock_identify.return_value = PlantIdentificationResult(
                crop_id="tomato",
                display_name="Tomato",
                confidence=52.0,
                probabilities={"tomato": 52.0},
                is_supported=True,
                is_ambiguous=False,
                is_rejected=False,
                status="AMBIGUOUS_SPECIES",
                warning="Marginal certainty"
            )
            decision_auto_uncertain = router.route(self.test_img, manual_crop="Auto-Detect (AI Plant ID)")
            self.assertTrue(decision_auto_uncertain.route_to_specialist)
            self.assertEqual(decision_auto_uncertain.routing_status, "ROUTED_TO_TOMATO_SPECIALIST_UNCERTAIN")
            self.assertIsNotNone(decision_auto_uncertain.warning)

    def test_5_healthy_tomato_receives_zero_chemicals(self):
        """Test 5: Healthy Tomato foliage strictly receives zero chemical prescriptions."""
        recommender = PotatoTreatmentRecommender()
        res = recommender.recommend_treatment("Tomato", "Tomato___healthy")

        self.assertTrue(res.is_supported)
        self.assertTrue(res.is_healthy)
        self.assertEqual(res.status, "HEALTHY_PREVENTIVE_ONLY")
        self.assertEqual(len(res.chemical_treatments), 0)
        self.assertGreater(len(res.non_chemical_control_measures), 0)
        self.assertGreater(len(res.safety_and_environmental_precautions), 0)
        self.assertGreater(len(res.evidence_sources), 0)
        self.assertIn("No chemical fungicide or pesticide treatment is required", res.summary_message)

    def test_6_tomato_disease_treatment_grounding(self):
        """Test 6: Verified CIBRC and ICAR-IIVR treatments for each Tomato disease."""
        recommender = PotatoTreatmentRecommender()

        # Late Blight
        res_lb = recommender.recommend_treatment("Tomato", "Tomato___Late_blight")
        self.assertTrue(res_lb.is_supported)
        self.assertFalse(res_lb.is_healthy)
        self.assertEqual(res_lb.status, "VERIFIED_RECOMMENDATION_AVAILABLE")
        lb_actives = [t.active_ingredient for t in res_lb.chemical_treatments]
        self.assertIn("Mancozeb", lb_actives)
        self.assertIn("Metalaxyl + Mancozeb", lb_actives)

        # Early Blight
        res_eb = recommender.recommend_treatment("Tomato", "Tomato___Early_blight")
        eb_actives = [t.active_ingredient for t in res_eb.chemical_treatments]
        self.assertIn("Mancozeb", eb_actives)
        self.assertIn("Difenoconazole", eb_actives)

        # Spider Mites (strictly acaricides / botanicals, not fungicides)
        res_sm = recommender.recommend_treatment("Tomato", "Tomato___Spider_mites Two-spotted_spider_mite")
        sm_actives = [t.active_ingredient for t in res_sm.chemical_treatments]
        self.assertIn("Spiromesifen", sm_actives)
        self.assertIn("Wettable Sulphur", sm_actives)
        self.assertNotIn("Metalaxyl + Mancozeb", sm_actives)

        # TYLCV (virus: strictly whitefly vector management, disclaims cure)
        res_tylcv = recommender.recommend_treatment("Tomato", "Tomato___Tomato_Yellow_Leaf_Curl_Virus")
        tylcv_actives = [t.active_ingredient for t in res_tylcv.chemical_treatments]
        self.assertTrue(any("Imidacloprid" in a for a in tylcv_actives))
        self.assertIn("Viral disease cannot be cured by chemical fungicides", res_tylcv.chemical_treatments[0].application_directions)

        # Tomato Mosaic Virus (mechanical transmission: skimmed milk barrier only, 0 chemical fungicides)
        res_tomv = recommender.recommend_treatment("Tomato", "Tomato___Tomato_mosaic_virus")
        self.assertEqual(len(res_tomv.chemical_treatments), 0)
        self.assertGreater(len(res_tomv.non_chemical_control_measures), 0)
        self.assertIn("cannot be cured by chemical pesticides", res_tomv.summary_message)

    def test_7_uncertain_and_rejected_predictions_withhold_chemicals(self):
        """Test 7: Uncertain (<60%) or rejected (<40%) predictions strictly withhold chemicals."""
        recommender = PotatoTreatmentRecommender()

        res_uncertain = recommender.recommend_treatment(
            "Tomato", "Tomato___Late_blight", is_low_confidence=True
        )
        self.assertEqual(len(res_uncertain.chemical_treatments), 0)
        self.assertEqual(res_uncertain.status, "UNCERTAIN_DIAGNOSIS_WARNING")

        res_rejected = recommender.recommend_treatment(
            "Tomato", "Tomato___Late_blight", is_rejected=True
        )
        self.assertEqual(len(res_rejected.chemical_treatments), 0)
        self.assertEqual(res_rejected.status, "PREDICTION_REJECTED")

    def test_8_broad_plant_compatibility_unsupported_crops(self):
        """Test 8: Broad-plant registry handles crops outside PlantVillage without forcing into Tomato or Potato."""
        registry = get_crop_registry()
        
        # Candidate crops outside PlantVillage
        for crop_name in ["Cotton", "Mango", "Banana"]:
            self.assertFalse(registry.is_crop_supported(crop_name), f"{crop_name} must report is_supported=False")
            spec = registry.get_crop(crop_name)
            self.assertIsNotNone(spec)
            self.assertFalse(spec.diagnostic_supported)
            self.assertFalse(spec.is_model_available())

        # Treatment recommender rejects candidate crops
        recommender = PotatoTreatmentRecommender()
        for crop_name in ["Cotton", "Mango", "Banana"]:
            res = recommender.recommend_treatment(crop_name, "Blast")
            self.assertFalse(res.is_supported)
            self.assertEqual(res.status, "UNSUPPORTED_CROP")
            self.assertEqual(len(res.chemical_treatments), 0)

    def test_9_epidemiological_risk_engine_tomato_rules(self):
        """Test 9: Risk engine evaluates Tomato diseases with appropriate biological compatibility rules."""
        engine = RiskEngine()
        hourly_temps = [20.0] * 24
        hourly_rh = [92.0] * 24
        weather_humid = {
            "hourly": {
                "temperatures_c": hourly_temps,
                "relative_humidities_pct": hourly_rh,
            },
            "summary_24h": {
                "min_temp_c": min(hourly_temps),
                "max_temp_c": max(hourly_temps),
                "avg_temp_c": sum(hourly_temps) / len(hourly_temps),
                "min_humidity_pct": min(hourly_rh),
                "max_humidity_pct": max(hourly_rh),
                "avg_humidity_pct": sum(hourly_rh) / len(hourly_rh),
            }
        }

        # Late Blight: Wallin DSV applied with explicit compatibility rule
        pred_lb = {
            "predicted_disease": "Tomato___Late_blight",
            "friendly_name": "Tomato Late Blight",
            "confidence": 95.0
        }
        assessment_lb = engine.assess_risk(pred_lb, weather_humid)
        self.assertEqual(assessment_lb.disease, "Tomato Late Blight")
        self.assertTrue(any("Phytophthora infestans" in f for f in assessment_lb.factors))

        # Early Blight: P-Day thermal model applied with explicit compatibility rule
        pred_eb = {
            "predicted_disease": "Tomato___Early_blight",
            "friendly_name": "Tomato Early Blight",
            "confidence": 90.0
        }
        assessment_eb = engine.assess_risk(pred_eb, weather_humid)
        self.assertEqual(assessment_eb.disease, "Tomato Early Blight")
        self.assertTrue(any("Alternaria solani" in f for f in assessment_eb.factors))

        # Spider Mites: hot/dry conditions (>28C, low RH); explicitly rejects fungal wetness models
        hot_dry_temps = [33.0] * 24
        hot_dry_rh = [35.0] * 24
        weather_hot_dry = {
            "hourly": {
                "temperatures_c": hot_dry_temps,
                "relative_humidities_pct": hot_dry_rh,
            },
            "summary_24h": {
                "min_temp_c": min(hot_dry_temps),
                "max_temp_c": max(hot_dry_temps),
                "avg_temp_c": sum(hot_dry_temps) / len(hot_dry_temps),
                "min_humidity_pct": min(hot_dry_rh),
                "max_humidity_pct": max(hot_dry_rh),
                "avg_humidity_pct": sum(hot_dry_rh) / len(hot_dry_rh),
            }
        }
        pred_sm = {
            "predicted_disease": "Tomato___Spider_mites Two-spotted_spider_mite",
            "friendly_name": "Spider Mites (Two-Spotted)",
            "confidence": 88.0
        }
        assessment_sm = engine.assess_risk(pred_sm, weather_hot_dry)
        self.assertEqual(assessment_sm.risk_level, "HIGH")
        self.assertTrue(any("Tetranychus urticae" in f for f in assessment_sm.factors))
        self.assertTrue(any("fungal leaf-wetness formulas" in f.lower() for f in assessment_sm.factors))

        # TYLCV Whitefly Vector risk
        tylcv_weather = {
            "hourly": {
                "temperatures_c": [29.0] * 24,
                "relative_humidities_pct": [50.0] * 24,
            },
            "summary_24h": {
                "min_temp_c": 29.0,
                "max_temp_c": 29.0,
                "avg_temp_c": 29.0,
                "min_humidity_pct": 50.0,
                "max_humidity_pct": 50.0,
                "avg_humidity_pct": 50.0,
            }
        }
        pred_tylcv = {
            "predicted_disease": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "friendly_name": "Tomato Yellow Leaf Curl Virus",
            "confidence": 85.0
        }
        assessment_tylcv = engine.assess_risk(pred_tylcv, tylcv_weather)
        self.assertTrue(any("Bemisia tabaci" in f for f in assessment_tylcv.factors))

    def test_10_gradcam_explainer_tomato(self):
        """Test 10: Grad-CAM generates visual heatmap for Tomato specialist."""
        from modules.explainability.gradcam import GradCAM
        explainer = GradCAM()
        result = explainer.generate_explanation(self.test_img, crop="Tomato")

        self.assertIsNotNone(result)
        self.assertIn("output_path", result)
        self.assertIn("heatmap_np", result)
        self.assertIn("predicted_disease", result)
        self.assertIn("friendly_name", result)
        self.assertIn("confidence", result)
        # Verify heatmap numpy array has spatial content
        import numpy as np
        heatmap = result["heatmap_np"]
        self.assertIsNotNone(heatmap)
        self.assertEqual(len(heatmap.shape), 2)
        # Target class should be a Tomato class
        predicted = result["predicted_disease"]
        self.assertTrue(
            predicted.startswith("Tomato") or "tomato" in predicted.lower(),
            f"Expected a Tomato class, got: {predicted}"
        )

    def test_11_potato_backward_compatibility_preserved(self):
        """Test 11: Potato specialist functionality remains 100% functional and unmodified."""
        predictor = DiseasePredictor()
        potato_model = predictor.get_model("Potato")
        self.assertIsNotNone(potato_model)
        self.assertEqual(len(predictor.classes), 3)

        router = PlantRouter()
        decision = router.route(self.test_img, manual_crop="Potato")
        self.assertTrue(decision.route_to_specialist)
        self.assertEqual(decision.routing_status, "ROUTED_TO_POTATO_SPECIALIST")

        recommender = PotatoTreatmentRecommender()
        res_potato = recommender.recommend_treatment("Potato", "Potato___Late_blight")
        self.assertTrue(res_potato.is_supported)
        self.assertEqual(res_potato.status, "VERIFIED_RECOMMENDATION_AVAILABLE")


if __name__ == "__main__":
    unittest.main()
