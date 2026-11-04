import unittest
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.crops import CropRegistry, CropSpec, get_crop_registry
from modules.treatment import (
    PotatoTreatmentRecommender,
    CropTreatmentRecommender,
    TreatmentOption,
    TreatmentRecord,
    TreatmentRegistry,
    get_treatment_registry,
    TreatmentCategory,
)
from config import settings


class TestCropRegistryMultiCropFoundation(unittest.TestCase):
    """Test CropRegistry extensibility, lookups, missing models, and safe defaults."""

    def setUp(self):
        self.registry = CropRegistry()

    def test_registry_lookup_and_metadata(self):
        """Test crop lookup and rich metadata for supported and candidate crops."""
        potato = self.registry.get_crop("Potato")
        self.assertIsNotNone(potato)
        self.assertEqual(potato.crop_id, "potato")
        self.assertEqual(potato.display_name, "Potato")
        self.assertTrue(potato.diagnostic_supported)
        self.assertEqual(potato.healthy_class, "Potato___healthy")
        self.assertIn("Potato___Early_blight", potato.disease_classes)
        self.assertIn("Potato___Late_blight", potato.disease_classes)
        self.assertGreater(len(potato.compatible_risk_models), 0)
        self.assertGreater(len(potato.trusted_knowledge_sources), 0)

        # Candidate crops must report diagnostic_supported=False until their specialist is active
        candidate_crops = ["Cotton", "Mango", "Banana"]
        for crop_name in candidate_crops:
            spec = self.registry.get_crop(crop_name)
            self.assertIsNotNone(spec, f"Expected {crop_name} to be registered in CropRegistry")
            self.assertFalse(spec.diagnostic_supported, f"{crop_name} must have diagnostic_supported=False until milestone active")
            self.assertGreater(len(spec.compatible_risk_models), 0)
            self.assertGreater(len(spec.trusted_knowledge_sources), 0)

        # Catalogued dataset candidate crops represent healthy and disease classes
        for crop_name in ["Rice", "Maize", "Wheat"]:
            spec = self.registry.get_crop(crop_name)
            self.assertIsNotNone(spec.healthy_class)
            self.assertGreater(len(spec.disease_classes), 0)

    def test_list_supported_and_unsupported_crops(self):
        """Test partitioned listing of supported vs candidate crops."""
        supported = self.registry.list_supported_crops()
        unsupported = self.registry.list_unsupported_crops()

        supported_ids = [c.crop_id for c in supported]
        unsupported_ids = [c.crop_id for c in unsupported]

        self.assertIn("potato", supported_ids)
        self.assertIn("tomato", supported_ids)
        self.assertIn("maize", supported_ids)
        self.assertIn("wheat", supported_ids)
        self.assertIn("rice", supported_ids)
        self.assertNotIn("potato", unsupported_ids)
        self.assertNotIn("tomato", unsupported_ids)
        self.assertNotIn("maize", unsupported_ids)
        self.assertNotIn("wheat", unsupported_ids)
        self.assertNotIn("rice", unsupported_ids)

        for candidate in ["cotton", "mango", "banana"]:
            self.assertIn(candidate, unsupported_ids)
            self.assertNotIn(candidate, supported_ids)

    def test_unknown_crops_fail_safely(self):
        """Test that completely unknown crops return None/empty and fail safely."""
        unknown_names = ["dragonfruit", "avocado", "Unknown_Crop_XYZ", ""]
        for name in unknown_names:
            self.assertIsNone(self.registry.get_crop(name))
            self.assertFalse(self.registry.is_crop_supported(name))
            self.assertFalse(self.registry.is_model_available(name))
            self.assertEqual(self.registry.get_compatible_risk_models(name), [])
            self.assertEqual(self.registry.get_trusted_knowledge_sources(name), [])

    def test_specialist_model_availability(self):
        """Test specialist model path checking and file-level existence verification."""
        # Potato and Tomato specialist models are trained and active on disk
        self.assertTrue(self.registry.is_model_available("Potato"))
        self.assertTrue(self.registry.is_model_available("potato"))
        potato_spec = self.registry.get_crop("potato")
        self.assertTrue(potato_spec.is_model_available())
        self.assertIsNotNone(potato_spec.model_path)
        self.assertIsNotNone(potato_spec.model_file_path)
        self.assertTrue(potato_spec.model_file_path.exists())

        self.assertTrue(self.registry.is_model_available("Tomato"))
        self.assertTrue(self.registry.is_model_available("tomato"))
        tomato_spec = self.registry.get_crop("tomato")
        self.assertTrue(tomato_spec.is_model_available())

        self.assertTrue(self.registry.is_model_available("Maize"))
        self.assertTrue(self.registry.get_crop("maize").is_model_available())
        self.assertTrue(self.registry.is_model_available("Wheat"))
        self.assertTrue(self.registry.get_crop("wheat").is_model_available())

        # Rice specialist model is trained and active on disk
        self.assertTrue(self.registry.is_model_available("Rice"))
        self.assertTrue(self.registry.is_model_available("rice"))
        self.assertTrue(self.registry.get_crop("rice").is_model_available())
        self.assertIsNotNone(self.registry.get_crop("rice").model_file_path)
        self.assertTrue(self.registry.get_crop("rice").model_file_path.exists())

        # Cotton, Mango, Banana do NOT have specialist models trained yet
        for crop_id in ["cotton", "mango", "banana"]:
            self.assertFalse(self.registry.is_model_available(crop_id))
            spec = self.registry.get_crop(crop_id)
            self.assertFalse(spec.is_model_available())

    def test_disease_membership_validation(self):
        """Test crop-specific disease membership checks."""
        potato = self.registry.get_crop("Potato")
        self.assertTrue(potato.has_disease("Potato___Early_blight"))
        self.assertTrue(potato.has_disease("Early Blight"))
        self.assertTrue(potato.has_disease("Late Blight"))
        self.assertTrue(potato.has_disease("healthy"))

        # Completely unrelated disease
        self.assertFalse(potato.has_disease("Tomato___Yellow_Leaf_Curl_Virus"))
        self.assertFalse(potato.has_disease("Fusarium_Wilt"))

    def test_crop_spec_to_dict_serialization(self):
        """Test that CropSpec serializes all fields into dictionary cleanly."""
        potato = self.registry.get_crop("Potato")
        data = potato.to_dict()
        self.assertEqual(data["crop_id"], "potato")
        self.assertEqual(data["display_name"], "Potato")
        self.assertTrue(data["diagnostic_supported"])
        self.assertEqual(data["healthy_class"], "Potato___healthy")
        self.assertTrue(data["is_model_available"])
        self.assertIsInstance(data["compatible_risk_models"], list)
        self.assertIsInstance(data["trusted_knowledge_sources"], list)


class TestExtensibleTreatmentSchemaAndSafety(unittest.TestCase):
    """Test extensible TreatmentRecord / TreatmentOption schema, evidence, and safety gating."""

    def test_treatment_record_fields_and_defaults(self):
        """Test extensible record fields: crop, region, safety fields, regulatory verification."""
        opt = TreatmentOption(
            crop="Potato",
            disease_name="Late Blight",
            category=TreatmentCategory.CURATIVE_SYSTEMIC.value,
            active_ingredient="Cymoxanil + Mancozeb",
            formulation="8% + 64% WP",
            application_rate="1.5 - 2.0 kg/ha in 500 L water (0.3 - 0.4%)",
            timing_and_frequency="Apply at initial disease onset; repeat after 7-10 days if humid",
            pre_harvest_interval_days=14,
            re_entry_interval_hours=24,
            ppe_and_safety_precautions=["Wear chemical-resistant gloves", "Protective eyewear"],
            evidence_source_title="ICAR-CPRI Potato Disease Compendium & CIBRC Approved Uses",
            evidence_url_or_ref="https://cibrc.gov.in/major-uses-of-pesticides",
            recommendation_type="chemical",
            applicable_region="India (CIBRC / ICAR)",
            verification_status="VERIFIED",
            cibrc_approved=True,
            regulatory_authority="CIBRC",
            warnings=["Do not apply during high wind to avoid drift.", "Toxic to aquatic life."]
        )

        self.assertEqual(opt.crop, "Potato")
        self.assertEqual(opt.recommendation_type, "chemical")
        self.assertEqual(opt.applicable_region, "India (CIBRC / ICAR)")
        self.assertTrue(opt.cibrc_approved)
        self.assertEqual(opt.regulatory_authority, "CIBRC")
        self.assertTrue(opt.is_verified())
        self.assertTrue(opt.is_chemical())
        self.assertTrue(opt.has_complete_safety_profile())

        # Test dictionary serialization
        d = opt.to_dict()
        self.assertEqual(d["crop"], "Potato")
        self.assertEqual(d["recommendation_type"], "chemical")
        self.assertEqual(d["applicable_region"], "India (CIBRC / ICAR)")
        self.assertEqual(d["verification_status"], "VERIFIED")
        self.assertTrue(d["cibrc_approved"])
        self.assertEqual(d["regulatory_authority"], "CIBRC")
        self.assertIn("drift", d["warnings"][0])

    def test_unverified_treatment_cannot_claim_approved_chemical_advice(self):
        """Test that unverified records cannot be masqueraded as verified chemical advice."""
        unverified_opt = TreatmentOption(
            crop="Tomato",
            disease_name="Tomato Late Blight",
            category=TreatmentCategory.CURATIVE_SYSTEMIC.value,
            active_ingredient="Hypothetical Fungicide X",
            formulation="50% SC",
            application_rate="1.0 L/ha",
            timing_and_frequency="Once",
            pre_harvest_interval_days=None,  # Missing safety data
            re_entry_interval_hours=None,
            ppe_and_safety_precautions=[],
            evidence_source_title="Unverified blog post",
            evidence_url_or_ref="http://example.com/blog",
            recommendation_type="chemical",
            applicable_region="Unknown",
            verification_status="UNVERIFIED",
            cibrc_approved=False,
            regulatory_authority="NONE",
            warnings=[]
        )

        self.assertFalse(unverified_opt.is_verified())
        self.assertFalse(unverified_opt.cibrc_approved)
        self.assertFalse(unverified_opt.has_complete_safety_profile())

    def test_treatment_registry_indexing_and_isolation(self):
        """Test TreatmentRegistry indexes verified treatments by crop and disease."""
        registry = TreatmentRegistry()
        potato_lb_treatments = registry.get_treatments("Potato", "Late Blight", verified_only=True)
        self.assertGreater(len(potato_lb_treatments), 0)

        # All returned treatments must be verified
        for t in potato_lb_treatments:
            self.assertTrue(t.is_verified())
            self.assertEqual(t.crop, "Potato")
            self.assertTrue(t.cibrc_approved)

        # Tomato has verified treatments in Milestone 3
        tomato_treatments = registry.get_treatments("Tomato", "Early Blight", verified_only=True)
        self.assertGreater(len(tomato_treatments), 0)

        # Candidate unsupported crops have no registered treatments
        rice_treatments = registry.get_treatments("Rice", "Blast", verified_only=True)
        self.assertEqual(len(rice_treatments), 0)

    def test_treatment_record_type_alias(self):
        """Verify TreatmentRecord is exported as alias for TreatmentOption."""
        self.assertIs(TreatmentRecord, TreatmentOption)


class TestRecommenderGatingAndBackwardCompatibility(unittest.TestCase):
    """Test Potato backward compatibility and safe rejection of unsupported crops and diseases."""

    def setUp(self):
        self.recommender = CropTreatmentRecommender()

    def test_potato_early_blight_backward_compatibility(self):
        """Existing Potato Early Blight recommendation remains 100% functional and verified."""
        res = self.recommender.recommend_treatment("Potato", "Potato___Early_blight")
        self.assertTrue(res.is_supported)
        self.assertFalse(res.is_healthy)
        self.assertEqual(res.status, "VERIFIED_RECOMMENDATION_AVAILABLE")
        self.assertGreater(len(res.chemical_treatments), 0)

        # Verify active ingredients
        active_ingredients = [t.active_ingredient for t in res.chemical_treatments]
        self.assertIn("Mancozeb", active_ingredients)
        self.assertIn("Chlorothalonil", active_ingredients)

        # All options have crop='Potato', is_verified()=True, and cibrc_approved=True
        for opt in res.chemical_treatments:
            self.assertEqual(opt.crop, "Potato")
            self.assertTrue(opt.cibrc_approved)
            self.assertTrue(opt.is_verified())
            self.assertIn("Verified", opt.verification_status)

    def test_potato_late_blight_backward_compatibility(self):
        """Existing Potato Late Blight recommendation returns protectants and curatives."""
        res = self.recommender.recommend_treatment("Potato", "Potato___Late_blight")
        self.assertTrue(res.is_supported)
        self.assertFalse(res.is_healthy)
        self.assertEqual(res.status, "VERIFIED_RECOMMENDATION_AVAILABLE")
        self.assertGreater(len(res.chemical_treatments), 0)

        active_ingredients = [t.active_ingredient for t in res.chemical_treatments]
        self.assertIn("Cymoxanil + Mancozeb", active_ingredients)
        self.assertIn("Metalaxyl + Mancozeb", active_ingredients)

    def test_potato_healthy_backward_compatibility(self):
        """Healthy prediction strictly returns zero chemical advice."""
        res = self.recommender.recommend_treatment("Potato", "Potato___healthy")
        self.assertTrue(res.is_supported)
        self.assertTrue(res.is_healthy)
        self.assertEqual(res.status, "HEALTHY_PREVENTIVE_ONLY")
        self.assertEqual(len(res.chemical_treatments), 0)
        self.assertGreater(len(res.non_chemical_control_measures), 0)

    def test_unsupported_crop_safely_rejected(self):
        """Candidate unsupported crops are rejected safely with status UNSUPPORTED_CROP and 0 chemicals."""
        for crop in ["Cotton", "Mango", "Banana", "Dragonfruit"]:
            res = self.recommender.recommend_treatment(crop, "Early Blight")
            self.assertFalse(res.is_supported)
            self.assertEqual(res.status, "UNSUPPORTED_CROP")
            self.assertEqual(len(res.chemical_treatments), 0)
            self.assertIn(crop, res.summary_message)

    def test_supported_crop_without_verified_treatment_registry_withholds_chemicals(self):
        """Maize/Wheat diagnosis is supported, but no crop-specific chemical registry is invented."""
        for crop in ["Maize", "Wheat"]:
            res = self.recommender.recommend_treatment(crop, f"{crop}___Some_Disease")
            self.assertTrue(res.is_supported)
            self.assertFalse(res.is_healthy)
            self.assertEqual(res.status, "VERIFIED_RECOMMENDATION_UNAVAILABLE")
            self.assertEqual(len(res.chemical_treatments), 0)
            self.assertTrue(any("No verified chemical treatment registry" in gap for gap in res.evidence_gaps))

    def test_unsupported_disease_safely_rejected(self):
        """Unsupported potato diseases report UNSUPPORTED_DISEASE and 0 chemicals."""
        res = self.recommender.recommend_treatment("Potato", "Potato___Bacterial_Wilt_XYZ")
        self.assertFalse(res.is_supported)
        self.assertEqual(res.status, "UNSUPPORTED_DISEASE")
        self.assertEqual(len(res.chemical_treatments), 0)
        self.assertIn("No verified treatment recommendation", res.summary_message)


class TestSyntheticDatasetScriptSafety(unittest.TestCase):
    """Test that scripts/create_sample_dataset.py cannot overwrite active training dataset."""

    def test_synthetic_script_target_path_isolated(self):
        """Verify the script targets archived synthetic folder, NOT data/raw/plant_village_subset."""
        script_path = ROOT_DIR / "scripts" / "create_sample_dataset.py"
        self.assertTrue(script_path.exists())

        content = script_path.read_text()
        # Must not set RAW_DATA_DIR / 'plant_village_subset' as target
        self.assertNotIn(
            'output_dir = settings.RAW_DATA_DIR / "plant_village_subset"',
            content,
            "Synthetic generator must not target the active plant_village_subset dataset"
        )
        self.assertIn("archived_synthetic_samples", content)

    def test_active_dataset_clean_and_intact(self):
        """Verify the active dataset directory contains only the 300 real images."""
        active_dataset_dir = settings.RAW_DATA_DIR / "plant_village_subset"
        self.assertTrue(active_dataset_dir.exists())

        for class_name in ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]:
            class_dir = active_dataset_dir / class_name
            self.assertTrue(class_dir.exists())
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG"))
            self.assertEqual(len(images), 100, f"Class {class_name} must contain exactly 100 real images")
            # No sample_*.jpg files
            synthetic_files = [f for f in images if f.name.startswith("sample_")]
            self.assertEqual(len(synthetic_files), 0, f"No synthetic files allowed in {class_name}")


if __name__ == "__main__":
    unittest.main()
