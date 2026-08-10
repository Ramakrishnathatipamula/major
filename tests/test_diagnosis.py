import unittest
from pathlib import Path
from modules.diagnosis import CropDoctor, DiagnosticResult

class TestDiagnosticPipeline(unittest.TestCase):

    def setUp(self):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.sample_image = self.root_dir / "data" / "raw" / "plant_village_subset" / "Potato___Late_blight" / "real_086.JPG"
        self.lat = 28.6139
        self.lon = 77.2090

    def test_pipeline_integration(self):
        if not self.sample_image.exists():
            self.skipTest(f"Sample image not found at {self.sample_image}")

        crop_doctor = CropDoctor()
        result = crop_doctor.diagnose(
            image_input=self.sample_image,
            latitude=self.lat,
            longitude=self.lon
        )

        self.assertIsInstance(result, DiagnosticResult)
        self.assertTrue(len(result.disease) > 0)
        self.assertTrue(0.0 <= result.confidence <= 100.0)
        self.assertTrue(len(result.class_probabilities) > 0)

        # Verify Grad-CAM file output exists
        heatmap_file = Path(result.gradcam_output_path)
        self.assertTrue(heatmap_file.exists(), f"Grad-CAM file does not exist: {heatmap_file}")

        # Verify weather data integration
        self.assertIn("current", result.weather_summary)
        self.assertIn("summary_24h", result.weather_summary)

        # Verify risk assessment metrics
        self.assertIn(result.risk_level, ["LOW", "MODERATE", "HIGH"])
        self.assertTrue(0.0 <= result.risk_index <= 100.0)
        self.assertTrue(len(result.risk_factors) > 0)
        self.assertTrue(len(result.explanation) > 0)

        # Verify disclaimers
        self.assertIn("risk_index_type", result.disclaimer)
        self.assertIn("pesticide_notice", result.disclaimer)
        self.assertIn("scientific_limitation", result.disclaimer)

if __name__ == "__main__":
    unittest.main()
