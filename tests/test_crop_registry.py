import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.crops import CropRegistry, CropSpec, get_crop_registry
from modules.disease_predictor import DiseasePredictor
from modules.diagnosis import CropDoctor, DiagnosticResult


from modules.weather import WeatherData, CurrentWeather, HourlyForecast


class TestMultiCropArchitecture(unittest.TestCase):

    def setUp(self):
        self.registry = CropRegistry()

    def test_default_crop_registrations(self):
        """Test default crop registrations in CropRegistry."""
        crops = self.registry.list_crops()
        crop_names = [c.display_name for c in crops]
        
        self.assertIn("Potato", crop_names)
        self.assertIn("Tomato", crop_names)
        self.assertIn("Rice", crop_names)
        self.assertIn("Wheat", crop_names)
        self.assertIn("Cotton", crop_names)
        self.assertIn("Maize", crop_names)

    def test_crop_support_status(self):
        """Test that crops with trained specialists are supported and candidates are not."""
        self.assertTrue(self.registry.is_crop_supported("Potato"))
        self.assertTrue(self.registry.is_crop_supported("potato"))
        self.assertTrue(self.registry.is_crop_supported("Tomato"))
        self.assertTrue(self.registry.is_crop_supported("tomato"))
        self.assertTrue(self.registry.is_crop_supported("Maize"))
        self.assertTrue(self.registry.is_crop_supported("maize"))
        self.assertTrue(self.registry.is_crop_supported("Wheat"))
        self.assertTrue(self.registry.is_crop_supported("wheat"))
        self.assertTrue(self.registry.is_crop_supported("Rice"))
        self.assertTrue(self.registry.is_crop_supported("rice"))

        self.assertFalse(self.registry.is_crop_supported("Cotton"))
        self.assertFalse(self.registry.is_crop_supported("Mango"))
        self.assertFalse(self.registry.is_crop_supported("Banana"))

    def test_unsupported_crop_prediction_graceful_response(self):
        """Test that DiseasePredictor gracefully handles unsupported crop requests."""
        predictor = DiseasePredictor()
        
        unsupported_crops = ["Cotton", "Mango", "Banana"]
        for crop in unsupported_crops:
            res = predictor.predict(image_input=None, crop=crop)
            self.assertEqual(res["crop"], crop)
            self.assertFalse(res["is_supported"])
            self.assertIn("Model Pending", res["friendly_name"])
            self.assertIsNotNone(res["warning"])
            self.assertIn(crop, res["warning"])

    @patch("modules.weather.WeatherClient.fetch_weather")
    def test_crop_doctor_unsupported_crop_pipeline(self, mock_fetch_weather):
        """Test CropDoctor pipeline execution with an unsupported crop."""
        potato_leaf = ROOT_DIR / "data" / "raw" / "plant_village_subset" / "Potato___healthy" / "real_001.JPG"
        if not potato_leaf.exists():
            self.skipTest(f"Sample image not found: {potato_leaf}")

        mock_weather = WeatherData(
            latitude=17.3850,
            longitude=78.4867,
            current=CurrentWeather(
                temperature_c=25.0,
                relative_humidity_pct=70.0,
                precipitation_mm=0.0,
                wind_speed_kmh=10.0,
                weather_code=0,
                timestamp="2026-08-21T12:00"
            ),
            hourly=HourlyForecast(
                timestamps=["2026-08-21T12:00"],
                temperatures_c=[25.0],
                relative_humidities_pct=[70.0],
                precipitations_mm=[0.0],
                wind_speeds_kmh=[10.0]
            ),
            summary_24h={"avg_temp_c": 25.0, "avg_humidity_pct": 70.0, "total_precipitation_mm": 0.0, "hours_high_humidity_gte_90pct": 0},
            provider="open-meteo"
        )
        mock_fetch_weather.return_value = mock_weather

        doctor = CropDoctor()
        diag_res = doctor.diagnose(
            image_input=potato_leaf,
            latitude=17.3850,
            longitude=78.4867,
            crop="Cotton"
        )

        self.assertIsInstance(diag_res, DiagnosticResult)
        self.assertIn("Cotton", diag_res.disease)
        self.assertEqual(diag_res.prediction_status, "UNSUPPORTED_CROP")



if __name__ == "__main__":
    unittest.main()
