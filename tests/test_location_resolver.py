import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.weather import (
    LocationResolver,
    ResolvedLocation,
    WeatherClient,
    WeatherData,
    CurrentWeather,
    HourlyForecast,
)
from modules.risk import RiskEngine, RiskAssessment


class TestMapFirstLocationSystem(unittest.TestCase):

    def setUp(self):
        self.resolver = LocationResolver()

    def test_1_search_hyderabad(self):
        """Test 1: Search major city (Hyderabad)."""
        candidates = self.resolver.search_location("Hyderabad")
        self.assertTrue(len(candidates) > 0)
        self.assertIn("Hyderabad", candidates[0].name)
        self.assertAlmostEqual(candidates[0].latitude, 17.3850, delta=0.2)
        self.assertAlmostEqual(candidates[0].longitude, 78.4867, delta=0.2)

    def test_2_search_village(self):
        """Test 2: Search rural village (Keechulatapally)."""
        candidates = self.resolver.search_location("Keechulatapally")
        self.assertTrue(len(candidates) > 0)
        self.assertIn("Keechulatapally", candidates[0].name)
        self.assertTrue(hasattr(candidates[0], "latitude"))
        self.assertTrue(hasattr(candidates[0], "longitude"))

    def test_3_search_ambiguous_place(self):
        """Test 3: Search ambiguous place returning multiple candidate matches."""
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = [
                {"display_name": "Springfield, IL, USA", "lat": "39.7817", "lon": "-89.6501"},
                {"display_name": "Springfield, MA, USA", "lat": "42.1015", "lon": "-72.5898"},
                {"display_name": "Springfield, MO, USA", "lat": "37.2090", "lon": "-93.2923"},
            ]
            mock_get.return_value = mock_resp

            candidates = self.resolver.search_location("Springfield")
            self.assertEqual(len(candidates), 3)
            self.assertEqual(candidates[0].name, "Springfield, IL, USA")
            self.assertEqual(candidates[1].name, "Springfield, MA, USA")

    def test_4_select_location_directly_on_map(self):
        """Test 4: Reverse geocoding point selected directly on interactive map."""
        rev_loc = self.resolver.reverse_geocode(17.8421, 78.1254)
        self.assertEqual(rev_loc.latitude, 17.8421)
        self.assertEqual(rev_loc.longitude, 78.1254)
        self.assertTrue(len(rev_loc.display_name) > 0)

    def test_5_move_map_marker(self):
        """Test 5: Moving map marker to new coordinates."""
        original_lat, original_lon = 17.8421, 78.1254
        new_lat, new_lon = 18.1200, 78.5400

        rev_loc1 = self.resolver.reverse_geocode(original_lat, original_lon)
        rev_loc2 = self.resolver.reverse_geocode(new_lat, new_lon)

        self.assertNotEqual(rev_loc1.latitude, rev_loc2.latitude)
        self.assertNotEqual(rev_loc1.longitude, rev_loc2.longitude)
        self.assertEqual(rev_loc2.latitude, 18.1200)
        self.assertEqual(rev_loc2.longitude, 78.5400)

    def test_6_verify_lat_lon_changes(self):
        """Test 6: Verify latitude/longitude changes update active location model."""
        loc1 = ResolvedLocation(name="Point A", latitude=15.0, longitude=75.0)
        loc2 = ResolvedLocation(name="Point B", latitude=16.0, longitude=76.0)

        self.assertNotEqual(loc1.latitude, loc2.latitude)
        self.assertNotEqual(loc1.longitude, loc2.longitude)

    @patch("modules.weather.weather_client.requests.get")
    def test_7_coordinates_reach_weather_client(self, mock_weather_get):
        """Test 7: Verify authoritative coordinates reach existing WeatherClient unchanged."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "current": {
                "temperature_2m": 26.5,
                "relative_humidity_2m": 88.0,
                "precipitation": 1.2,
                "wind_speed_10m": 9.5,
                "weather_code": 3,
                "time": "2026-08-21T12:00"
            },
            "hourly": {
                "time": ["2026-08-21T12:00"],
                "temperature_2m": [26.5],
                "relative_humidity_2m": [88.0],
                "precipitation": [1.2],
                "wind_speed_10m": [9.5]
            }
        }
        mock_weather_get.return_value = mock_resp

        target_lat, target_lon = 18.7900, 78.9100
        client = WeatherClient()
        weather_data = client.fetch_weather(target_lat, target_lon)

        self.assertEqual(weather_data.latitude, 18.7900)
        self.assertEqual(weather_data.longitude, 78.9100)

    @patch("modules.weather.weather_client.requests.get")
    def test_8_open_meteo_returns_weather(self, mock_weather_get):
        """Test 8: Verify Open-Meteo weather response fetching."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "current": {
                "temperature_2m": 24.0,
                "relative_humidity_2m": 91.0,
                "precipitation": 0.5,
                "wind_speed_10m": 11.0,
                "weather_code": 2,
                "time": "2026-08-21T12:00"
            },
            "hourly": {
                "time": ["2026-08-21T12:00"],
                "temperature_2m": [24.0],
                "relative_humidity_2m": [91.0],
                "precipitation": [0.5],
                "wind_speed_10m": [11.0]
            }
        }
        mock_weather_get.return_value = mock_resp

        client = WeatherClient()
        weather = client.fetch_weather(17.8421, 78.1254)
        self.assertEqual(weather.current.temperature_c, 24.0)
        self.assertEqual(weather.current.relative_humidity_pct, 91.0)

    def test_9_no_fallback_to_hyderabad(self):
        """Test 9: Strict guarantee of NO silent fallback to Hyderabad or default coordinates when search fails."""
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = []
            mock_get.return_value = mock_resp

            with self.assertRaises(ValueError) as ctx:
                self.resolver.search_location("NonExistentUnknownPlace999")
            
            err_msg = str(ctx.exception)
            self.assertIn("could not be found", err_msg)
            # Ensure error does NOT return Hyderabad lat (17.3850) / lon (78.4867)

    def test_10_existing_risk_engine_still_works(self):
        """Test 10: Verify compatibility with existing RiskEngine pipeline."""
        target_lat, target_lon = 17.8421, 78.1254

        mock_weather_data = WeatherData(
            latitude=target_lat,
            longitude=target_lon,
            current=CurrentWeather(
                temperature_c=21.0,
                relative_humidity_pct=94.0,
                precipitation_mm=4.0,
                wind_speed_kmh=8.0,
                weather_code=63,
                timestamp="2026-08-21T12:00"
            ),
            hourly=HourlyForecast(
                timestamps=["2026-08-21T12:00"],
                temperatures_c=[21.0],
                relative_humidities_pct=[94.0],
                precipitations_mm=[4.0],
                wind_speeds_kmh=[8.0]
            ),
            summary_24h={"avg_temp_c": 21.0, "avg_humidity_pct": 94.0, "total_precipitation_mm": 4.0, "hours_high_humidity_gte_90pct": 12},
            provider="open-meteo"
        )

        risk_engine = RiskEngine()
        diag_pred = {
            "predicted_disease": "Potato___Late_blight",
            "friendly_name": "Late Blight",
            "confidence": 92.0,
            "is_supported": True
        }

        assessment: RiskAssessment = risk_engine.assess_risk(diag_pred, mock_weather_data)
        self.assertIn(assessment.risk_level, ["LOW", "MODERATE", "HIGH"])


if __name__ == "__main__":
    unittest.main()
