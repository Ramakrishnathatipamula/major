import unittest
from unittest.mock import patch, MagicMock
from modules.weather import (
    WeatherClient,
    WeatherData,
    CurrentWeather,
    HourlyForecast,
)

class TestWeatherClient(unittest.TestCase):

    def test_invalid_coordinates(self):
        client = WeatherClient()
        with self.assertRaises(ValueError):
            client.fetch_weather(95.0, 77.0)

        with self.assertRaises(ValueError):
            client.fetch_weather(28.0, -190.0)

    def test_weather_data_to_dict(self):
        current = CurrentWeather(
            temperature_c=25.5,
            relative_humidity_pct=85.0,
            precipitation_mm=1.2,
            wind_speed_kmh=12.0,
            weather_code=3,
            timestamp="2026-08-10T12:00"
        )
        hourly = HourlyForecast(
            timestamps=["2026-08-10T12:00", "2026-08-10T13:00"],
            temperatures_c=[25.5, 26.0],
            relative_humidities_pct=[85.0, 92.0],
            precipitations_mm=[1.2, 0.0],
            wind_speeds_kmh=[12.0, 10.0]
        )
        summary = {
            "max_temp_c": 26.0,
            "min_temp_c": 25.5,
            "hours_high_humidity_gte_90pct": 1
        }

        data = WeatherData(
            latitude=28.61,
            longitude=77.20,
            current=current,
            hourly=hourly,
            summary_24h=summary,
            provider="open-meteo"
        )

        d = data.to_dict()
        self.assertEqual(d["latitude"], 28.61)
        self.assertEqual(d["current"]["temperature_c"], 25.5)
        self.assertEqual(d["current"]["relative_humidity_pct"], 85.0)
        self.assertEqual(len(d["hourly"]["temperatures_c"]), 2)
        self.assertEqual(d["summary_24h"]["hours_high_humidity_gte_90pct"], 1)

    @patch("modules.weather.weather_client.requests.get")
    def test_mocked_open_meteo_response(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "current": {
                "temperature_2m": 22.0,
                "relative_humidity_2m": 91.0,
                "precipitation": 0.5,
                "wind_speed_10m": 15.0,
                "weather_code": 61,
                "time": "2026-08-10T12:00"
            },
            "hourly": {
                "time": ["2026-08-10T12:00", "2026-08-10T13:00"],
                "temperature_2m": [22.0, 21.0],
                "relative_humidity_2m": [91.0, 95.0],
                "precipitation": [0.5, 2.0],
                "wind_speed_10m": [15.0, 18.0]
            }
        }
        mock_get.return_value = mock_response

        client = WeatherClient(provider="open-meteo")
        weather_data = client.fetch_weather(43.5, -80.5)

        self.assertEqual(weather_data.latitude, 43.5)
        self.assertEqual(weather_data.longitude, -80.5)
        self.assertEqual(weather_data.current.temperature_c, 22.0)
        self.assertEqual(weather_data.current.relative_humidity_pct, 91.0)
        self.assertEqual(weather_data.current.precipitation_mm, 0.5)
        self.assertEqual(weather_data.summary_24h["hours_high_humidity_gte_90pct"], 2)

    def test_live_open_meteo_fetch(self):
        """Live integration test querying Open-Meteo endpoint."""
        client = WeatherClient(provider="open-meteo")
        try:
            data = client.fetch_weather(28.6139, 77.2090)
        except Exception as e:
            self.skipTest(f"Live network fetch failed: {e}")

        self.assertIsInstance(data, WeatherData)
        self.assertTrue(-50.0 <= data.current.temperature_c <= 60.0)
        self.assertTrue(0.0 <= data.current.relative_humidity_pct <= 100.0)
        self.assertTrue(data.current.precipitation_mm >= 0.0)
        self.assertTrue(len(data.hourly.temperatures_c) > 0)
        self.assertIn("max_temp_c", data.summary_24h)

if __name__ == "__main__":
    unittest.main()

