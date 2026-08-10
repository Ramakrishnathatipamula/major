import unittest
from unittest.mock import patch, MagicMock
from copy import deepcopy

from modules.weather import WeatherData, CurrentWeather, HourlyForecast
from modules.risk import RiskEngine, RiskAssessment
from modules.simulation import ScenarioSimulator, ScenarioComparison

class TestScenarioSimulator(unittest.TestCase):

    def setUp(self):
        self.disease_pred = {
            "predicted_disease": "Potato___Late_blight",
            "friendly_name": "Late Blight",
            "confidence": 98.5,
            "probabilities": {"Late Blight": 98.5, "Early Blight": 1.0, "Healthy": 0.5}
        }

        self.baseline_weather = WeatherData(
            latitude=28.6139,
            longitude=77.2090,
            current=CurrentWeather(
                temperature_c=22.0,
                relative_humidity_pct=70.0,
                precipitation_mm=0.0,
                wind_speed_kmh=10.0,
                weather_code=1,
                timestamp="2026-08-10T12:00"
            ),
            hourly=HourlyForecast(
                timestamps=[f"2026-08-10T{i:02d}:00" for i in range(24)],
                temperatures_c=[22.0] * 24,
                relative_humidities_pct=[70.0] * 24,  # All dry
                precipitations_mm=[0.0] * 24,
                wind_speeds_kmh=[10.0] * 24
            ),
            summary_24h={
                "max_temp_c": 22.0,
                "min_temp_c": 22.0,
                "avg_temp_c": 22.0,
                "max_humidity_pct": 70.0,
                "avg_humidity_pct": 70.0,
                "total_precipitation_mm": 0.0,
                "max_wind_speed_kmh": 10.0,
                "hours_high_humidity_gte_90pct": 0
            },
            provider="open-meteo"
        )
        self.simulator = ScenarioSimulator()

    def test_1_no_scenario_change(self):
        """Test 1: No scenario change -> baseline and scenario should match."""
        res = self.simulator.simulate(self.disease_pred, self.baseline_weather)
        self.assertEqual(res.baseline_risk_level, res.scenario_risk_level)
        self.assertEqual(res.baseline_environmental_vulnerability_index, res.scenario_environmental_vulnerability_index)
        self.assertEqual(res.risk_index_change, 0.0)

    def test_2_increased_rh_changes_payload(self):
        """Test 2: Increased RH changes the weather payload and risk level."""
        res = self.simulator.simulate(
            self.disease_pred,
            self.baseline_weather,
            override_relative_humidity_pct=95.0
        )
        self.assertEqual(res.scenario_conditions["avg_humidity_pct"], 95.0)
        self.assertEqual(res.scenario_conditions["hours_high_humidity_gte_90pct"], 24)
        self.assertTrue(res.scenario_environmental_vulnerability_index > res.baseline_environmental_vulnerability_index)

    def test_3_temperature_scenario(self):
        """Test 3: Temperature scenario changes temperature correctly."""
        res = self.simulator.simulate(
            self.disease_pred,
            self.baseline_weather,
            delta_temperature_c=3.5
        )
        self.assertEqual(res.scenario_conditions["avg_temp_c"], 25.5)

    def test_4_precipitation_scenario(self):
        """Test 4: Precipitation scenario changes precipitation correctly."""
        res = self.simulator.simulate(
            self.disease_pred,
            self.baseline_weather,
            delta_precipitation_mm=12.0
        )
        self.assertEqual(res.scenario_conditions["total_precipitation_mm"], 12.0)

    def test_5_baseline_weather_remains_unchanged(self):
        """Test 5: Original baseline weather data remains 100% unchanged."""
        original_temp = self.baseline_weather.current.temperature_c
        original_rh = self.baseline_weather.current.relative_humidity_pct
        original_hourly_rh = list(self.baseline_weather.hourly.relative_humidities_pct)

        self.simulator.simulate(
            self.disease_pred,
            self.baseline_weather,
            delta_temperature_c=10.0,
            override_relative_humidity_pct=99.0,
            delta_precipitation_mm=50.0
        )

        self.assertEqual(self.baseline_weather.current.temperature_c, original_temp)
        self.assertEqual(self.baseline_weather.current.relative_humidity_pct, original_rh)
        self.assertEqual(self.baseline_weather.hourly.relative_humidities_pct, original_hourly_rh)

    def test_6_existing_risk_engine_reused(self):
        """Test 6: Existing RiskEngine is actually reused (assess_risk called twice)."""
        mock_risk_engine = MagicMock(spec=RiskEngine)
        mock_assessment = RiskAssessment(
            disease="Late Blight",
            risk_level="LOW",
            risk_score=10.0,
            forecast_window="7-day forecast",
            factors=["Factor A"],
            explanation="Explanation A"
        )
        mock_risk_engine.assess_risk.return_value = mock_assessment

        simulator = ScenarioSimulator(risk_engine=mock_risk_engine)
        simulator.simulate(
            self.disease_pred,
            self.baseline_weather,
            override_relative_humidity_pct=85.0
        )

        self.assertEqual(mock_risk_engine.assess_risk.call_count, 2)

    def test_7_invalid_scenario_values_rejected(self):
        """Test 7: Invalid scenario values are rejected."""
        with self.assertRaises(ValueError):
            self.simulator.simulate(self.disease_pred, self.baseline_weather, override_relative_humidity_pct=150.0)

        with self.assertRaises(ValueError):
            self.simulator.simulate(self.disease_pred, self.baseline_weather, override_relative_humidity_pct=-10.0)

        with self.assertRaises(ValueError):
            self.simulator.simulate(self.disease_pred, self.baseline_weather, delta_temperature_c=100.0)

    def test_8_scenario_comparison_structure(self):
        """Test 8: Scenario comparison is generated with all required fields and disclaimers."""
        res = self.simulator.simulate(
            self.disease_pred,
            self.baseline_weather,
            delta_relative_humidity_pct=20.0
        )

        self.assertIsInstance(res, ScenarioComparison)
        d = res.to_dict()

        required_keys = [
            "simulation_type",
            "baseline_conditions",
            "baseline_risk_level",
            "baseline_environmental_vulnerability_index",
            "scenario_parameters",
            "scenario_conditions",
            "scenario_risk_level",
            "scenario_environmental_vulnerability_index",
            "risk_index_change",
            "risk_level_change",
            "changed_risk_factors",
            "explanation",
            "disclaimer"
        ]
        for key in required_keys:
            self.assertIn(key, d, f"Missing key '{key}' in ScenarioComparison output")

        self.assertIn("index_type", d["disclaimer"])
        self.assertIn("Environmental Vulnerability Index", d["disclaimer"]["index_type"])

if __name__ == "__main__":
    unittest.main()
