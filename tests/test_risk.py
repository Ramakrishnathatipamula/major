import unittest
from modules.risk import (
    RiskEngine,
    RiskAssessment,
    LateBlightCalculator,
    EarlyBlightCalculator,
    HealthyBaselineCalculator,
)
from modules.weather import WeatherData, CurrentWeather, HourlyForecast

class TestRiskEngine(unittest.TestCase):

    def setUp(self):
        self.engine = RiskEngine()

        # High risk weather data (16 hours RH >= 90% at 20°C)
        self.high_risk_weather = WeatherData(
            latitude=28.61,
            longitude=77.20,
            current=CurrentWeather(
                temperature_c=20.0,
                relative_humidity_pct=92.0,
                precipitation_mm=2.5,
                wind_speed_kmh=8.0,
                weather_code=61,
                timestamp="2026-08-10T12:00"
            ),
            hourly=HourlyForecast(
                timestamps=[f"2026-08-10T{i:02d}:00" for i in range(24)],
                temperatures_c=[20.0] * 24,
                relative_humidities_pct=[92.0] * 16 + [70.0] * 8,  # 16 hours RH >= 90%
                precipitations_mm=[0.2] * 24,
                wind_speeds_kmh=[10.0] * 24
            ),
            summary_24h={
                "max_temp_c": 22.0,
                "min_temp_c": 18.0,
                "avg_temp_c": 20.0,
                "max_humidity_pct": 95.0,
                "avg_humidity_pct": 84.0,
                "total_precipitation_mm": 4.8,
                "max_wind_speed_kmh": 12.0,
                "hours_high_humidity_gte_90pct": 16
            },
            provider="open-meteo"
        )

        # Low risk weather data (all dry, RH = 50%)
        self.low_risk_weather = WeatherData(
            latitude=28.61,
            longitude=77.20,
            current=CurrentWeather(
                temperature_c=25.0,
                relative_humidity_pct=50.0,
                precipitation_mm=0.0,
                wind_speed_kmh=5.0,
                weather_code=1,
                timestamp="2026-08-10T12:00"
            ),
            hourly=HourlyForecast(
                timestamps=[f"2026-08-10T{i:02d}:00" for i in range(24)],
                temperatures_c=[25.0] * 24,
                relative_humidities_pct=[50.0] * 24,
                precipitations_mm=[0.0] * 24,
                wind_speeds_kmh=[5.0] * 24
            ),
            summary_24h={
                "max_temp_c": 28.0,
                "min_temp_c": 22.0,
                "avg_temp_c": 25.0,
                "max_humidity_pct": 55.0,
                "avg_humidity_pct": 50.0,
                "total_precipitation_mm": 0.0,
                "max_wind_speed_kmh": 6.0,
                "hours_high_humidity_gte_90pct": 0
            },
            provider="open-meteo"
        )

    def test_wallin_dsv_calculator(self):
        # 16 hours RH >= 90% at 20°C (Range 15.1 - 26.6°C) should return 4 DSVs
        temps = [20.0] * 24
        rh = [92.0] * 16 + [70.0] * 8
        dsv, avg_temp, hrs = LateBlightCalculator.calculate_daily_dsv(temps, rh)
        self.assertEqual(dsv, 4)
        self.assertEqual(hrs, 16)
        self.assertEqual(avg_temp, 20.0)

    def test_early_blight_pday_calculator(self):
        pday = EarlyBlightCalculator.calculate_daily_pday(t_min=18.0, t_max=28.0)
        self.assertTrue(pday > 0.0)

    def test_late_blight_high_risk_assessment(self):
        prediction = {
            "predicted_disease": "Potato___Late_blight",
            "friendly_name": "Late Blight",
            "confidence": 98.5
        }
        res = self.engine.assess_risk(prediction, self.high_risk_weather)

        self.assertIsInstance(res, RiskAssessment)
        self.assertEqual(res.disease, "Late Blight")
        self.assertEqual(res.risk_level, "HIGH")
        self.assertTrue(res.risk_score >= 50.0)
        self.assertIn("Cumulative DSV", res.explanation)

        res_dict = res.to_dict()
        self.assertIn("disease", res_dict)
        self.assertIn("risk_level", res_dict)
        self.assertIn("risk_score", res_dict)
        self.assertIn("forecast_window", res_dict)
        self.assertIn("factors", res_dict)
        self.assertIn("explanation", res_dict)

    def test_healthy_low_risk_assessment(self):
        prediction = {
            "predicted_disease": "Potato___healthy",
            "friendly_name": "Healthy",
            "confidence": 99.1
        }
        res = self.engine.assess_risk(prediction, self.low_risk_weather)

        self.assertEqual(res.disease, "Healthy")
        self.assertEqual(res.risk_level, "LOW")
        self.assertEqual(res.risk_score, 0.0)
        self.assertIn("Healthy", res.explanation)

    def test_early_blight_assessment(self):
        prediction = {
            "predicted_disease": "Potato___Early_blight",
            "friendly_name": "Early Blight",
            "confidence": 92.0
        }
        res = self.engine.assess_risk(prediction, self.high_risk_weather)

        self.assertEqual(res.disease, "Early Blight")
        self.assertIn(res.risk_level, ["MODERATE", "HIGH"])
        self.assertTrue(len(res.factors) > 0)

if __name__ == "__main__":
    unittest.main()
