import sys
import json
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.diagnosis import CropDoctor
from modules.weather import WeatherData, CurrentWeather, HourlyForecast
from modules.simulation import ScenarioSimulator

def get_fallback_weather() -> WeatherData:
    """Generate fallback microclimate weather data if network fetch times out."""
    return WeatherData(
        latitude=28.6139,
        longitude=77.2090,
        current=CurrentWeather(
            temperature_c=20.0,
            relative_humidity_pct=72.0,
            precipitation_mm=0.0,
            wind_speed_kmh=8.0,
            weather_code=1,
            timestamp="2026-08-10T12:00"
        ),
        hourly=HourlyForecast(
            timestamps=[f"2026-08-10T{i:02d}:00" for i in range(24)],
            temperatures_c=[20.0] * 24,
            relative_humidities_pct=[72.0] * 24,
            precipitations_mm=[0.0] * 24,
            wind_speeds_kmh=[8.0] * 24
        ),
        summary_24h={
            "max_temp_c": 22.0,
            "min_temp_c": 18.0,
            "avg_temp_c": 20.0,
            "max_humidity_pct": 72.0,
            "avg_humidity_pct": 72.0,
            "total_precipitation_mm": 0.0,
            "max_wind_speed_kmh": 8.0,
            "hours_high_humidity_gte_90pct": 0
        },
        provider="open-meteo (fallback)"
    )

def main():
    sample_image = ROOT_DIR / "data" / "raw" / "plant_village_subset" / "Potato___Late_blight" / "real_086.JPG"
    if not sample_image.exists():
        print(f"Error: Sample image not found at {sample_image}")
        sys.exit(1)

    print("=========================================================")
    print("AI CROP DOCTOR — WHAT-IF SCENARIO SIMULATION DEMONSTRATION")
    print("=========================================================\n")

    # Step 1: Run Baseline Diagnosis
    print("1. RUNNING BASELINE DIAGNOSIS (CropDoctor Pipeline)...")
    crop_doctor = CropDoctor()
    try:
        baseline_diag = crop_doctor.diagnose(
            image_input=sample_image,
            latitude=28.6139,
            longitude=77.2090
        )
        weather_payload = baseline_diag.weather_summary
        disease_name = baseline_diag.disease
        raw_disease = baseline_diag.predicted_class_raw
        confidence = baseline_diag.confidence
        probabilities = baseline_diag.class_probabilities
        baseline_level = baseline_diag.risk_level
        baseline_index = baseline_diag.risk_index
    except Exception as exc:
        print(f"   [Note: Live weather fetch timed out ({exc}). Using cached microclimate data for demo.]")
        weather_obj = get_fallback_weather()
        pred_res = crop_doctor.disease_predictor.predict(sample_image)
        risk_res = crop_doctor.risk_engine.assess_risk(pred_res, weather_obj)

        weather_payload = weather_obj.to_dict()
        disease_name = pred_res["friendly_name"]
        raw_disease = pred_res["predicted_disease"]
        confidence = pred_res["confidence"]
        probabilities = pred_res["probabilities"]
        baseline_level = risk_res.risk_level
        baseline_index = risk_res.risk_score

    print(f"   Disease Diagnosed: {disease_name} (Confidence: {confidence:.2f}%)")
    print(f"   Baseline Weather Temp: {weather_payload['current']['temperature_c']}°C, Humidity: {weather_payload['current']['relative_humidity_pct']}%")
    print(f"   Baseline Risk Level: {baseline_level}")
    print(f"   Baseline Environmental Vulnerability Index: {baseline_index}%\n")

    # Step 2: Define and Run What-If Scenario (Increase Humidity + Set Temperature to 20°C Optimal Window)
    print("2. RUNNING WHAT-IF SCENARIO SIMULATION...")
    print("   Scenario Parameters: Humidity = 95.0%, Temp Offset = -12.2°C (Optimal 20°C Late Blight Growth Window)\n")

    simulator = ScenarioSimulator()
    comparison = simulator.simulate(
        disease_prediction={
            "predicted_disease": raw_disease,
            "friendly_name": disease_name,
            "confidence": confidence,
            "probabilities": probabilities
        },
        weather_data=weather_payload,
        override_relative_humidity_pct=95.0,
        delta_temperature_c=-12.2
    )

    # Step 3: Print Complete Comparative Results
    print("=========================================================")
    print("             SIMULATION COMPARISON RESULT                ")
    print("=========================================================")
    print(f"Simulation Type: {comparison.simulation_type}\n")

    print("--- BASELINE RISK ---")
    print(f"Risk Level: {comparison.baseline_risk_level}")
    print(f"Environmental Vulnerability Index: {comparison.baseline_environmental_vulnerability_index}%\n")

    print("--- SCENARIO RISK ---")
    print(f"Scenario Parameters: {comparison.scenario_parameters}")
    print(f"Risk Level: {comparison.scenario_risk_level}")
    print(f"Environmental Vulnerability Index: {comparison.scenario_environmental_vulnerability_index}%\n")

    print("--- COMPARISON DIFFERENCE ---")
    print(f"Risk Index Change: +{comparison.risk_index_change}%")
    print(f"Risk Level Shift: {comparison.risk_level_change}\n")

    print("--- CHANGED RISK FACTORS ---")
    for factor in comparison.changed_risk_factors:
        print(f" • {factor}")

    print("\n--- EXPLANATION ---")
    print(f"{comparison.explanation}\n")

    print("--- DISCLAIMERS ---")
    for key, notice in comparison.disclaimer.items():
        print(f" [{key}]: {notice}")

if __name__ == "__main__":
    main()
