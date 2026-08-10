import sys
import copy
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Union, Tuple

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.weather.weather_client import WeatherData, CurrentWeather, HourlyForecast
from modules.risk.risk_engine import RiskEngine, RiskAssessment

SIMULATION_DISCLAIMERS = {
    "simulation_nature": "Scenario-based environmental vulnerability simulation. This does NOT predict guaranteed future weather or atmospheric physics.",
    "index_type": "Environmental Vulnerability Index (Derived from microclimate temperature & humidity duration). This is NOT a calibrated statistical ML probability.",
    "pesticide_notice": "No automated chemical or pesticide prescriptions are generated. Consult local certified agricultural extension officers for guidance.",
    "scientific_limitation": "Epidemiological thresholds (Wallin DSV / P-Day models) rely on general rules of thumb and may require local microclimate validation."
}

@dataclass
class ScenarioComparison:
    """
    Structured Output Payload comparing Baseline Risk vs Scenario Simulation Risk.
    """
    simulation_type: str
    baseline_conditions: Dict[str, Any]
    baseline_risk_level: str
    baseline_environmental_vulnerability_index: float
    scenario_parameters: Dict[str, Any]
    scenario_conditions: Dict[str, Any]
    scenario_risk_level: str
    scenario_environmental_vulnerability_index: float
    risk_index_change: float
    risk_level_change: str
    changed_risk_factors: List[str]
    explanation: str
    disclaimer: Dict[str, str] = field(default_factory=lambda: SIMULATION_DISCLAIMERS)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class ScenarioSimulator:
    """
    What-If Scenario Simulation Engine for AI Crop Doctor.
    Reuses the existing RiskEngine to evaluate hypothetical microclimate perturbations.
    """
    def __init__(self, risk_engine: Optional[RiskEngine] = None):
        self.risk_engine = risk_engine if risk_engine is not None else RiskEngine()

    def simulate(
        self,
        disease_prediction: Dict[str, Any],
        weather_data: Union[WeatherData, Dict[str, Any]],
        delta_temperature_c: float = 0.0,
        override_relative_humidity_pct: Optional[float] = None,
        delta_relative_humidity_pct: float = 0.0,
        delta_precipitation_mm: float = 0.0,
    ) -> ScenarioComparison:
        """
        Run a scenario simulation on top of baseline disease diagnosis and weather metrics.

        Args:
            disease_prediction: Dict returned by DiseasePredictor or CropDoctor
            weather_data: WeatherData object or dict returned by WeatherClient
            delta_temperature_c: Temperature offset in °C (+/-)
            override_relative_humidity_pct: Fixed relative humidity override % (0 - 100)
            delta_relative_humidity_pct: Relative humidity offset % (+/-)
            delta_precipitation_mm: Rainfall offset in mm (+/-)

        Returns:
            ScenarioComparison object detailing baseline vs scenario vulnerability shifts.
        """
        # Validate scenario input parameters
        self._validate_inputs(
            override_relative_humidity_pct,
            delta_relative_humidity_pct,
            delta_temperature_c,
            delta_precipitation_mm
        )

        # Normalize weather input into WeatherData instance
        if isinstance(weather_data, dict):
            w_obj = self._dict_to_weather_data(weather_data)
        elif isinstance(weather_data, WeatherData):
            w_obj = weather_data
        else:
            raise ValueError("weather_data must be a WeatherData instance or dictionary.")

        # DEEP COPY baseline weather data to preserve original data completely
        modified_weather = copy.deepcopy(w_obj)

        # Apply scenario modifications ONLY to the copied weather object
        scenario_params = self._apply_scenario_overrides(
            modified_weather,
            delta_temperature_c,
            override_relative_humidity_pct,
            delta_relative_humidity_pct,
            delta_precipitation_mm
        )

        # Recalculate 24h summary statistics on modified weather data
        self._recalculate_summary_stats(modified_weather)

        # Evaluate BASELINE risk using EXISTING RiskEngine
        baseline_assessment: RiskAssessment = self.risk_engine.assess_risk(
            disease_prediction=disease_prediction,
            weather_data=w_obj
        )

        # Evaluate SCENARIO risk using EXISTING RiskEngine
        scenario_assessment: RiskAssessment = self.risk_engine.assess_risk(
            disease_prediction=disease_prediction,
            weather_data=modified_weather
        )

        # Compute delta comparison metrics
        risk_index_diff = round(
            scenario_assessment.risk_score - baseline_assessment.risk_score, 1
        )

        if baseline_assessment.risk_level == scenario_assessment.risk_level:
            level_change = f"Unchanged ({baseline_assessment.risk_level})"
        else:
            level_change = f"{baseline_assessment.risk_level} -> {scenario_assessment.risk_level}"

        # Generate comparative explanation and factor delta
        changed_factors, explanation = self._generate_comparison_narrative(
            baseline_assessment,
            scenario_assessment,
            scenario_params,
            risk_index_diff,
            level_change
        )

        return ScenarioComparison(
            simulation_type="Scenario-based environmental vulnerability simulation",
            baseline_conditions={
                "disease": baseline_assessment.disease,
                "avg_temp_c": w_obj.summary_24h.get("avg_temp_c", 0.0),
                "avg_humidity_pct": w_obj.summary_24h.get("avg_humidity_pct", 0.0),
                "hours_high_humidity_gte_90pct": w_obj.summary_24h.get("hours_high_humidity_gte_90pct", 0),
                "total_precipitation_mm": w_obj.summary_24h.get("total_precipitation_mm", 0.0)
            },
            baseline_risk_level=baseline_assessment.risk_level,
            baseline_environmental_vulnerability_index=baseline_assessment.risk_score,
            scenario_parameters=scenario_params,
            scenario_conditions={
                "disease": scenario_assessment.disease,
                "avg_temp_c": modified_weather.summary_24h.get("avg_temp_c", 0.0),
                "avg_humidity_pct": modified_weather.summary_24h.get("avg_humidity_pct", 0.0),
                "hours_high_humidity_gte_90pct": modified_weather.summary_24h.get("hours_high_humidity_gte_90pct", 0),
                "total_precipitation_mm": modified_weather.summary_24h.get("total_precipitation_mm", 0.0)
            },
            scenario_risk_level=scenario_assessment.risk_level,
            scenario_environmental_vulnerability_index=scenario_assessment.risk_score,
            risk_index_change=risk_index_diff,
            risk_level_change=level_change,
            changed_risk_factors=changed_factors,
            explanation=explanation,
            disclaimer=SIMULATION_DISCLAIMERS
        )

    def _validate_inputs(
        self,
        override_rh: Optional[float],
        delta_rh: float,
        delta_temp: float,
        delta_precip: float
    ) -> None:
        if override_rh is not None:
            if not (0.0 <= override_rh <= 100.0):
                raise ValueError(f"override_relative_humidity_pct must be between 0.0 and 100.0, got: {override_rh}")
        if not (-50.0 <= delta_temp <= 50.0):
            raise ValueError(f"delta_temperature_c out of realistic range (-50 to +50°C), got: {delta_temp}")

    def _apply_scenario_overrides(
        self,
        weather: WeatherData,
        delta_temp: float,
        override_rh: Optional[float],
        delta_rh: float,
        delta_precip: float
    ) -> Dict[str, Any]:
        """Apply parameter offsets to modified WeatherData instance."""
        scenario_params = {}

        # 1. Temperature override
        if delta_temp != 0.0:
            scenario_params["delta_temperature_c"] = delta_temp
            weather.hourly.temperatures_c = [
                round(t + delta_temp, 2) for t in weather.hourly.temperatures_c
            ]
            weather.current.temperature_c = round(weather.current.temperature_c + delta_temp, 2)

        # 2. Relative Humidity override
        if override_rh is not None:
            scenario_params["override_relative_humidity_pct"] = override_rh
            weather.hourly.relative_humidities_pct = [
                float(override_rh) for _ in weather.hourly.relative_humidities_pct
            ]
            weather.current.relative_humidity_pct = float(override_rh)
        elif delta_rh != 0.0:
            scenario_params["delta_relative_humidity_pct"] = delta_rh
            weather.hourly.relative_humidities_pct = [
                round(min(100.0, max(0.0, rh + delta_rh)), 2) for rh in weather.hourly.relative_humidities_pct
            ]
            weather.current.relative_humidity_pct = round(
                min(100.0, max(0.0, weather.current.relative_humidity_pct + delta_rh)), 2
            )

        # 3. Precipitation override
        if delta_precip != 0.0:
            scenario_params["delta_precipitation_mm"] = delta_precip
            weather.hourly.precipitations_mm = [
                round(max(0.0, p + (delta_precip / max(1, len(weather.hourly.precipitations_mm)))), 2)
                for p in weather.hourly.precipitations_mm
            ]
            weather.current.precipitation_mm = round(max(0.0, weather.current.precipitation_mm + delta_precip), 2)

        if not scenario_params:
            scenario_params["status"] = "No overrides applied (Identical to baseline)"

        return scenario_params

    def _recalculate_summary_stats(self, weather: WeatherData) -> None:
        """Recalculate 24h summary metrics on modified hourly weather series."""
        hourly = weather.hourly
        if not hourly.temperatures_c:
            return

        temps_24 = hourly.temperatures_c[:24]
        hum_24 = hourly.relative_humidities_pct[:24]
        rain_24 = hourly.precipitations_mm[:24]
        wind_24 = hourly.wind_speeds_kmh[:24] if hourly.wind_speeds_kmh else [0.0] * 24

        hours_high_humidity = sum(1 for h in hum_24 if h >= 90.0)

        weather.summary_24h = {
            "max_temp_c": max(temps_24),
            "min_temp_c": min(temps_24),
            "avg_temp_c": round(sum(temps_24) / len(temps_24), 2),
            "max_humidity_pct": max(hum_24),
            "avg_humidity_pct": round(sum(hum_24) / len(hum_24), 2),
            "total_precipitation_mm": round(sum(rain_24), 2),
            "max_wind_speed_kmh": max(wind_24),
            "hours_high_humidity_gte_90pct": hours_high_humidity,
        }

    def _generate_comparison_narrative(
        self,
        baseline: RiskAssessment,
        scenario: RiskAssessment,
        scenario_params: Dict[str, Any],
        risk_diff: float,
        level_change: str
    ) -> Tuple[List[str], str]:
        """Build factor list and comparative explanation narrative."""
        factors = []
        if risk_diff > 0:
            factors.append(f"Environmental Vulnerability Index increased by +{risk_diff}% (Risk Level: {level_change})")
        elif risk_diff < 0:
            factors.append(f"Environmental Vulnerability Index decreased by {risk_diff}% (Risk Level: {level_change})")
        else:
            factors.append("Environmental Vulnerability Index unchanged")

        for k, v in scenario_params.items():
            factors.append(f"Scenario Change Applied: {k} = {v}")

        for sf in scenario.factors:
            if sf not in baseline.factors:
                factors.append(f"New Scenario Factor: {sf}")

        if risk_diff > 0:
            explanation = (
                f"Simulating scenario changes ({scenario_params}) causes the Environmental Vulnerability Index "
                f"to increase from {baseline.risk_score}% ({baseline.risk_level}) to {scenario.risk_score}% ({scenario.risk_level}). "
                "The altered microclimate conditions enhance spore germination and pathogen proliferation potential."
            )
        elif risk_diff < 0:
            explanation = (
                f"Simulating scenario changes ({scenario_params}) causes the Environmental Vulnerability Index "
                f"to decrease from {baseline.risk_score}% ({baseline.risk_level}) to {scenario.risk_score}% ({scenario.risk_level}). "
                "The microclimate adjustment creates less favorable conditions for pathogen development."
            )
        else:
            explanation = (
                f"Simulating scenario changes ({scenario_params}) resulted in no change to the Environmental "
                f"Vulnerability Index ({baseline.risk_score}%, {baseline.risk_level})."
            )

        return factors, explanation

    def _dict_to_weather_data(self, d: Dict[str, Any]) -> WeatherData:
        """Convert a dictionary payload into a WeatherData dataclass instance."""
        curr = d.get("current", {})
        hourly = d.get("hourly", {})
        summary = d.get("summary_24h", {})

        c_obj = CurrentWeather(
            temperature_c=curr.get("temperature_c", 0.0),
            relative_humidity_pct=curr.get("relative_humidity_pct", 0.0),
            precipitation_mm=curr.get("precipitation_mm", 0.0),
            wind_speed_kmh=curr.get("wind_speed_kmh", 0.0),
            weather_code=curr.get("weather_code", 0),
            timestamp=curr.get("timestamp", "")
        )
        h_obj = HourlyForecast(
            timestamps=hourly.get("timestamps", []),
            temperatures_c=hourly.get("temperatures_c", []),
            relative_humidities_pct=hourly.get("relative_humidities_pct", []),
            precipitations_mm=hourly.get("precipitations_mm", []),
            wind_speeds_kmh=hourly.get("wind_speeds_kmh", [])
        )
        return WeatherData(
            latitude=d.get("latitude", 0.0),
            longitude=d.get("longitude", 0.0),
            current=c_obj,
            hourly=h_obj,
            summary_24h=summary,
            provider=d.get("provider", "open-meteo")
        )
