import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Union

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.weather.weather_client import WeatherData
from .calculators import (
    LateBlightCalculator,
    EarlyBlightCalculator,
    HealthyBaselineCalculator,
)

@dataclass
class RiskAssessment:
    """
    Standardized Output Payload for Crop Disease Risk Engine.
    """
    disease: str
    risk_level: str
    risk_score: float
    forecast_window: str
    factors: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RiskEngine:
    """
    Deterministic Epidemiological Risk Assessment Engine for AI Crop Doctor.
    Combines DiseasePredictor output and WeatherClient microclimate data.
    """

    def assess_risk(
        self, 
        disease_prediction: Dict[str, Any], 
        weather_data: Union[WeatherData, Dict[str, Any]]
    ) -> RiskAssessment:
        """
        Assess crop disease risk based on visual diagnosis and weather metrics.

        Args:
            disease_prediction: Dictionary returned by DiseasePredictor.predict()
            weather_data: WeatherData instance or dict returned by WeatherClient.fetch_weather()

        Returns:
            RiskAssessment object with standardized risk fields.
        """
        # Extract disease diagnosis info
        if isinstance(disease_prediction, dict):
            predicted_disease = disease_prediction.get("predicted_disease", "Potato___healthy")
            friendly_name = disease_prediction.get("friendly_name", "Healthy")
            confidence = disease_prediction.get("confidence", 0.0)
        else:
            predicted_disease = str(disease_prediction)
            friendly_name = predicted_disease
            confidence = 100.0

        # Normalize weather input
        if isinstance(weather_data, WeatherData):
            w_dict = weather_data.to_dict()
        elif isinstance(weather_data, dict):
            w_dict = weather_data
        else:
            raise ValueError("weather_data must be a WeatherData instance or dict.")

        hourly = w_dict.get("hourly", {})
        hourly_temps = hourly.get("temperatures_c", [])
        hourly_rh = hourly.get("relative_humidities_pct", [])
        summary_24h = w_dict.get("summary_24h", {})

        # Route calculation based on predicted disease
        if predicted_disease == "Potato___Late_blight":
            return self._assess_late_blight(friendly_name, confidence, hourly_temps, hourly_rh)
        elif predicted_disease == "Potato___Early_blight":
            return self._assess_early_blight(friendly_name, confidence, hourly_temps, hourly_rh, summary_24h)
        elif predicted_disease == "Potato___healthy":
            return self._assess_healthy(friendly_name, hourly_temps, hourly_rh, summary_24h)
        else:
            return self._assess_generic(friendly_name, hourly_temps, hourly_rh)

    def _assess_late_blight(
        self, 
        friendly_name: str, 
        confidence: float, 
        hourly_temps: List[float], 
        hourly_rh: List[float]
    ) -> RiskAssessment:
        """Evaluate risk for active Late Blight infection."""
        eval_res = LateBlightCalculator.evaluate_7day_risk(hourly_temps, hourly_rh)

        factors = [
            f"Image Diagnosis: {friendly_name} (Confidence: {confidence:.1f}%)"
        ] + eval_res["factors"]

        risk_level = eval_res["risk_level"]
        risk_score = eval_res["risk_score"]

        if risk_level == "HIGH":
            explanation = (
                f"Active {friendly_name} detected in leaf sample. Environmental conditions "
                f"are highly favorable (Cumulative DSV: {eval_res['cumulative_dsv']}) for rapid sporangia germination "
                "and secondary field spread. Immediate fungicide intervention recommended."
            )
        elif risk_level == "MODERATE":
            explanation = (
                f"Active {friendly_name} detected. Weather forecast indicates moderate risk of spore "
                f"proliferation (Cumulative DSV: {eval_res['cumulative_dsv']}). Monitor canopy wetness closely."
            )
        else:
            explanation = (
                f"Active {friendly_name} detected in leaf sample, but current humidity/temperature forecast "
                "is relatively dry, slowing rapid pathogen escalation."
            )

        return RiskAssessment(
            disease=friendly_name,
            risk_level=risk_level,
            risk_score=risk_score,
            forecast_window="7-day forecast",
            factors=factors,
            explanation=explanation
        )

    def _assess_early_blight(
        self, 
        friendly_name: str, 
        confidence: float, 
        hourly_temps: List[float], 
        hourly_rh: List[float], 
        summary_24h: Dict[str, Any]
    ) -> RiskAssessment:
        """Evaluate risk for active Early Blight infection."""
        t_min = summary_24h.get("min_temp_c", 15.0)
        t_max = summary_24h.get("max_temp_c", 28.0)

        eval_res = EarlyBlightCalculator.evaluate_risk(t_min, t_max, hourly_rh)

        factors = [
            f"Image Diagnosis: {friendly_name} (Confidence: {confidence:.1f}%)"
        ] + eval_res["factors"]

        risk_level = eval_res["risk_level"]
        risk_score = eval_res["risk_score"]

        if risk_level == "HIGH":
            explanation = (
                f"Active {friendly_name} detected. Warm temperatures and high humidity "
                f"(P-Day: {eval_res['pday']}, RH >= 85%: {eval_res['high_rh_hours']} hrs) accelerate secondary "
                "conidia formation and leaf defoliation."
            )
        elif risk_level == "MODERATE":
            explanation = (
                f"Active {friendly_name} detected. Microclimate thermal accumulation favors moderate "
                "lesion expansion on lower foliage."
            )
        else:
            explanation = (
                f"Active {friendly_name} detected, but microclimate thermal accumulation remains low."
            )

        return RiskAssessment(
            disease=friendly_name,
            risk_level=risk_level,
            risk_score=risk_score,
            forecast_window="24-hour forecast",
            factors=factors,
            explanation=explanation
        )

    def _assess_healthy(
        self, 
        friendly_name: str, 
        hourly_temps: List[float], 
        hourly_rh: List[float], 
        summary_24h: Dict[str, Any]
    ) -> RiskAssessment:
        """Evaluate preventive risk for healthy leaf sample."""
        # Calculate raw environmental vulnerability using Late Blight DSV model
        lb_eval = LateBlightCalculator.evaluate_7day_risk(hourly_temps, hourly_rh)
        env_score = lb_eval["risk_score"]

        healthy_eval = HealthyBaselineCalculator.adjust_risk_for_healthy(env_score, lb_eval["factors"])

        risk_level = healthy_eval["risk_level"]
        risk_score = healthy_eval["risk_score"]

        if risk_level == "HIGH":
            explanation = (
                "Leaf sample appears Healthy, but microclimate weather forecast predicts high humidity "
                "and conducive temperatures. Preventive field monitoring advised against latent spore invasion."
            )
        elif risk_level == "MODERATE":
            explanation = (
                "Leaf sample appears Healthy. Weather forecast indicates moderate environmental vulnerability. "
                "Routine field scouting recommended."
            )
        else:
            explanation = (
                "Leaf sample appears Healthy and weather forecast indicates low environmental vulnerability. "
                "Standard crop management applies."
            )

        return RiskAssessment(
            disease=friendly_name,
            risk_level=risk_level,
            risk_score=risk_score,
            forecast_window="7-day forecast",
            factors=healthy_eval["factors"],
            explanation=explanation
        )

    def _assess_generic(
        self, 
        disease_name: str, 
        hourly_temps: List[float], 
        hourly_rh: List[float]
    ) -> RiskAssessment:
        """Fallback risk evaluator for non-standard disease labels."""
        lb_eval = LateBlightCalculator.evaluate_7day_risk(hourly_temps, hourly_rh)
        return RiskAssessment(
            disease=disease_name,
            risk_level=lb_eval["risk_level"],
            risk_score=lb_eval["risk_score"],
            forecast_window="7-day forecast",
            factors=[f"Diagnosis: {disease_name}"] + lb_eval["factors"],
            explanation=f"Evaluated environmental risk for {disease_name} based on humidity/temp microclimate."
        )
