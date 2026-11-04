import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Union, Optional

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
        # Late Blight: Phytophthora infestans (Shared Pathogen Compatibility Rule)
        if predicted_disease in ["Potato___Late_blight", "Tomato___Late_blight"]:
            is_tomato = "Tomato" in str(predicted_disease)
            rule = (
                "Epidemiological Compatibility: Phytophthora infestans pathogen biology is shared between Potato and Tomato; "
                "Wallin DSV leaf-wetness threshold modeling is verified and applicable."
                if is_tomato else None
            )
            return self._assess_late_blight(friendly_name, confidence, hourly_temps, hourly_rh, compatibility_rule=rule)

        # Early Blight: Alternaria solani (Shared Thermal Compatibility Rule)
        elif predicted_disease in ["Potato___Early_blight", "Tomato___Early_blight"]:
            is_tomato = "Tomato" in str(predicted_disease)
            rule = (
                "Epidemiological Compatibility: Alternaria solani thermal threshold (P-Day) model applies to "
                "conidia germination on solanaceous foliage."
                if is_tomato else None
            )
            return self._assess_early_blight(friendly_name, confidence, hourly_temps, hourly_rh, summary_24h, compatibility_rule=rule)

        # Healthy Potato
        elif predicted_disease == "Potato___healthy":
            return self._assess_healthy(friendly_name, hourly_temps, hourly_rh, summary_24h)

        # Healthy Tomato (Evaluated without imposing Potato Late Blight DSV)
        elif predicted_disease == "Tomato___healthy":
            return self._assess_healthy_tomato(friendly_name, hourly_temps, hourly_rh, summary_24h)

        # Tomato Spider Mites (Arthropod Pest: Hot & Dry Conditions)
        elif "spider_mite" in predicted_disease.lower():
            return self._assess_tomato_spider_mites(friendly_name, confidence, hourly_temps, hourly_rh, summary_24h)

        # Tomato Whitefly Vector / TYLCV
        elif "yellow_leaf_curl" in predicted_disease.lower():
            return self._assess_tomato_tylcv(friendly_name, confidence, hourly_temps, hourly_rh, summary_24h)

        # Tomato Mosaic Virus (Mechanical Transmission)
        elif "mosaic_virus" in predicted_disease.lower():
            return self._assess_tomato_mosaic(friendly_name, confidence, hourly_temps, hourly_rh, summary_24h)

        # Tomato Bacterial Spot
        elif "bacterial_spot" in predicted_disease.lower():
            return self._assess_tomato_bacterial_spot(friendly_name, confidence, hourly_temps, hourly_rh, summary_24h)

        # Tomato Leaf Mold / Septoria / Target Spot
        elif any(k in predicted_disease.lower() for k in ["leaf_mold", "septoria", "target_spot"]):
            return self._assess_tomato_foliar_fungal(friendly_name, confidence, hourly_temps, hourly_rh, summary_24h)

        # Maize / Wheat / Rice: no pathogen-specific epidemiological model is
        # validated for these crops yet, so use an explicitly labelled generic
        # cereal baseline.
        elif str(predicted_disease).startswith(("Corn_(maize)___", "Wheat___", "Rice___")):
            return self._assess_cereal_baseline(
                friendly_name,
                confidence,
                hourly_temps,
                hourly_rh,
                summary_24h,
            )

        else:
            return self._assess_generic(friendly_name, hourly_temps, hourly_rh)

    def _assess_late_blight(
        self, 
        friendly_name: str, 
        confidence: float, 
        hourly_temps: List[float], 
        hourly_rh: List[float],
        compatibility_rule: Optional[str] = None
    ) -> RiskAssessment:
        """Evaluate risk for active Late Blight infection."""
        eval_res = LateBlightCalculator.evaluate_7day_risk(hourly_temps, hourly_rh)

        factors = [
            f"Image Diagnosis: {friendly_name} (Confidence: {confidence:.1f}%)"
        ]
        if compatibility_rule:
            factors.append(compatibility_rule)
        factors.extend(eval_res["factors"])

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
        summary_24h: Dict[str, Any],
        compatibility_rule: Optional[str] = None
    ) -> RiskAssessment:
        """Evaluate risk for active Early Blight infection."""
        t_min = summary_24h.get("min_temp_c", 15.0)
        t_max = summary_24h.get("max_temp_c", 28.0)

        eval_res = EarlyBlightCalculator.evaluate_risk(t_min, t_max, hourly_rh)

        factors = [
            f"Image Diagnosis: {friendly_name} (Confidence: {confidence:.1f}%)"
        ]
        if compatibility_rule:
            factors.append(compatibility_rule)
        factors.extend(eval_res["factors"])

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

    def _assess_healthy_tomato(
        self,
        friendly_name: str,
        hourly_temps: List[float],
        hourly_rh: List[float],
        summary_24h: Dict[str, Any]
    ) -> RiskAssessment:
        """Evaluate preventive microclimate risk for healthy tomato leaves without potato DSV formulas."""
        avg_rh = summary_24h.get("avg_humidity_pct", 65.0)
        high_rh_hrs = sum(1 for rh in hourly_rh[:24] if rh >= 85.0)
        max_temp = summary_24h.get("max_temp_c", 28.0)

        # Baseline environmental vulnerability for tomato
        if high_rh_hrs >= 12 and avg_rh >= 80.0:
            risk_level = "HIGH"
            risk_score = 65.0
            explanation = "Healthy foliage detected, but high canopy humidity (>85% RH for 12+ hrs) creates high vulnerability for foliar pathogens. Preventive monitoring advised."
        elif high_rh_hrs >= 6 or avg_rh >= 70.0:
            risk_level = "MODERATE"
            risk_score = 40.0
            explanation = "Healthy foliage detected with moderate canopy humidity. Standard preventive cultural practices and weekly scouting advised."
        else:
            risk_level = "LOW"
            risk_score = 15.0
            explanation = "Healthy foliage detected and weather conditions are dry with low foliar disease vulnerability."

        factors = [
            "Leaf sample visual diagnosis: Healthy (Tomato)",
            "Epidemiological Rule: Independent tomato canopy humidity baseline evaluated (potato late blight DSV not imposed)",
            f"Canopy hours with RH >= 85%: {high_rh_hrs} hrs (24h avg: {avg_rh:.1f}%)",
            f"Daily peak temperature: {max_temp:.1f}°C"
        ]

        return RiskAssessment(
            disease=friendly_name,
            risk_level=risk_level,
            risk_score=risk_score,
            forecast_window="24-hour forecast",
            factors=factors,
            explanation=explanation
        )

    def _assess_tomato_spider_mites(
        self,
        friendly_name: str,
        confidence: float,
        hourly_temps: List[float],
        hourly_rh: List[float],
        summary_24h: Dict[str, Any]
    ) -> RiskAssessment:
        """Evaluate environmental risk for Tomato Spider Mites (Arthropod pest favoured by hot, dry conditions)."""
        max_t = summary_24h.get("max_temp_c", 28.0)
        min_rh = summary_24h.get("min_humidity_pct", 50.0)
        avg_rh = summary_24h.get("avg_humidity_pct", 65.0)

        if max_t >= 32.0 and min_rh <= 45.0:
            risk_level = "HIGH"
            risk_score = 85.0
            explanation = f"Active {friendly_name} detected under hot and dry conditions (Max: {max_t:.1f}°C, Min RH: {min_rh:.1f}%). Rapid mite reproduction and webbing escalation expected."
        elif max_t >= 27.0 and min_rh <= 60.0:
            risk_level = "MODERATE"
            risk_score = 55.0
            explanation = f"Active {friendly_name} detected. Warm and moderately dry conditions support continued spider mite population expansion."
        else:
            risk_level = "LOW"
            risk_score = 25.0
            explanation = f"Active {friendly_name} detected, but cooler temperatures or higher humidity suppress rapid spider mite proliferation."

        factors = [
            f"Image Diagnosis: {friendly_name} (Confidence: {confidence:.1f}%)",
            "Epidemiological Rule: Tetranychus urticae is an arthropod pest multiplying in hot and dry weather; fungal leaf-wetness formulas are strictly inapplicable",
            f"Peak temperature: {max_t:.1f}°C (Threshold >= 28°C accelerates egg-to-adult cycle)",
            f"Minimum relative humidity: {min_rh:.1f}% (Low humidity promotes mite reproduction)"
        ]

        return RiskAssessment(
            disease=friendly_name,
            risk_level=risk_level,
            risk_score=risk_score,
            forecast_window="24-hour forecast",
            factors=factors,
            explanation=explanation
        )

    def _assess_tomato_tylcv(
        self,
        friendly_name: str,
        confidence: float,
        hourly_temps: List[float],
        hourly_rh: List[float],
        summary_24h: Dict[str, Any]
    ) -> RiskAssessment:
        """Evaluate risk for Tomato Yellow Leaf Curl Virus based on whitefly vector activity."""
        max_t = summary_24h.get("max_temp_c", 28.0)
        avg_t = summary_24h.get("avg_temp_c", 24.0)

        if max_t >= 28.0 and avg_t >= 22.0:
            risk_level = "HIGH"
            risk_score = 80.0
            explanation = f"Active {friendly_name} detected. Warm temperatures ({avg_t:.1f}°C to {max_t:.1f}°C) favor flight and reproduction of the whitefly vector (Bemisia tabaci). Rogue infected plants immediately."
        elif max_t >= 22.0:
            risk_level = "MODERATE"
            risk_score = 50.0
            explanation = f"Active {friendly_name} detected. Moderate temperatures support whitefly vector movement. Monitor yellow sticky traps and rogue infected plants."
        else:
            risk_level = "LOW"
            risk_score = 30.0
            explanation = f"Active {friendly_name} detected, but cooler weather restricts whitefly flight activity."

        factors = [
            f"Image Diagnosis: {friendly_name} (Confidence: {confidence:.1f}%)",
            "Epidemiological Rule: TYLCV is a Begomovirus transmitted by Bemisia tabaci whiteflies; fungal spore models are not applicable",
            f"Temperature profile: Avg {avg_t:.1f}°C, Max {max_t:.1f}°C (Whitefly activity peaks between 25°C and 34°C)"
        ]

        return RiskAssessment(
            disease=friendly_name,
            risk_level=risk_level,
            risk_score=risk_score,
            forecast_window="24-hour forecast",
            factors=factors,
            explanation=explanation
        )

    def _assess_tomato_mosaic(
        self,
        friendly_name: str,
        confidence: float,
        hourly_temps: List[float],
        hourly_rh: List[float],
        summary_24h: Dict[str, Any]
    ) -> RiskAssessment:
        """Evaluate risk for Tomato Mosaic Virus (Mechanical and seed transmission)."""
        factors = [
            f"Image Diagnosis: {friendly_name} (Confidence: {confidence:.1f}%)",
            "Epidemiological Rule: Tomato Mosaic Virus (ToMV) spreads mechanically via tools, hands, and seed; weather-driven fungal models are strictly inapplicable",
            "Primary Risk Driver: Field handling, pruning, staking, and contaminated nursery stock"
        ]
        return RiskAssessment(
            disease=friendly_name,
            risk_level="HIGH",
            risk_score=75.0,
            forecast_window="Crop Cycle",
            factors=factors,
            explanation=f"Active {friendly_name} detected. ToMV is highly contagious and spreads mechanically via sap on pruning shears and hands. Disinfect tools and rogue infected plants immediately."
        )

    def _assess_tomato_bacterial_spot(
        self,
        friendly_name: str,
        confidence: float,
        hourly_temps: List[float],
        hourly_rh: List[float],
        summary_24h: Dict[str, Any]
    ) -> RiskAssessment:
        """Evaluate risk for Tomato Bacterial Spot (Warm, wet weather bacterial pathogen)."""
        max_t = summary_24h.get("max_temp_c", 28.0)
        high_rh_hrs = sum(1 for rh in hourly_rh[:24] if rh >= 85.0)

        if max_t >= 24.0 and high_rh_hrs >= 8:
            risk_level = "HIGH"
            risk_score = 75.0
            explanation = f"Active {friendly_name} detected under warm and wet canopy conditions ({high_rh_hrs} hrs high humidity, Max {max_t:.1f}°C). Xanthomonas bacteria multiply rapidly in free moisture."
        elif high_rh_hrs >= 4:
            risk_level = "MODERATE"
            risk_score = 45.0
            explanation = f"Active {friendly_name} detected. Moderate moisture supports secondary bacterial spot spread."
        else:
            risk_level = "LOW"
            risk_score = 25.0
            explanation = f"Active {friendly_name} detected, but dry canopy conditions limit bacterial spread."

        factors = [
            f"Image Diagnosis: {friendly_name} (Confidence: {confidence:.1f}%)",
            "Epidemiological Rule: Xanthomonas vesicatoria requires warm temperatures and splashing moisture/leaf wetness",
            f"Canopy hours with RH >= 85%: {high_rh_hrs} hrs",
            f"Maximum temperature: {max_t:.1f}°C"
        ]

        return RiskAssessment(
            disease=friendly_name,
            risk_level=risk_level,
            risk_score=risk_score,
            forecast_window="24-hour forecast",
            factors=factors,
            explanation=explanation
        )

    def _assess_tomato_foliar_fungal(
        self,
        friendly_name: str,
        confidence: float,
        hourly_temps: List[float],
        hourly_rh: List[float],
        summary_24h: Dict[str, Any]
    ) -> RiskAssessment:
        """Evaluate risk for foliar fungal diseases (Leaf Mold, Septoria, Target Spot) under humidity thresholds."""
        high_rh_hrs = sum(1 for rh in hourly_rh[:24] if rh >= 85.0)
        avg_temp = summary_24h.get("avg_temp_c", 24.0)

        if high_rh_hrs >= 10 and 18.0 <= avg_temp <= 28.0:
            risk_level = "HIGH"
            risk_score = 70.0
            explanation = f"Active {friendly_name} detected. Extended canopy humidity ({high_rh_hrs} hrs >= 85% RH) and moderate temperature ({avg_temp:.1f}°C) accelerate fungal sporulation and lesion expansion."
        elif high_rh_hrs >= 5:
            risk_level = "MODERATE"
            risk_score = 45.0
            explanation = f"Active {friendly_name} detected. Moderate humidity supports gradual disease progression. Increase aeration and monitor lower foliage."
        else:
            risk_level = "LOW"
            risk_score = 25.0
            explanation = f"Active {friendly_name} detected, but relatively dry canopy conditions retard fungal spore germination."

        factors = [
            f"Image Diagnosis: {friendly_name} (Confidence: {confidence:.1f}%)",
            "Epidemiological Rule: Foliar fungal conidia require extended relative humidity duration without applying potato late blight DSV formulas",
            f"Hours with relative humidity >= 85%: {high_rh_hrs} hrs",
            f"Mean temperature: {avg_temp:.1f}°C"
        ]

        return RiskAssessment(
            disease=friendly_name,
            risk_level=risk_level,
            risk_score=risk_score,
            forecast_window="24-hour forecast",
            factors=factors,
            explanation=explanation
        )

    def _assess_cereal_baseline(
        self,
        friendly_name: str,
        confidence: float,
        hourly_temps: List[float],
        hourly_rh: List[float],
        summary_24h: Dict[str, Any],
    ) -> RiskAssessment:
        """
        Transparent generic microclimate baseline for Maize/Wheat.

        No pathogen-specific epidemiological model (e.g. Wallin DSV) has been
        validated for these crops here, so this method deliberately does not
        reuse Potato/Tomato pathogen formulas.
        """
        avg_rh = float(summary_24h.get("avg_humidity_pct", 65.0))
        max_temp = float(summary_24h.get("max_temp_c", 28.0))
        total_precip = float(summary_24h.get("total_precipitation_mm", 0.0))
        high_rh_hrs = sum(1 for rh in hourly_rh[:24] if rh >= 85.0)

        if high_rh_hrs >= 12 or total_precip >= 10.0:
            risk_level = "HIGH"
            risk_score = 60.0
        elif high_rh_hrs >= 6 or avg_rh >= 70.0 or total_precip >= 2.0:
            risk_level = "MODERATE"
            risk_score = 35.0
        else:
            risk_level = "LOW"
            risk_score = 12.0

        factors = [
            f"Image Diagnosis: {friendly_name} (Confidence: {confidence:.1f}%)",
            "Risk Model: Generic cereal microclimate baseline (no pathogen-specific epidemiological model validated for this crop).",
            f"Canopy hours with RH >= 85%: {high_rh_hrs} hrs (24h avg: {avg_rh:.1f}%)",
            f"Daily peak temperature: {max_temp:.1f}°C; 24h precipitation: {total_precip:.1f} mm",
        ]
        explanation = (
            f"Active {friendly_name} detected. The generic cereal baseline indicates "
            f"{risk_level.lower()} environmental favorability for foliar disease spread. "
            "This is not a crop-specific validated epidemic probability; use it for scouting "
            "and consult local extension guidance."
        )
        return RiskAssessment(
            disease=friendly_name,
            risk_level=risk_level,
            risk_score=risk_score,
            forecast_window="24-hour forecast",
            factors=factors,
            explanation=explanation,
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
