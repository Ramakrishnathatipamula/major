import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

LOW_CONFIDENCE_THRESHOLD = 0.60  # 60% confidence threshold

@dataclass
class DiagnosticContext:
    """
    Structured, lightweight diagnostic context payload.
    Excludes raw image tensors, model weights, and HTTP response objects.
    """
    crop: str
    disease: str
    confidence: float
    class_probabilities: Dict[str, float]
    is_low_confidence: bool
    weather_summary: Dict[str, Any]
    weather_available: bool
    risk_level: str
    risk_index: float
    risk_factors: List[str]
    gradcam_path: Optional[str] = None
    what_if_scenario: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ContextBuilder:
    """
    Orchestrates transformation of raw outputs from CropDoctor diagnostic pipeline
    and ScenarioSimulator into a clean DiagnosticContext object.
    """
    @staticmethod
    def build_context(
        diagnostic_result: Dict[str, Any],
        scenario_comparison: Optional[Dict[str, Any]] = None,
        crop: str = "Potato"
    ) -> DiagnosticContext:
        """
        Build a clean DiagnosticContext from CropDoctor output and optional ScenarioSimulator output.
        """
        disease = diagnostic_result.get("disease", "Unknown")
        confidence = float(diagnostic_result.get("confidence", 0.0))
        class_probs = diagnostic_result.get("class_probabilities", {})

        # Low confidence check (< 60%)
        is_low_conf = confidence < LOW_CONFIDENCE_THRESHOLD

        # Weather availability check
        weather_summary = diagnostic_result.get("weather_summary", {})
        weather_available = bool(weather_summary and "error" not in weather_summary)

        # Risk information
        risk_level = diagnostic_result.get("risk_level", "UNKNOWN")
        risk_index = float(diagnostic_result.get("risk_index", 0.0))
        risk_factors = diagnostic_result.get("risk_factors", [])
        gradcam_path = diagnostic_result.get("gradcam_output_path")

        # Scenario information processing
        processed_scenario = None
        if scenario_comparison:
            scen_params = scenario_comparison.get("scenario_parameters", {})
            processed_scenario = {
                "is_hypothetical": True,
                "disclaimer": "HYPOTHETICAL SCENARIO — Simulated environmental conditions only. Not actual weather.",
                "scenario_parameters": scen_params,
                "baseline_risk_level": scenario_comparison.get("baseline_risk_level", "UNKNOWN"),
                "baseline_risk_index": float(scenario_comparison.get("baseline_environmental_vulnerability_index", 0.0)),
                "scenario_risk_level": scenario_comparison.get("scenario_risk_level", "UNKNOWN"),
                "scenario_risk_index": float(scenario_comparison.get("scenario_environmental_vulnerability_index", 0.0)),
                "risk_level_changed": scenario_comparison.get("risk_level_change", "Unchanged") != "Unchanged",
                "risk_index_delta": float(scenario_comparison.get("risk_index_change", 0.0)),
                "scenario_factors": scenario_comparison.get("changed_risk_factors", [])
            }

        return DiagnosticContext(
            crop=crop,
            disease=disease,
            confidence=confidence,
            class_probabilities=class_probs,
            is_low_confidence=is_low_conf,
            weather_summary=weather_summary,
            weather_available=weather_available,
            risk_level=risk_level,
            risk_index=risk_index,
            risk_factors=risk_factors,
            gradcam_path=gradcam_path,
            what_if_scenario=processed_scenario
        )
