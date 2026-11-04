import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Configurable confidence thresholds in standardized percentage units [0.0, 100.0]
# NOTE: Softmax confidence indicates relative probability distribution over trained classes;
# it is NOT an absolute guarantee of biological or agronomic accuracy.
CONFIDENCE_WARNING_THRESHOLD: float = 60.0    # Marginal certainty boundary (60.0%)
CONFIDENCE_REJECTION_THRESHOLD: float = 40.0  # Near random-guess boundary for 3 classes (40.0%)
LOW_CONFIDENCE_THRESHOLD: float = CONFIDENCE_WARNING_THRESHOLD  # Backward compatibility alias


def normalize_confidence_pct(val: Any) -> float:
    """
    Normalize confidence input to consistent percentage units [0.0, 100.0].
    Converts fractional inputs (0.0 < val <= 1.0] from legacy mocks or callers to percentage.
    """
    try:
        c = float(val)
    except (TypeError, ValueError):
        return 0.0

    if 0.0 < c <= 1.0:
        return round(c * 100.0, 2)
    return round(max(0.0, min(100.0, c)), 2)


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
    is_rejected: bool = False
    prediction_status: str = "SUPPORTED_CONFIDENT"
    warning: Optional[str] = None
    requested_crop: Optional[str] = None

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
        crop: str = "Potato",
        requested_crop: Optional[str] = None
    ) -> DiagnosticContext:
        """
        Build a clean DiagnosticContext from CropDoctor output and optional ScenarioSimulator output.

        Args:
            diagnostic_result: Pipeline result payload.
            scenario_comparison: Optional What-If simulation comparison.
            crop: Canonical crop actually diagnosed.
            requested_crop: Original user-selected crop, retained for audit.
        """
        disease = diagnostic_result.get("disease", "Unknown")
        raw_conf = diagnostic_result.get("confidence", 0.0)
        confidence = normalize_confidence_pct(raw_conf)
        class_probs = diagnostic_result.get("class_probabilities", {})

        # Confidence policy evaluation
        is_rejected = (
            diagnostic_result.get("is_rejected")
            if "is_rejected" in diagnostic_result
            else (confidence < CONFIDENCE_REJECTION_THRESHOLD)
        )
        is_low_conf = (
            diagnostic_result.get("is_low_confidence")
            if "is_low_confidence" in diagnostic_result
            else (confidence < CONFIDENCE_WARNING_THRESHOLD)
        )

        pred_status = diagnostic_result.get("prediction_status")
        if not pred_status:
            if is_rejected:
                pred_status = "REJECTED_LOW_CONFIDENCE"
            elif is_low_conf:
                pred_status = "SUPPORTED_UNCERTAIN"
            else:
                pred_status = "SUPPORTED_CONFIDENT"

        warning = diagnostic_result.get("warning")
        if not warning:
            if is_rejected:
                warning = (
                    f"Prediction rejected: Diagnostic confidence ({confidence:.1f}%) is below minimum threshold ({CONFIDENCE_REJECTION_THRESHOLD}%). "
                    "Prediction is near random guess."
                )
            elif is_low_conf:
                warning = f"Low Diagnostic Certainty ({confidence:.1f}% < {CONFIDENCE_WARNING_THRESHOLD}%). Inspect secondary leaves."

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
            what_if_scenario=processed_scenario,
            is_rejected=is_rejected,
            prediction_status=pred_status,
            warning=warning,
            requested_crop=requested_crop
        )
