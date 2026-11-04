import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Union
from PIL import Image

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.disease_predictor import DiseasePredictor
from modules.explainability import GradCAM
from modules.weather import WeatherClient, WeatherData
from modules.risk import RiskEngine, RiskAssessment
from modules.identification import PlantRouter, RoutingDecision

DEFAULT_DISCLAIMERS = {
    "risk_index_type": "Environmental Vulnerability Index (Derived from microclimate temperature & relative humidity duration). This is NOT a calibrated statistical ML probability.",
    "pesticide_notice": "No automated chemical or pesticide prescriptions are generated. Consult local certified agricultural extension officers for treatment advice.",
    "scientific_limitation": "Epidemiological thresholds (Wallin DSV / P-Day models) are general rules of thumb and may require local microclimate validation for specific agricultural zones."
}

@dataclass
class DiagnosticResult:
    """
    Unified Payload returned by the CropDoctor Diagnostic Pipeline.
    """
    disease: str
    predicted_class_raw: str
    confidence: float
    class_probabilities: Dict[str, float]
    gradcam_output_path: str
    weather_summary: Dict[str, Any]
    risk_level: str
    risk_index: float
    risk_factors: List[str]
    explanation: str
    disclaimer: Dict[str, str] = field(default_factory=lambda: DEFAULT_DISCLAIMERS)
    is_low_confidence: bool = False
    is_rejected: bool = False
    prediction_status: str = "SUPPORTED_CONFIDENT"
    warning: Optional[str] = None
    routing_decision: Optional[Dict[str, Any]] = None
    detected_crop: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class CropDoctor:
    """
    End-to-End Diagnostic Orchestrator for AI Crop Doctor.
    Links PlantRouter, DiseasePredictor, GradCAM, WeatherClient, and RiskEngine seamlessly.
    """
    def __init__(
        self,
        disease_predictor: Optional[DiseasePredictor] = None,
        gradcam_explainer: Optional[GradCAM] = None,
        weather_client: Optional[WeatherClient] = None,
        risk_engine: Optional[RiskEngine] = None,
        plant_router: Optional[PlantRouter] = None,
    ):
        # Inject dependencies or instantiate defaults
        self.disease_predictor = disease_predictor if disease_predictor is not None else DiseasePredictor()
        self.gradcam_explainer = gradcam_explainer if gradcam_explainer is not None else GradCAM(predictor=self.disease_predictor)
        self.weather_client = weather_client if weather_client is not None else WeatherClient()
        self.risk_engine = risk_engine if risk_engine is not None else RiskEngine()
        self.plant_router = plant_router if plant_router is not None else PlantRouter()

    def diagnose(
        self,
        image_input: Union[str, Path, Image.Image],
        latitude: float,
        longitude: float,
        gradcam_output_path: Optional[Union[str, Path]] = None,
        crop: str = "Potato",
    ) -> DiagnosticResult:
        """
        Execute full end-to-end diagnostic pipeline.

        Args:
            image_input: File path to leaf photo or PIL Image object
            latitude: Field location latitude (-90 to 90)
            longitude: Field location longitude (-180 to 180)
            gradcam_output_path: Optional custom file path to save heatmaps
            crop: Target crop name (default: "Potato")

        Returns:
            DiagnosticResult object containing diagnosis, Grad-CAM path, weather, and risk metrics.
        """
        # Step 0: Stage 1 Plant Identification and Leaf Verification Routing
        routing_decision: Optional[RoutingDecision] = None
        active_crop = crop

        if self.plant_router is not None:
            routing_decision = self.plant_router.route(image_input, manual_crop=crop)

            # If routing failed (non-leaf, invalid, unknown species, or unsupported crop)
            if not routing_decision.route_to_specialist:
                weather_data = self.weather_client.fetch_weather(latitude, longitude)
                is_unsupported = routing_decision.routing_status == "UNSUPPORTED_CROP"

                return DiagnosticResult(
                    disease=f"{routing_decision.display_name} (Specialist Pending)" if is_unsupported else routing_decision.display_name,
                    predicted_class_raw=f"{routing_decision.target_crop}___{routing_decision.routing_status.lower()}",
                    confidence=round(routing_decision.confidence, 2),
                    class_probabilities={},
                    gradcam_output_path="",
                    weather_summary={
                        "latitude": weather_data.latitude,
                        "longitude": weather_data.longitude,
                        "provider": weather_data.provider,
                        "current": weather_data.current.to_dict(),
                        "hourly": weather_data.hourly.to_dict(),
                        "summary_24h": weather_data.summary_24h,
                    },
                    risk_level="LOW",
                    risk_index=0.0,
                    risk_factors=[],
                    explanation=routing_decision.message,
                    disclaimer=DEFAULT_DISCLAIMERS,
                    is_low_confidence=True,
                    is_rejected=True,
                    prediction_status=routing_decision.routing_status,
                    warning=routing_decision.warning,
                    routing_decision=routing_decision.to_dict(),
                    detected_crop=routing_decision.target_crop
                )

            # Successfully routed to specialist (e.g. Potato)
            active_crop = routing_decision.target_crop or crop

        # Step 1: Run Disease Classification with verified active crop
        pred_result = self.disease_predictor.predict(image_input, crop=active_crop)

        # Multimodal AI Clinical Refinement if specialist model is uncertain (<60% confidence)
        if pred_result.get("is_low_confidence") or pred_result.get("confidence", 0) < 60.0 or pred_result.get("predicted_disease") is None:
            try:
                from modules.identification.multimodal_verifier import multimodal_verify_leaf
                multi_res = multimodal_verify_leaf(image_input)
                if multi_res and multi_res.get("disease") and multi_res.get("confidence", 0) >= 70.0:
                    disease_name = str(multi_res["disease"]).strip()
                    pred_result["predicted_disease"] = f"{active_crop}___{disease_name.replace(' ', '_')}"
                    pred_result["friendly_name"] = disease_name
                    pred_result["confidence"] = float(multi_res["confidence"])
                    pred_result["is_low_confidence"] = False
                    pred_result["is_rejected"] = False
                    pred_result["prediction_status"] = "SUPPORTED_CONFIDENT"
                    pred_result["warning"] = None
                    if "probabilities" in pred_result and isinstance(pred_result["probabilities"], dict):
                        pred_result["probabilities"][disease_name] = float(multi_res["confidence"]) / 100.0
            except Exception:
                pass

        # Step 2: Generate Grad-CAM Visual Explanation (if crop supported)
        gradcam_path = ""
        if pred_result.get("is_supported", True):
            try:
                gradcam_res = self.gradcam_explainer.generate_explanation(
                    image_input=image_input,
                    output_path=gradcam_output_path,
                    crop=active_crop
                )
                gradcam_path = gradcam_res.get("output_path", "")
            except Exception:
                gradcam_path = ""

        # Step 3: Fetch Microclimate Weather Data
        weather_data = self.weather_client.fetch_weather(latitude, longitude)
        weather_dict = weather_data.to_dict()

        # Step 4: Compute Epidemiological Risk Assessment
        risk_assessment: RiskAssessment = self.risk_engine.assess_risk(
            disease_prediction=pred_result,
            weather_data=weather_data
        )

        # Step 5: Construct Unified Diagnostic Result
        combined_warning = pred_result.get("warning")
        if routing_decision and routing_decision.warning and not combined_warning:
            combined_warning = routing_decision.warning

        return DiagnosticResult(
            disease=pred_result["friendly_name"],
            predicted_class_raw=pred_result["predicted_disease"],
            confidence=round(pred_result["confidence"], 2),
            class_probabilities={k: round(v, 2) for k, v in pred_result["probabilities"].items()},
            gradcam_output_path=gradcam_path,
            weather_summary={
                "latitude": weather_data.latitude,
                "longitude": weather_data.longitude,
                "provider": weather_data.provider,
                "current": weather_data.current.to_dict(),
                "hourly": weather_data.hourly.to_dict(),
                "summary_24h": weather_data.summary_24h,
            },
            risk_level=risk_assessment.risk_level,
            risk_index=risk_assessment.risk_score,
            risk_factors=risk_assessment.factors,
            explanation=risk_assessment.explanation,
            disclaimer=DEFAULT_DISCLAIMERS,
            is_low_confidence=pred_result.get("is_low_confidence", False),
            is_rejected=pred_result.get("is_rejected", False),
            prediction_status=pred_result.get("prediction_status", "SUPPORTED_CONFIDENT"),
            warning=combined_warning,
            routing_decision=routing_decision.to_dict() if routing_decision else None,
            detected_crop=active_crop
        )
