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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class CropDoctor:
    """
    End-to-End Diagnostic Orchestrator for AI Crop Doctor.
    Links DiseasePredictor, GradCAM, WeatherClient, and RiskEngine seamlessly.
    """
    def __init__(
        self,
        disease_predictor: Optional[DiseasePredictor] = None,
        gradcam_explainer: Optional[GradCAM] = None,
        weather_client: Optional[WeatherClient] = None,
        risk_engine: Optional[RiskEngine] = None,
    ):
        # Inject dependencies or instantiate defaults
        self.disease_predictor = disease_predictor if disease_predictor is not None else DiseasePredictor()
        self.gradcam_explainer = gradcam_explainer if gradcam_explainer is not None else GradCAM(predictor=self.disease_predictor)
        self.weather_client = weather_client if weather_client is not None else WeatherClient()
        self.risk_engine = risk_engine if risk_engine is not None else RiskEngine()

    def diagnose(
        self,
        image_input: Union[str, Path, Image.Image],
        latitude: float,
        longitude: float,
        gradcam_output_path: Optional[Union[str, Path]] = None,
    ) -> DiagnosticResult:
        """
        Execute full end-to-end diagnostic pipeline.

        Args:
            image_input: File path to leaf photo or PIL Image object
            latitude: Field location latitude (-90 to 90)
            longitude: Field location longitude (-180 to 180)
            gradcam_output_path: Optional custom file path to save heatmaps

        Returns:
            DiagnosticResult object containing diagnosis, Grad-CAM path, weather, and risk metrics.
        """
        # Step 1: Run Disease Classification
        pred_result = self.disease_predictor.predict(image_input)

        # Step 2: Generate Grad-CAM Visual Explanation
        gradcam_res = self.gradcam_explainer.generate_explanation(
            image_input=image_input,
            output_path=gradcam_output_path
        )
        gradcam_path = gradcam_res.get("output_path", "")

        # Step 3: Fetch Microclimate Weather Data
        weather_data = self.weather_client.fetch_weather(latitude, longitude)
        weather_dict = weather_data.to_dict()

        # Step 4: Compute Epidemiological Risk Assessment
        risk_assessment: RiskAssessment = self.risk_engine.assess_risk(
            disease_prediction=pred_result,
            weather_data=weather_data
        )

        # Step 5: Construct Unified Diagnostic Result
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
            disclaimer=DEFAULT_DISCLAIMERS
        )
