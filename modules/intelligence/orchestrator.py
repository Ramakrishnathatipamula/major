import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.crops import get_crop_registry
from modules.diagnosis import CropDoctor
from modules.simulation import ScenarioSimulator
from modules.intelligence.context_builder import ContextBuilder, DiagnosticContext
from modules.intelligence.query_generator import RAGQueryGenerator
from modules.rag import RAGAnswerer, RAGAnswer
from modules.treatment import PotatoTreatmentRecommender, TreatmentRecommendationResult

STANDARD_DISCLAIMERS = [
    "Environmental Vulnerability Index is a deterministic microclimate rule index, not a calibrated statistical probability.",
    "What-If scenario results are hypothetical sensitivity simulations and do not represent actual weather forecasts.",
    "RAG knowledge answers provide general educational agricultural extension guidance.",
    "No automatic chemical pesticide prescriptions or custom dosage formulations are provided."
]


def _canonical_crop_name(crop_value: Any) -> Optional[str]:
    """Normalize a crop id/name to the registry display name, preserving unknown labels."""
    if crop_value is None:
        return None

    raw = str(crop_value).strip()
    if not raw:
        return None

    crop_spec = get_crop_registry().get_crop(raw)
    return crop_spec.display_name if crop_spec is not None else raw


def _resolve_canonical_crop(diag_res: Dict[str, Any], requested_crop: str) -> str:
    """
    Resolve the crop that was actually diagnosed.

    Precedence:
      1. Pipeline detected_crop (authoritative for both auto-detect and manual overrides).
      2. Routing decision target_crop.
      3. Requested crop, only as a safe backward-compatible fallback.
    """
    if not isinstance(diag_res, dict):
        diag_res = {}

    detected_crop = _canonical_crop_name(diag_res.get("detected_crop"))
    if detected_crop:
        return detected_crop

    routing_decision = diag_res.get("routing_decision")
    target_crop = None
    if isinstance(routing_decision, dict):
        target_crop = routing_decision.get("target_crop") or routing_decision.get("crop")
    elif routing_decision is not None:
        target_crop = (
            getattr(routing_decision, "target_crop", None)
            or getattr(routing_decision, "crop", None)
        )

    routed_crop = _canonical_crop_name(target_crop)
    if routed_crop:
        return routed_crop

    return _canonical_crop_name(requested_crop) or "Potato"



@dataclass
class IntegratedCropDoctorResult:
    """
    Complete structured payload for the Integrated AI Crop Doctor Intelligence Layer.
    """
    crop: str
    disease: str
    confidence: float
    class_probabilities: Dict[str, float]
    is_low_confidence: bool
    gradcam_output_path: Optional[str]
    weather_summary: Dict[str, Any]
    important_weather_factors: List[str]
    weather_available: bool
    risk_level: str
    risk_index: float
    risk_factors: List[str]
    risk_explanation: str
    what_if: Optional[Dict[str, Any]]
    rag_query: str
    grounded_answer: str
    grounding_status: bool
    insufficient_context: bool
    gemini_available: bool
    retrieved_sources: List[Dict[str, Any]]
    disclaimers: List[str]
    treatment_recommendation: Optional[Dict[str, Any]] = None
    is_rejected: bool = False
    prediction_status: str = "SUPPORTED_CONFIDENT"
    warning: Optional[str] = None
    requested_crop: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "DIAGNOSIS": {
                "crop": self.crop,
                "requested_crop": self.requested_crop,
                "disease": self.disease,
                "confidence": self.confidence,
                "class_probabilities": self.class_probabilities,
                "is_low_confidence": self.is_low_confidence,
                "is_rejected": self.is_rejected,
                "prediction_status": self.prediction_status,
                "warning": self.warning or (
                    "Inconclusive / Rejected Prediction (<40% confidence or invalid input). Chemical treatments suppressed."
                    if self.is_rejected
                    else ("Low Diagnostic Certainty (<60%). Chemical treatments withheld." if self.is_low_confidence else None)
                )
            },
            "TREATMENT": self.treatment_recommendation,
            "EXPLAINABILITY": {
                "gradcam_output_path": self.gradcam_output_path
            },
            "ENVIRONMENT": {
                "weather_summary": self.weather_summary,
                "important_weather_factors": self.important_weather_factors,
                "weather_available": self.weather_available,
                "warning": None if self.weather_available else "Live weather service unavailable; microclimate risk calculation offline."
            },
            "RISK": {
                "risk_level": self.risk_level,
                "risk_index": self.risk_index,
                "risk_factors": self.risk_factors,
                "risk_explanation": self.risk_explanation
            },
            "WHAT_IF": self.what_if,
            "KNOWLEDGE": {
                "generated_query": self.rag_query,
                "grounded_answer": self.grounded_answer,
                "grounding_status": self.grounding_status,
                "insufficient_context": self.insufficient_context,
                "gemini_available": self.gemini_available,
                "retrieved_sources": self.retrieved_sources
            },
            "DISCLAIMERS": self.disclaimers
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class CropDoctorIntelligenceLayer:
    """
    Orchestrates existing CropDoctor Diagnostic Pipeline, ScenarioSimulator, and RAGAnswerer
    into a unified intelligence layer.
    """
    def __init__(
        self,
        pipeline: Optional[CropDoctor] = None,
        simulator: Optional[ScenarioSimulator] = None,
        answerer: Optional[RAGAnswerer] = None,
        treatment_recommender: Optional[PotatoTreatmentRecommender] = None
    ):
        self.pipeline = pipeline if pipeline is not None else CropDoctor()
        self.simulator = simulator if simulator is not None else ScenarioSimulator()
        self.answerer = answerer if answerer is not None else RAGAnswerer()
        retriever = getattr(self.answerer, "retriever", None)
        self.treatment_recommender = (
            treatment_recommender
            if treatment_recommender is not None
            else PotatoTreatmentRecommender(retriever=retriever)
        )

    def analyze(
        self,
        image_path: str,
        latitude: float,
        longitude: float,
        crop: str = "Potato",
        scenario_params: Optional[Dict[str, float]] = None
    ) -> IntegratedCropDoctorResult:
        """
        Execute full end-to-end intelligence diagnosis, optional What-If simulation,
        RAG knowledge retrieval, and grounded Gemini answering.
        """
        requested_crop = crop

        # Step 1: Execute primary diagnostic pipeline
        diag_res_raw = self.pipeline.diagnose(image_path, latitude, longitude, crop=crop)
        diag_res = diag_res_raw.to_dict() if hasattr(diag_res_raw, "to_dict") else diag_res_raw

        # The pipeline's detected/routed crop is authoritative. The user's requested
        # crop is retained separately for audit and is used only as a safe fallback.
        canonical_crop = _resolve_canonical_crop(diag_res, requested_crop)


        # Step 2: Process What-If Simulation if scenario parameters provided
        scenario_comp = None
        if scenario_params:
            try:
                # Format scenario parameters for ScenarioSimulator.simulate()
                weather_dict = diag_res.get("weather_summary", {})
                scenario_comp = self.simulator.simulate(
                    disease_prediction=diag_res,
                    weather_data=weather_dict,
                    delta_temperature_c=scenario_params.get("delta_temperature_c", 0.0),
                    override_relative_humidity_pct=scenario_params.get("override_relative_humidity_pct"),
                    delta_relative_humidity_pct=scenario_params.get("delta_relative_humidity_pct", 0.0),
                    delta_precipitation_mm=scenario_params.get("delta_precipitation_mm", 0.0)
                )
                scenario_comp = scenario_comp.to_dict() if hasattr(scenario_comp, "to_dict") else scenario_comp
            except Exception as e:
                scenario_comp = None

        # Step 3: Build Diagnostic Context
        context: DiagnosticContext = ContextBuilder.build_context(
            diagnostic_result=diag_res,
            scenario_comparison=scenario_comp,
            crop=canonical_crop,
            requested_crop=requested_crop
        )

        # Step 4: Generate dynamic RAG Query & Metadata Filters
        rag_query, crop_filter, disease_filter = RAGQueryGenerator.generate_query(context)

        # Step 5: Execute RAG Answerer with grounded Gemini inference
        gemini_available = True
        try:
            rag_ans: RAGAnswer = self.answerer.answer_question(
                query=rag_query,
                crop=crop_filter,
                disease=disease_filter
            )
        except Exception as exc:
            # Handle RAG/Gemini service exceptions gracefully
            gemini_available = False
            rag_ans = RAGAnswer(
                query=rag_query,
                answer=f"Gemini natural-language knowledge explanation unavailable ({exc}).",
                sources=[],
                is_grounded=False,
                insufficient_context=True,
                disclaimer=STANDARD_DISCLAIMERS[2]
            )

        # Step 6: Generate Grounded Treatment / Medicine Recommendation
        try:
            treatment_res: TreatmentRecommendationResult = self.treatment_recommender.recommend_treatment(
                crop=canonical_crop,
                disease=context.disease,
                risk_level=context.risk_level,
                is_low_confidence=context.is_low_confidence,
                is_rejected=getattr(context, "is_rejected", False)
            )
            treatment_payload = treatment_res.to_dict()
        except Exception as exc:
            treatment_payload = {
                "crop": canonical_crop,
                "predicted_disease": context.disease,
                "is_supported": False,
                "is_healthy": False,
                "status": "TREATMENT_ERROR",
                "summary_message": f"Treatment recommendation error ({exc}). Consult local agricultural officer.",
                "chemical_treatments": [],
                "non_chemical_control_measures": [],
                "safety_and_environmental_precautions": [],
                "evidence_sources": [],
                "evidence_gaps": [str(exc)],
                "disclaimer": STANDARD_DISCLAIMERS[3]
            }

        # Step 7: Assemble Integrated Result
        return IntegratedCropDoctorResult(
            crop=canonical_crop,
            requested_crop=requested_crop,
            disease=context.disease,
            confidence=context.confidence,
            class_probabilities=context.class_probabilities,
            is_low_confidence=context.is_low_confidence,
            is_rejected=getattr(context, "is_rejected", False),
            prediction_status=getattr(context, "prediction_status", "SUPPORTED_CONFIDENT"),
            warning=getattr(context, "warning", None),
            gradcam_output_path=context.gradcam_path,
            weather_summary=context.weather_summary,
            important_weather_factors=context.risk_factors,
            weather_available=context.weather_available,
            risk_level=context.risk_level,
            risk_index=context.risk_index,
            risk_factors=context.risk_factors,
            risk_explanation=diag_res.get("explanation", ""),
            what_if=context.what_if_scenario,
            rag_query=rag_query,
            grounded_answer=rag_ans.answer,
            grounding_status=rag_ans.is_grounded,
            insufficient_context=rag_ans.insufficient_context,
            gemini_available=gemini_available,
            retrieved_sources=rag_ans.sources,
            disclaimers=STANDARD_DISCLAIMERS,
            treatment_recommendation=treatment_payload
        )
