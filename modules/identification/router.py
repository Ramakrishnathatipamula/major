import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
from PIL import Image

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.crops import get_crop_registry
from modules.identification.leaf_verifier import LeafVerifier, LeafVerificationResult
from modules.identification.plant_identifier import (
    PlantIdentifier,
    PlantIdentificationResult,
    CONFIDENCE_ROUTING_THRESHOLD,
    CONFIDENCE_REJECTION_THRESHOLD
)


@dataclass
class RoutingDecision:
    """
    Final Stage 1 decision payload dictating whether and how an image is routed to disease specialists.
    """
    target_crop: str
    display_name: str
    route_to_specialist: bool
    routing_status: str     # ROUTED_TO_POTATO_SPECIALIST, ROUTED_TO_TOMATO_SPECIALIST, UNSUPPORTED_CROP, REJECTED_NON_LEAF, UNKNOWN_SPECIES, INVALID_INPUT
    confidence: float       # Specialist/identification confidence; 0.0 when no model assessed the crop
    message: str
    leaf_verification: Optional[Dict[str, Any]] = None
    plant_identification: Optional[Dict[str, Any]] = None
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_crop": self.target_crop,
            "display_name": self.display_name,
            "route_to_specialist": self.route_to_specialist,
            "routing_status": self.routing_status,
            "confidence": round(self.confidence, 2),
            "message": self.message,
            "leaf_verification": self.leaf_verification,
            "plant_identification": self.plant_identification,
            "warning": self.warning
        }


class PlantRouter:
    """
    Stage 1 Router: Links Leaf Verification and Plant Identification.
    Safely gates inference to guarantee:
    1. Non-leaf or invalid images never reach disease models.
    2. Unsupported crops are identified without receiving fabricated disease or chemical advice.
    3. Only verified supported-crop images reach their disease specialist model.
    """

    def __init__(
        self,
        leaf_verifier: Optional[LeafVerifier] = None,
        plant_identifier: Optional[PlantIdentifier] = None,
        preprocessing_mode: Optional[str] = None
    ):
        self.leaf_verifier = leaf_verifier if leaf_verifier is not None else LeafVerifier()
        self.plant_identifier = (
            plant_identifier
            if plant_identifier is not None
            else PlantIdentifier(preprocessing_mode=preprocessing_mode)
        )
        self.crop_registry = get_crop_registry()

    def route(
        self,
        image_input: Union[str, Path, Image.Image],
        manual_crop: Optional[str] = None
    ) -> RoutingDecision:
        """
        Evaluate image and determine routing to specialist disease models.

        Args:
            image_input: File path or PIL Image.
            manual_crop: Optional user-selected crop override ("Auto-Detect", "Potato", "Tomato", etc.)

        Returns:
            RoutingDecision payload.
        """
        # Resolve a manual crop selection (if any) before verification so the
        # unsupported branch can reuse the verified leaf evidence.
        is_auto_detect = manual_crop is None or str(manual_crop).strip().lower() in ["auto-detect", "auto", "auto-detect (ai plant id)", ""]
        crop_spec = None
        crop_id = None
        crop_name = None
        if not is_auto_detect:
            crop_spec = self.crop_registry.get_crop(manual_crop)
            crop_id = crop_spec.crop_id if crop_spec else str(manual_crop).lower()
            crop_name = crop_spec.display_name if crop_spec else str(manual_crop)

        # Step 1: Front-line Leaf Verification (runs for every mode, including
        # manual selections, so no routing decision is made without examining
        # the uploaded image).
        leaf_res: LeafVerificationResult = self.leaf_verifier.verify_image(image_input)
        if not leaf_res.is_leaf:
            status = leaf_res.status if leaf_res.status in ["REJECTED_BLANK", "REJECTED_NON_LEAF"] else "INVALID_INPUT"
            return RoutingDecision(
                target_crop="unknown",
                display_name="Unknown",
                route_to_specialist=False,
                routing_status=status,
                confidence=leaf_res.confidence,
                message=leaf_res.reason,
                leaf_verification=leaf_res.to_dict(),
                plant_identification=None,
                warning=leaf_res.reason
            )

        # Manual selection of a crop without a production disease specialist.
        # The leaf has been verified above, but no model has assessed crop
        # identity or disease status, so report 0.0 confidence rather than a
        # fabricated 100.0.
        if not is_auto_detect and (crop_spec is None or not crop_spec.is_supported):
            msg = crop_spec.support_status_message if crop_spec else f"Crop '{crop_name}' disease specialist is currently pending development."
            return RoutingDecision(
                target_crop=crop_id,
                display_name=crop_name,
                route_to_specialist=False,
                routing_status="UNSUPPORTED_CROP",
                confidence=0.0,
                message=msg,
                leaf_verification=leaf_res.to_dict(),
                plant_identification=None,
                warning=msg
            )

        # Handle Manual Crop Selection for any registry-supported crop
        if not is_auto_detect and crop_spec and crop_spec.is_supported:
            plant_res_dict = None
            warning = None
            if self.plant_identifier.is_model_available():
                plant_res = self.plant_identifier.identify(image_input)
                plant_res_dict = plant_res.to_dict()
                if plant_res.crop_id != crop_id and plant_res.confidence >= 80.0:
                    warning = (
                        f"Potential Mismatch Notice: You selected {crop_name}, but AI Plant ID identified "
                        f"{plant_res.display_name} with {plant_res.confidence:.1f}% confidence. "
                        f"Proceeding with {crop_name} specialist as requested."
                    )

            specialist_status = f"ROUTED_TO_{crop_id.upper()}_SPECIALIST"
            return RoutingDecision(
                target_crop=crop_id,
                display_name=crop_name,
                route_to_specialist=True,
                routing_status=specialist_status,
                confidence=100.0,
                message=f"Manual selection: {crop_name} specialist pipeline active.",
                leaf_verification=leaf_res.to_dict(),
                plant_identification=plant_res_dict,
                warning=warning
            )

        # Step 2: Auto-Detect Mode — Run Stage 1 Plant Identification
        plant_res: PlantIdentificationResult = self.plant_identifier.identify(image_input)

        # Case A: Identification rejected, confidence < 40%, or OOD rejected
        if plant_res.is_rejected or plant_res.ood_status == "REJECTED_OOD":
            detail = f"Confidence: {plant_res.confidence:.1f}%"
            if plant_res.prototype_similarity is not None:
                detail += f", Prototype Sim: {plant_res.prototype_similarity:.3f}"
            if plant_res.energy_score is not None:
                detail += f", Energy: {plant_res.energy_score:.2f}"

            msg = (
                f"Unrecognized plant species: Model evidence is insufficient ({detail}). "
                "Cannot reliably identify crop or diagnose disease. Chemical treatments are safely withheld."
            )
            return RoutingDecision(
                target_crop="unknown",
                display_name="Unknown Plant",
                route_to_specialist=False,
                routing_status="UNKNOWN_SPECIES",
                confidence=plant_res.confidence,
                message=msg,
                leaf_verification=leaf_res.to_dict(),
                plant_identification=plant_res.to_dict(),
                warning=msg
            )

        # Case B: Identified as a crop with an active disease specialist.
        # This is registry-driven, so Potato/Tomato keep their existing status
        # strings and Maize/Wheat can route as soon as their checkpoints exist.
        identified_spec = self.crop_registry.get_crop(plant_res.crop_id)
        if identified_spec and identified_spec.is_supported:
            crop_id = identified_spec.crop_id
            crop_name = identified_spec.display_name
            is_confident = (
                not plant_res.is_ambiguous
                and plant_res.confidence >= CONFIDENCE_ROUTING_THRESHOLD
            )
            status_suffix = "" if is_confident else "_UNCERTAIN"
            if is_confident:
                message = (
                    f"Plant identified as {crop_name} ({plant_res.confidence:.1f}% confidence). "
                    f"Routing to {crop_name} disease specialist."
                )
                warning = None
            else:
                message = (
                    f"Marginal {crop_name} identification certainty "
                    f"({plant_res.confidence:.1f}% confidence). Routing to {crop_name} "
                    "specialist with uncertain status flag."
                )
                if plant_res.warning:
                    message = f"{message} Note: {plant_res.warning}"
                warning = message

            return RoutingDecision(
                target_crop=crop_id,
                display_name=crop_name,
                route_to_specialist=True,
                routing_status=f"ROUTED_TO_{crop_id.upper()}_SPECIALIST{status_suffix}",
                confidence=plant_res.confidence,
                message=message,
                leaf_verification=leaf_res.to_dict(),
                plant_identification=plant_res.to_dict(),
                warning=warning
            )

        # Case C: Identified as another recognized crop without a disease specialist
        crop_spec = self.crop_registry.get_crop(plant_res.crop_id)
        crop_name = plant_res.display_name
        support_msg = (
            f"Plant identified as {crop_name} ({plant_res.confidence:.1f}% confidence). "
            f"Disease diagnostic specialist for {crop_name} is currently under development. "
            "Chemical treatments are safely withheld."
        )

        return RoutingDecision(
            target_crop=plant_res.crop_id,
            display_name=crop_name,
            route_to_specialist=False,
            routing_status="UNSUPPORTED_CROP",
            confidence=plant_res.confidence,
            message=support_msg,
            leaf_verification=leaf_res.to_dict(),
            plant_identification=plant_res.to_dict(),
            warning=support_msg
        )
