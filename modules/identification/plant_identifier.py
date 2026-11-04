import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import settings
from modules.model import build_resnet_model, get_device
from modules.dataset import IMAGENET_MEAN, IMAGENET_STD
from modules.crops import get_crop_registry
from modules.image_utils import (
    validate_and_load_image,
    get_preprocessing_transform,
    DEFAULT_LETTERBOX_FILL
)

# Supported 15 plant species: 14 PlantVillage species plus Wheat, which is sourced
# from a separate public field dataset because PlantVillage has no Wheat class.
PLANT_SPECIES_CLASSES: List[str] = [
    "apple",
    "bell_pepper",
    "blueberry",
    "cherry",
    "grape",
    "maize",
    "orange",
    "peach",
    "potato",
    "raspberry",
    "soybean",
    "squash",
    "strawberry",
    "tomato",
    "wheat",
]

PLANT_SPECIES_DISPLAY_NAMES: Dict[str, str] = {
    "apple": "Apple",
    "bell_pepper": "Pepper, Bell",
    "blueberry": "Blueberry",
    "cherry": "Cherry",
    "grape": "Grape",
    "maize": "Maize",
    "orange": "Orange",
    "peach": "Peach",
    "potato": "Potato",
    "raspberry": "Raspberry",
    "soybean": "Soybean",
    "squash": "Squash",
    "strawberry": "Strawberry",
    "tomato": "Tomato",
    "wheat": "Wheat",
}

# Provisional routing and certainty thresholds (represented in percentage units 0.0 to 100.0)
# NOTE: Softmax confidence on closed-set data is provisional until calibrated against open-domain field images.
CONFIDENCE_ROUTING_THRESHOLD: float = 60.0    # Minimum confidence to safely route to specialist
CONFIDENCE_REJECTION_THRESHOLD: float = 40.0  # Threshold near random-guess boundary for 15 classes (~6.7%)

# Feature prototype cosine similarity thresholds [-1.0 to 1.0]
# Derived strictly from training split embeddings (data/identification/splits/train.json).
# Validation leaves exhibit mean similarity 0.910 (min 0.719), whereas non-plant/OOD samples exhibit < 0.70.
PROTOTYPE_SIM_ROUTING_THRESHOLD: float = 0.80   # Minimum prototype similarity for confident routing
PROTOTYPE_SIM_REJECTION_THRESHOLD: float = 0.70 # Minimum similarity below which sample is rejected as OOD/Unknown

# Free energy score thresholds T * logsumexp(logits / T)
# In-distribution validation leaves have mean energy 7.34 (min 3.57).
ENERGY_ROUTING_THRESHOLD: float = 4.5
ENERGY_REJECTION_THRESHOLD: float = 3.5
DEFAULT_TEMPERATURE: float = 1.0


@dataclass
class PlantIdentificationResult:
    """
    Structured outcome of Stage 1 plant identification inference.
    Includes open-set rejection metrics (energy score, prototype similarity).
    """
    crop_id: str
    display_name: str
    confidence: float  # Percentage 0.0 to 100.0
    probabilities: Dict[str, float]
    is_supported: bool
    is_ambiguous: bool
    is_rejected: bool
    status: str        # CONFIDENT_IDENTIFICATION, AMBIGUOUS_SPECIES, REJECTED_LOW_CONFIDENCE, REJECTED_UNKNOWN_SPECIES, INVALID_INPUT, MODEL_UNAVAILABLE
    warning: Optional[str] = None
    energy_score: Optional[float] = None
    prototype_similarity: Optional[float] = None
    prototype_distance: Optional[float] = None
    prototype_crop: Optional[str] = None
    ood_status: str = "IN_DISTRIBUTION"  # IN_DISTRIBUTION, AMBIGUOUS_OOD, REJECTED_OOD, NOT_EVALUATED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crop_id": self.crop_id,
            "display_name": self.display_name,
            "confidence": round(self.confidence, 2),
            "probabilities": {k: round(v, 2) for k, v in self.probabilities.items()},
            "is_supported": self.is_supported,
            "is_ambiguous": self.is_ambiguous,
            "is_rejected": self.is_rejected,
            "status": self.status,
            "warning": self.warning,
            "energy_score": round(self.energy_score, 3) if self.energy_score is not None else None,
            "prototype_similarity": round(self.prototype_similarity, 4) if self.prototype_similarity is not None else None,
            "prototype_distance": round(self.prototype_distance, 4) if self.prototype_distance is not None else None,
            "prototype_crop": self.prototype_crop,
            "ood_status": self.ood_status,
        }


class PlantIdentifier:
    """
    Stage 1 Plant Identification Engine.
    Uses fine-tuned ResNet-18 across 15 cultivated plant species to identify species
    before routing to disease specialists.

    Includes Open-Set Rejection via:
    1. Feature prototype cosine distance against training split centroids.
    2. Logit free-energy scoring for out-of-distribution detection.
    3. Multi-metric gating to reject unfamiliar wild plants and non-crop foliage.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        prototypes_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        confidence_routing_threshold: float = CONFIDENCE_ROUTING_THRESHOLD,
        confidence_rejection_threshold: float = CONFIDENCE_REJECTION_THRESHOLD,
        prototype_sim_routing_threshold: float = PROTOTYPE_SIM_ROUTING_THRESHOLD,
        prototype_sim_rejection_threshold: float = PROTOTYPE_SIM_REJECTION_THRESHOLD,
        energy_routing_threshold: float = ENERGY_ROUTING_THRESHOLD,
        energy_rejection_threshold: float = ENERGY_REJECTION_THRESHOLD,
        temperature: float = DEFAULT_TEMPERATURE,
        preprocessing_mode: Optional[str] = None,
    ):
        self.device = device if device is not None else get_device()
        self.crop_registry = get_crop_registry()

        if model_path is None:
            model_path = ROOT_DIR / "models" / "identification" / "plant_id_model_15class.pth"
        self.model_path = Path(model_path)

        if prototypes_path is None:
            prototypes_path = ROOT_DIR / "models" / "identification" / "crop_prototypes_15class.pt"
        self.prototypes_path = Path(prototypes_path)

        self.classes = PLANT_SPECIES_CLASSES
        self.display_names = PLANT_SPECIES_DISPLAY_NAMES

        # Configurable thresholds
        self.confidence_routing_threshold = confidence_routing_threshold
        self.confidence_rejection_threshold = confidence_rejection_threshold
        self.prototype_sim_routing_threshold = prototype_sim_routing_threshold
        self.prototype_sim_rejection_threshold = prototype_sim_rejection_threshold
        self.energy_routing_threshold = energy_routing_threshold
        self.energy_rejection_threshold = energy_rejection_threshold
        self.temperature = temperature

        # Reversible preprocessing transform
        self.preprocessing_mode = preprocessing_mode or getattr(settings, "DEFAULT_PREPROCESSING_MODE", "letterbox")
        self.transform = get_preprocessing_transform(
            mode=self.preprocessing_mode,
            target_size=(224, 224),
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )

        self.model = self._load_model() if self.model_path.exists() else None
        self.backbone = self._build_backbone() if self.model is not None else None
        self.proto_matrix, self.prototypes_dict = self._load_prototypes()

    def _load_model(self) -> Optional[nn.Module]:
        """Load trained plant identification weights if present."""
        if not self.model_path.exists():
            return None
        try:
            model = build_resnet_model(num_classes=len(self.classes), pretrained=False)
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            return model
        except Exception as exc:
            print(f"Warning: Failed to load PlantIdentifier model weights: {exc}")
            return None

    def _build_backbone(self) -> Optional[nn.Module]:
        """Construct feature extractor backbone (layers up to avgpool) for embedding extraction."""
        if self.model is None:
            return None
        modules = list(self.model.children())
        backbone = nn.Sequential(*modules[:-1]).to(self.device)
        backbone.eval()
        return backbone

    def _load_prototypes(self) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Load training-derived feature prototype centroids.
        Derived exclusively from data/identification/splits/train.json.
        """
        if not self.prototypes_path.exists():
            return None, {}
        try:
            payload = torch.load(self.prototypes_path, map_location=self.device)
            proto_matrix = payload["proto_matrix"].to(self.device)
            prototypes_dict = {k: v.to(self.device) for k, v in payload.get("prototypes", {}).items()}
            return proto_matrix, prototypes_dict
        except Exception as exc:
            print(f"Warning: Failed to load crop prototypes from {self.prototypes_path}: {exc}")
            return None, {}

    def is_model_available(self) -> bool:
        """Return True if model checkpoint exists and is loaded."""
        return self.model is not None

    def identify(self, image_input: Union[str, Path, Image.Image]) -> PlantIdentificationResult:
        """
        Identify plant species from an input image with open-set rejection.

        Args:
            image_input: File path or PIL Image.

        Returns:
            PlantIdentificationResult containing identified species, confidence,
            energy score, and prototype similarity.
        """
        # Validate and load image
        is_valid, err_msg, loaded_image = validate_and_load_image(image_input)
        if not is_valid or loaded_image is None:
            return PlantIdentificationResult(
                crop_id="unknown",
                display_name="Unknown",
                confidence=0.0,
                probabilities={},
                is_supported=False,
                is_ambiguous=True,
                is_rejected=True,
                status="INVALID_INPUT",
                warning=f"Input image invalid: {err_msg}",
                energy_score=None,
                prototype_similarity=None,
                prototype_distance=None,
                prototype_crop=None,
                ood_status="REJECTED_OOD"
            )

        # Check model availability
        if self.model is None:
            return PlantIdentificationResult(
                crop_id="potato",  # Fallback default
                display_name="Potato",
                confidence=0.0,
                probabilities={},
                is_supported=True,
                is_ambiguous=True,
                is_rejected=False,
                status="MODEL_UNAVAILABLE",
                warning="Plant identification model weights not found on disk. Defaulting to Potato with provisional status.",
                energy_score=None,
                prototype_similarity=None,
                prototype_distance=None,
                prototype_crop=None,
                ood_status="NOT_EVALUATED"
            )

        # Transform and forward pass
        img_tensor = self.transform(loaded_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            # Energy-based OOD score: T * logsumexp(outputs / T)
            energy_score = float((self.temperature * torch.logsumexp(outputs / self.temperature, dim=1)[0]).item())

        top_val, top_idx = torch.max(probs, dim=0)
        predicted_crop_id = self.classes[top_idx.item()]
        predicted_display = self.display_names.get(predicted_crop_id, predicted_crop_id.capitalize())
        confidence_pct = round(float(top_val.item() * 100.0), 2)

        # Feature prototype cosine similarity extraction
        proto_sim: Optional[float] = None
        proto_dist: Optional[float] = None
        proto_crop: Optional[str] = None

        if self.backbone is not None and self.proto_matrix is not None:
            with torch.no_grad():
                feat = torch.flatten(self.backbone(img_tensor), 1)
                feat = feat / torch.norm(feat, p=2, dim=1, keepdim=True)
                sims = torch.mm(feat, self.proto_matrix.t())[0]
                max_sim, max_proto_idx = torch.max(sims, dim=0)
                proto_sim = round(float(max_sim.item()), 4)
                proto_dist = round(float(1.0 - proto_sim), 4)
                proto_crop = self.classes[max_proto_idx.item()]

        # Map all probabilities
        prob_dict = {}
        for idx, crop_cls in enumerate(self.classes):
            display = self.display_names.get(crop_cls, crop_cls)
            prob_dict[display] = round(float(probs[idx].item() * 100.0), 2)

        # Crop support status in CropRegistry
        crop_spec = self.crop_registry.get_crop(predicted_crop_id)
        is_crop_supported = bool(crop_spec and crop_spec.is_supported)

        # Multi-metric Open-Set Rejection & Ambiguity Evaluation
        # 1. Low confidence rejection (< 40%)
        is_conf_rejected = confidence_pct < self.confidence_rejection_threshold

        # 2. Prototype distance rejection (< 0.70 similarity)
        is_proto_rejected = (proto_sim is not None) and (proto_sim < self.prototype_sim_rejection_threshold)

        # 3. Energy score rejection (< 3.5)
        is_energy_rejected = energy_score < self.energy_rejection_threshold

        # Composite rejection
        is_rejected = is_conf_rejected or is_proto_rejected or is_energy_rejected

        # Ambiguity conditions
        is_conf_ambiguous = confidence_pct < self.confidence_routing_threshold
        is_proto_ambiguous = (proto_sim is not None) and (proto_sim < self.prototype_sim_routing_threshold)
        is_energy_ambiguous = energy_score < self.energy_routing_threshold
        is_proto_mismatch = (proto_crop is not None) and (proto_crop != predicted_crop_id)

        is_ambiguous = is_conf_ambiguous or is_proto_ambiguous or is_energy_ambiguous or is_proto_mismatch

        if is_rejected:
            ood_status = "REJECTED_OOD"
            if is_proto_rejected:
                status = "REJECTED_UNKNOWN_SPECIES"
                warning = (
                    f"Out-of-Distribution Rejection: Feature prototype similarity ({proto_sim:.3f}) is below "
                    f"acceptance threshold ({self.prototype_sim_rejection_threshold:.2f}). "
                    "Image does not match any recognized agricultural crop manifold. Rejecting as unfamiliar/wild species."
                )
            elif is_energy_rejected:
                status = "REJECTED_LOW_CONFIDENCE"
                warning = (
                    f"Out-of-Distribution Rejection: Energy score ({energy_score:.2f}) is below threshold ({self.energy_rejection_threshold:.2f}). "
                    "Inconclusive species representation."
                )
            else:
                status = "REJECTED_LOW_CONFIDENCE"
                warning = (
                    f"Species identification rejected: Softmax confidence ({confidence_pct:.1f}%) is below minimum threshold ({self.confidence_rejection_threshold}%). "
                    "Prediction is near random guess across 15 plant classes."
                )
        elif is_ambiguous:
            ood_status = "AMBIGUOUS_OOD"
            status = "AMBIGUOUS_SPECIES"
            if is_proto_mismatch:
                warning = (
                    f"Ambiguous botanical classification: Model logits favor {predicted_display} ({confidence_pct:.1f}%), "
                    f"but feature prototype embedding is nearest to {self.display_names.get(proto_crop, proto_crop)} (sim: {proto_sim:.3f}). "
                    "Foliage certainty is marginal."
                )
            else:
                warning = (
                    f"Ambiguous species identification ({confidence_pct:.1f}% confidence, prototype sim: {proto_sim if proto_sim is not None else 'N/A'}). "
                    f"Foliage resembles {predicted_display}, but certainty is marginal."
                )
        else:
            ood_status = "IN_DISTRIBUTION"
            status = "CONFIDENT_IDENTIFICATION"
            warning = None

        return PlantIdentificationResult(
            crop_id=predicted_crop_id,
            display_name=predicted_display,
            confidence=confidence_pct,
            probabilities=prob_dict,
            is_supported=is_crop_supported,
            is_ambiguous=is_ambiguous,
            is_rejected=is_rejected,
            status=status,
            warning=warning,
            energy_score=energy_score,
            prototype_similarity=proto_sim,
            prototype_distance=proto_dist,
            prototype_crop=proto_crop,
            ood_status=ood_status
        )
