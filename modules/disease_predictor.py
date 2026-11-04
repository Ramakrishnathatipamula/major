import os
import sys
from pathlib import Path
from typing import Dict, Union, Any, Optional
import torch
from torchvision import transforms
from PIL import Image

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
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

# Configurable confidence policy thresholds (represented in percentage units 0.0 to 100.0)
# NOTE: Softmax confidence reflects output distribution across trained classes, NOT a guarantee of correctness.
CONFIDENCE_WARNING_THRESHOLD: float = 60.0    # Marginal certainty threshold (60.0%)
CONFIDENCE_REJECTION_THRESHOLD: float = 40.0  # Near random-guess boundary for 3 classes (40.0%)

DEFAULT_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

FRIENDLY_NAMES = {
    "Potato___Early_blight": "Early Blight",
    "Potato___Late_blight": "Late Blight",
    "Potato___healthy": "Healthy"
}

# Tomato specialist class list (10 PlantVillage classes, alphabetically sorted per training manifest)
TOMATO_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

TOMATO_FRIENDLY_NAMES = {
    "Tomato___Bacterial_spot": "Bacterial Spot",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___Leaf_Mold": "Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spider Mites",
    "Tomato___Target_Spot": "Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus": "Mosaic Virus",
    "Tomato___healthy": "Healthy"
}

class DiseasePredictor:
    """
    Reusable Multi-Crop Disease Predictor Engine.
    Loads trained model checkpoints for supported crops (Potato, Tomato, Maize, Wheat)
    and runs single-image inference. Gracefully reports support status for unsupported
    crops and missing checkpoints.
    """
    def __init__(
        self, 
        model_path: Union[str, Path, None] = None, 
        device: torch.device = None,
        preprocessing_mode: Optional[str] = None,
        use_field_adapted_models: Optional[bool] = None
    ):
        self.device = device if device is not None else get_device()
        self.crop_registry = get_crop_registry()
        
        env_field_adapt = os.getenv("USE_FIELD_ADAPTED_MODELS", "0").lower() in ("1", "true", "yes")
        self.use_field_adapted_models = (
            use_field_adapted_models if use_field_adapted_models is not None else env_field_adapt
        )

        self.custom_model_path = Path(model_path) if model_path is not None else None
        
        # Resolve initial model path (experimental or baseline)
        if self.custom_model_path is not None:
            self.model_path = self.custom_model_path
        elif self.use_field_adapted_models:
            adapted_potato = settings.BASE_DIR / "models" / "disease" / "potato_field_adapted_model.pth"
            self.model_path = adapted_potato if adapted_potato.exists() else (
                settings.BASE_DIR / "models" / "disease" / "potato_disease_model.pth"
            )
        else:
            self.model_path = settings.BASE_DIR / "models" / "disease" / "potato_disease_model.pth"
        
        self.classes = DEFAULT_CLASSES
        self.friendly_names = FRIENDLY_NAMES
        
        # Preprocessing transformation pipeline (reversible between letterbox and baseline)
        self.preprocessing_mode = preprocessing_mode or getattr(settings, "DEFAULT_PREPROCESSING_MODE", "letterbox")
        self.transform = get_preprocessing_transform(
            mode=self.preprocessing_mode,
            target_size=(224, 224),
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
        
        # Internal cache of loaded specialist models
        self._models: Dict[str, torch.nn.Module] = {}
        
        # Backward compatibility: load default potato model if present
        if self.model_path.exists():
            try:
                self.model = self._load_model_from_path(self.model_path, num_classes=len(self.classes))
                self._models["potato"] = self.model
            except Exception:
                self.model = None
        else:
            self.model = None

    def _load_model_from_path(self, path: Path, num_classes: int) -> torch.nn.Module:
        """Internal helper to instantiate architecture and load trained weights."""
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {path}")
            
        model = build_resnet_model(num_classes=num_classes, pretrained=False)
        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _load_model(self) -> torch.nn.Module:
        """Legacy helper for backward compatibility."""
        return self._load_model_from_path(self.model_path, len(self.classes))

    def get_model(self, crop: str = "Potato") -> Optional[torch.nn.Module]:
        """
        Retrieve or lazily load the specialist model for a specified crop.
        """
        crop_spec = self.crop_registry.get_crop(crop)
        if not crop_spec:
            return None
            
        c_id = crop_spec.crop_id.lower()
        if c_id in self._models:
            return self._models[c_id]
            
        # Check if custom model path was provided on initialization
        if self.custom_model_path is not None and c_id == "potato":
            chk_path = self.custom_model_path
        elif self.use_field_adapted_models:
            field_adapted_path = settings.BASE_DIR / "models" / "disease" / f"{c_id}_field_adapted_model.pth"
            if field_adapted_path.exists():
                chk_path = field_adapted_path
            else:
                chk_path = crop_spec.model_file_path
        else:
            chk_path = crop_spec.model_file_path
            
        if chk_path is None or not chk_path.exists():
            return None
            
        num_classes = len(crop_spec.supported_diseases)
        try:
            loaded = self._load_model_from_path(chk_path, num_classes=num_classes)
            self._models[c_id] = loaded
            return loaded
        except Exception:
            return None

    def predict(
        self, 
        image_input: Union[str, Path, Image.Image],
        crop: str = "Potato"
    ) -> Dict[str, Any]:
        """
        Run disease detection inference for the specified crop.
        
        Args:
            image_input: File path (str/Path) or PIL Image object.
            crop: Target crop name or ID (default: "Potato")
            
        Returns:
            Dictionary containing prediction metadata, confidence, probabilities, and crop support status.
        """
        crop_spec = self.crop_registry.get_crop(crop)
        crop_display = crop_spec.display_name if crop_spec else str(crop)

        # Handle unsupported crops gracefully
        if not crop_spec or not crop_spec.is_supported:
            msg = crop_spec.support_status_message if crop_spec else f"Crop '{crop}' disease model is not currently active."
            return {
                "crop": crop_display,
                "predicted_disease": f"{crop_display}___unsupported",
                "friendly_name": f"{crop_display} (Model Pending)",
                "confidence": 0.0,
                "probabilities": {},
                "raw_probabilities": {},
                "is_supported": False,
                "is_low_confidence": True,
                "is_rejected": True,
                "prediction_status": "UNSUPPORTED_CROP",
                "warning": msg
            }

        # Retrieve specialist model
        model = self.get_model(crop)
        if model is None:
            msg = f"Specialist checkpoint for {crop_display} is missing ({crop_spec.model_path})."
            return {
                "crop": crop_display,
                "predicted_disease": f"{crop_display}___missing_checkpoint",
                "friendly_name": f"{crop_display} (Checkpoint Missing)",
                "confidence": 0.0,
                "probabilities": {},
                "raw_probabilities": {},
                "is_supported": False,
                "is_low_confidence": True,
                "is_rejected": True,
                "prediction_status": "MISSING_CHECKPOINT",
                "warning": msg
            }

        # Safe image validation and loading
        is_valid, err_msg, loaded_image = validate_and_load_image(image_input)
        if not is_valid or loaded_image is None:
            return {
                "crop": crop_display,
                "predicted_disease": f"{crop_display}___invalid_input",
                "friendly_name": f"{crop_display} (Invalid Image)",
                "confidence": 0.0,
                "probabilities": {},
                "raw_probabilities": {},
                "is_supported": False,
                "is_low_confidence": True,
                "is_rejected": True,
                "prediction_status": "INVALID_INPUT",
                "warning": f"Image validation failed: {err_msg}"
            }

        # Resolve crop-specific classes and friendly names
        active_classes = crop_spec.supported_diseases if crop_spec.supported_diseases else self.classes
        active_friendly = crop_spec.disease_friendly_names if crop_spec.disease_friendly_names else self.friendly_names

        # Apply preprocessing
        img_tensor = self.transform(loaded_image).unsqueeze(0).to(self.device) # Shape: [1, 3, 224, 224]

        # Run inference using torch.no_grad()
        with torch.no_grad():
            outputs = model(img_tensor)
            softmax_probs = torch.softmax(outputs, dim=1)[0]

        confidence, predicted_idx = torch.max(softmax_probs, dim=0)
        predicted_class = active_classes[predicted_idx.item()]
        friendly_label = active_friendly.get(predicted_class, predicted_class)
        # Consistent percentage representation [0.0, 100.0]
        confidence_pct = round(float(confidence.item() * 100.0), 2)

        # Map probabilities
        raw_probs = {}
        friendly_probs = {}
        for idx, cls_name in enumerate(active_classes):
            prob_pct = round(float(softmax_probs[idx].item() * 100.0), 2)
            raw_probs[cls_name] = prob_pct
            friendly_key = active_friendly.get(cls_name, cls_name)
            friendly_probs[friendly_key] = prob_pct

        # Apply configurable confidence policy
        is_rejected = confidence_pct < CONFIDENCE_REJECTION_THRESHOLD
        is_low_conf = confidence_pct < CONFIDENCE_WARNING_THRESHOLD

        if is_rejected:
            pred_status = "REJECTED_LOW_CONFIDENCE"
            warning_msg = (
                f"Prediction rejected: Diagnostic confidence ({confidence_pct:.1f}%) is below minimum threshold ({CONFIDENCE_REJECTION_THRESHOLD}%). "
                "Prediction is near random guess. Re-take leaf photo under clear lighting."
            )
        elif is_low_conf:
            pred_status = "SUPPORTED_UNCERTAIN"
            warning_msg = (
                f"Low Diagnostic Certainty ({confidence_pct:.1f}% < {CONFIDENCE_WARNING_THRESHOLD}%). "
                "Model confidence is marginal; inspect secondary canopy leaves before taking action."
            )
        else:
            pred_status = "SUPPORTED_CONFIDENT"
            warning_msg = None

        return {
            "crop": crop_display,
            "predicted_disease": predicted_class,
            "friendly_name": friendly_label,
            "confidence": confidence_pct,
            "probabilities": friendly_probs,
            "raw_probabilities": raw_probs,
            "is_supported": True,
            "is_low_confidence": is_low_conf,
            "is_rejected": is_rejected,
            "prediction_status": pred_status,
            "warning": warning_msg
        }

