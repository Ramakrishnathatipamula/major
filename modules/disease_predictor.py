import sys
from pathlib import Path
from typing import Dict, Union, Any
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

class DiseasePredictor:
    """
    Reusable Disease Predictor Engine for Potato Plant Leaf Diagnosis.
    Loads trained ResNet-18 model checkpoint and runs single-image inference.
    """
    def __init__(
        self, 
        model_path: Union[str, Path, None] = None, 
        device: torch.device = None
    ):
        self.device = device if device is not None else get_device()
        
        if model_path is None:
            model_path = settings.BASE_DIR / "models" / "disease" / "potato_disease_model.pth"
            
        self.model_path = Path(model_path)
        self.classes = DEFAULT_CLASSES
        self.friendly_names = FRIENDLY_NAMES
        
        # Preprocessing transformation pipeline matching training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        # Load model weights
        self.model = self._load_model()
        
    def _load_model(self) -> torch.nn.Module:
        """Internal helper to instantiate architecture and load trained weights."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {self.model_path}")
            
        model = build_resnet_model(num_classes=len(self.classes), pretrained=False)
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_input: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """
        Run disease detection inference on a single image.
        
        Args:
            image_input: File path (str/Path) or PIL Image object.
            
        Returns:
            Dictionary containing:
            - 'predicted_disease': String class name
            - 'friendly_name': Human-readable disease name
            - 'confidence': Confidence score float (0-100)
            - 'probabilities': Dict of friendly name -> percentage float
            - 'raw_probabilities': Dict of exact class name -> percentage float
        """
        if isinstance(image_input, (str, Path)):
            image_path = Path(image_input)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("Input must be a valid file path or PIL Image object.")
            
        # Apply preprocessing
        img_tensor = self.transform(image).unsqueeze(0).to(self.device) # Shape: [1, 3, 224, 224]
        
        # Run inference using torch.no_grad()
        with torch.no_grad():
            outputs = self.model(img_tensor)
            softmax_probs = torch.softmax(outputs, dim=1)[0]
            
        confidence, predicted_idx = torch.max(softmax_probs, dim=0)
        predicted_class = self.classes[predicted_idx.item()]
        friendly_label = self.friendly_names.get(predicted_class, predicted_class)
        confidence_pct = float(confidence.item() * 100.0)
        
        # Map probabilities
        raw_probs = {}
        friendly_probs = {}
        for idx, cls_name in enumerate(self.classes):
            prob_pct = float(softmax_probs[idx].item() * 100.0)
            raw_probs[cls_name] = prob_pct
            friendly_key = self.friendly_names.get(cls_name, cls_name)
            friendly_probs[friendly_key] = prob_pct
            
        return {
            "predicted_disease": predicted_class,
            "friendly_name": friendly_label,
            "confidence": confidence_pct,
            "probabilities": friendly_probs,
            "raw_probabilities": raw_probs
        }
