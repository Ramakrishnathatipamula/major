import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import sys

# Add project root to path if needed
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.dataset import get_transforms

def get_device() -> torch.device:
    """Detect available compute device (MPS for Mac Apple Silicon, CUDA, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def build_resnet_model(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """
    Construct ResNet-18 architecture with transfer learning weights.
    
    Args:
        num_classes: Number of output classes (3 for Potato Early/Late Blight & Healthy).
        pretrained: Whether to load ImageNet pretrained backbone weights.
        
    Returns:
        PyTorch nn.Module instance ready for training or evaluation.
    """
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)
        
    # Replace final classification layer for our custom classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def load_trained_model(
    model_path: Path, 
    num_classes: int = 3, 
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Load a trained model checkpoint from disk.
    
    Args:
        model_path: Path to .pth checkpoint file.
        num_classes: Number of target classes.
        device: Target torch device (CPU/MPS/CUDA).
        
    Returns:
        Evaluated PyTorch nn.Module model.
    """
    if device is None:
        device = get_device()
        
    model = build_resnet_model(num_classes=num_classes, pretrained=False)
    
    path_obj = Path(model_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {path_obj}")
        
    state_dict = torch.load(path_obj, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model

def predict_image(
    model: nn.Module, 
    image_input, 
    class_names: List[str], 
    device: Optional[torch.device] = None
) -> Tuple[str, float, Dict[str, float]]:
    """
    Run disease classification inference on a single image.
    
    Args:
        model: Trained PyTorch classification model.
        image_input: PIL Image object or file path.
        class_names: List of class string names matching dataset indices.
        device: Torch computing device.
        
    Returns:
        Tuple of (predicted_class_name, confidence_percentage, class_probabilities_dict)
    """
    if device is None:
        device = get_device()
        
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")
        
    val_transform = get_transforms()["val"]
    img_tensor = val_transform(image).unsqueeze(0).to(device) # Shape: [1, 3, 224, 224]
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        
    confidence, predicted_idx = torch.max(probabilities, dim=0)
    predicted_class = class_names[predicted_idx.item()]
    
    probs_dict = {
        class_names[i]: float(probabilities[i].item() * 100) 
        for i in range(len(class_names))
    }
    
    return predicted_class, float(confidence.item() * 100), probs_dict
