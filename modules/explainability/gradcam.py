import sys
from pathlib import Path
from typing import Dict, Union, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.cm as cm

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import settings
from modules.dataset import IMAGENET_MEAN, IMAGENET_STD
from modules.disease_predictor import DiseasePredictor

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for ResNet-18.
    Visualizes the specific regions of a leaf photo that influenced the AI model's diagnosis.
    """
    def __init__(
        self, 
        predictor: Optional[DiseasePredictor] = None, 
        target_layer_name: str = "layer4"
    ):
        if predictor is None:
            self.predictor = DiseasePredictor()
        else:
            self.predictor = predictor
            
        self.model = self.predictor.model
        self.device = self.predictor.device
        
        # Target layer for ResNet-18 is layer4[-1]
        self.target_layer = self._get_target_layer(target_layer_name)
        
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _get_target_layer(self, layer_name: str) -> nn.Module:
        """Find target layer in ResNet-18."""
        if hasattr(self.model, layer_name):
            layer = getattr(self.model, layer_name)
            if isinstance(layer, nn.Sequential):
                return layer[-1] # Final residual block in layer4
            return layer
        else:
            raise ValueError(f"Layer '{layer_name}' not found in ResNet-18 model.")

    def _register_hooks(self):
        """Register forward and backward hooks to capture feature maps and gradients."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def _get_target_layer_for_model(self, model: nn.Module, layer_name: str) -> nn.Module:
        """Find target layer in ResNet-18 model."""
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            if isinstance(layer, nn.Sequential):
                return layer[-1]
            return layer
        else:
            raise ValueError(f"Layer '{layer_name}' not found in ResNet-18 model.")

    def generate_explanation(
        self, 
        image_input: Union[str, Path, Image.Image], 
        output_path: Optional[Union[str, Path]] = None,
        target_class_idx: Optional[int] = None,
        crop: str = "Potato"
    ) -> Dict[str, Any]:
        """
        Generate Grad-CAM heatmap visualization and diagnostic metadata.
        
        Args:
            image_input: Path to leaf photo or PIL Image object.
            output_path: File path where blended heatmap visualization image will be saved.
            target_class_idx: Optional class index to explain (defaults to top predicted class).
            crop: Target crop name (e.g. "Potato" or "Tomato").
            
        Returns:
            Dictionary containing:
            - 'predicted_disease': Primary predicted class string
            - 'friendly_name': Human-readable disease name
            - 'confidence': Confidence percentage float
            - 'output_path': Absolute file path to saved heatmap visualization
            - 'heatmap_np': 2D NumPy array of normalized heatmap intensities
        """
        # Load and preprocess image
        if isinstance(image_input, (str, Path)):
            image_path = Path(image_input)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            orig_pil = Image.open(image_path).convert("RGB")
        else:
            orig_pil = image_input.convert("RGB")

        # Resolve active model for target crop
        active_model = self.predictor.get_model(crop)
        if active_model is None:
            active_model = self.model
            
        # Hook activations and gradients on active model
        activations = None
        gradients = None
        
        target_layer = self._get_target_layer_for_model(active_model, "layer4")
        
        def forward_hook(module, input, output):
            nonlocal activations
            activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0].detach()

        h_f = target_layer.register_forward_hook(forward_hook)
        h_b = target_layer.register_full_backward_hook(backward_hook)

        # Align preprocessing transform with active predictor (letterbox or baseline)
        transform = getattr(self.predictor, "transform", None)
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        
        img_tensor = transform(orig_pil).unsqueeze(0).to(self.device) # Shape: [1, 3, 224, 224]
        img_tensor.requires_grad_()

        # 1. Forward Pass
        active_model.zero_grad()
        outputs = active_model(img_tensor)
        softmax_probs = torch.softmax(outputs, dim=1)[0]
        
        if target_class_idx is None:
            target_class_idx = torch.argmax(softmax_probs).item()

        score = outputs[0, target_class_idx]
        
        # 2. Backward Pass to calculate gradients w.r.t. target class score
        score.backward()
        
        # Remove hooks after backward pass
        h_f.remove()
        h_b.remove()

        # 3. Compute Grad-CAM weights and heatmap
        grad_sample = gradients[0] # Shape: [512, 7, 7]
        act_sample = activations[0] # Shape: [512, 7, 7]

        # Global average pooling of gradients
        weights = torch.mean(grad_sample, dim=(1, 2)) # Shape: [512]

        # Weighted combination of feature maps
        cam = torch.zeros(act_sample.shape[1:], dtype=torch.float32, device=self.device)
        for i, w in enumerate(weights):
            cam += w * act_sample[i]

        # Apply ReLU to zero out negative contributions
        cam = torch.relu(cam)

        # Normalize heatmap to [0, 1]
        cam_np = cam.cpu().numpy()
        if cam_np.max() > 0:
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
        else:
            cam_np = np.zeros_like(cam_np)

        # Resize heatmap to match original image size (224x224)
        heatmap_pil = Image.fromarray((cam_np * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        heatmap_resized = np.array(heatmap_pil) / 255.0

        # Create colorized heatmap overlay (JET colormap)
        jet_colormap = cm.get_cmap("jet")
        heatmap_color = jet_colormap(heatmap_resized)[:, :, :3] # Shape: [224, 224, 3] in [0, 1]

        # Prepare original resized image as numpy array [0, 1]
        orig_resized = orig_pil.resize((224, 224))
        orig_np = np.array(orig_resized) / 255.0

        # Blend original image (60%) and colorized heatmap (40%)
        blended_np = 0.6 * orig_np + 0.4 * heatmap_color
        blended_np = np.clip(blended_np * 255.0, 0, 255).astype(np.uint8)
        blended_pil = Image.fromarray(blended_np)

        # Save output image
        if output_path is None:
            save_dir = settings.DATA_DIR / "processed"
            save_dir.mkdir(parents=True, exist_ok=True)
            output_path = save_dir / "gradcam_output.jpg"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        blended_pil.save(output_path, "JPEG")

        # Get diagnostic prediction result
        pred_result = self.predictor.predict(orig_pil, crop=crop)

        return {
            "predicted_disease": pred_result["predicted_disease"],
            "friendly_name": pred_result["friendly_name"],
            "confidence": pred_result["confidence"],
            "probabilities": pred_result["probabilities"],
            "output_path": str(output_path.resolve()),
            "heatmap_np": cam_np
        }
