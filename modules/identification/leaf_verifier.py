import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.image_utils import validate_and_load_image


@dataclass
class LeafVerificationResult:
    """
    Result of leaf verification screening.
    Ensures non-plant, non-leaf, and severely corrupted inputs are rejected before routing.
    """
    is_leaf: bool
    confidence: float  # Heuristic leaf-likeness confidence score (0.0 to 100.0)
    status: str        # VERIFIED_LEAF, REJECTED_NON_LEAF, REJECTED_BLANK, REJECTED_CORRUPT
    reason: str
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_leaf": self.is_leaf,
            "confidence": round(self.confidence, 2),
            "status": self.status,
            "reason": self.reason,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()}
        }


class LeafVerifier:
    """
    Lightweight rule- and heuristic-based leaf verification engine.
    Analyzes visual, chromatic, and spatial features to confirm if an image exhibits foliar characteristics.
    
    PlantVillage contains ONLY leaf photos against uniform backdrops. A closed-set neural network
    alone will hallucinate high confidence for arbitrary non-leaf objects (tractors, pets, documents).
    LeafVerifier provides essential front-line protection against non-leaf inputs.
    """

    def __init__(
        self,
        min_foliar_pixel_ratio: float = 0.06,
        min_color_variance: float = 5.0,
        min_spatial_gradient: float = 2.0
    ):
        self.min_foliar_pixel_ratio = min_foliar_pixel_ratio
        self.min_color_variance = min_color_variance
        self.min_spatial_gradient = min_spatial_gradient

    def verify_image(self, image_input: Union[str, Path, Image.Image]) -> LeafVerificationResult:
        """
        Screen an input image to verify whether it exhibits valid leaf characteristics.
        
        Args:
            image_input: Image file path or PIL Image object.
            
        Returns:
            LeafVerificationResult indicating whether the image passes leaf screening.
        """
        # Step 1: Base validation (format, size, corrupt data, blank pixels)
        is_valid, err_msg, pil_img = validate_and_load_image(image_input)
        if not is_valid or pil_img is None:
            err_lower = str(err_msg).lower()
            status = "REJECTED_BLANK" if ("blank" in err_lower or "uniform" in err_lower) else "REJECTED_CORRUPT"
            return LeafVerificationResult(
                is_leaf=False,
                confidence=0.0,
                status=status,
                reason=f"Image validation failed: {err_msg}",
                metrics={"pixel_std": 0.0}
            )

        # Step 2: Extract visual representations
        rgb_img = pil_img.convert("RGB")
        img_arr = np.array(rgb_img, dtype=np.float32)

        # Overall color variance
        pixel_std = float(np.std(img_arr))
        if pixel_std < self.min_color_variance:
            return LeafVerificationResult(
                is_leaf=False,
                confidence=0.0,
                status="REJECTED_BLANK",
                reason="Image is uniform, blank, or lacks visual texture.",
                metrics={"pixel_std": pixel_std}
            )

        # Step 3: Color space analysis (HSV foliar hue spectrum)
        hsv_img = pil_img.convert("HSV")
        hsv_arr = np.array(hsv_img, dtype=np.float32)
        h = hsv_arr[:, :, 0]  # Hue in [0, 255] (corresponds to 0-360 degrees)
        s = hsv_arr[:, :, 1]  # Saturation in [0, 255]
        v = hsv_arr[:, :, 2]  # Value / Brightness in [0, 255]

        # In PIL HSV: H ranges 0-255.
        # Foliar spectrum includes:
        # - Chlorophyll green: H ~35-100
        # - Chlorotic yellow / lime: H ~25-35
        # - Necrotic brown blights / lesions: H ~10-25 with moderate saturation
        foliar_mask = (h >= 10) & (h <= 115) & (s >= 20) & (v >= 20)
        foliar_pixel_count = np.sum(foliar_mask)
        total_pixels = img_arr.shape[0] * img_arr.shape[1]
        foliar_ratio = float(foliar_pixel_count / total_pixels)

        # Spatial texture gradient (horizontal and vertical differences)
        diff_x = np.abs(img_arr[:, 1:, :] - img_arr[:, :-1, :])
        diff_y = np.abs(img_arr[1:, :, :] - img_arr[:-1, :, :])
        spatial_grad = float(np.mean(diff_x) + np.mean(diff_y))

        # Check for extreme non-foliar synthetic color artifacts (e.g. pure cyan or magenta blocks)
        non_foliar_extreme_mask = ((h > 120) & (h < 220) & (s > 100)) # Blue / purple / cyan
        non_foliar_ratio = float(np.sum(non_foliar_extreme_mask) / total_pixels)

        metrics = {
            "foliar_ratio": foliar_ratio,
            "pixel_std": pixel_std,
            "spatial_gradient": spatial_grad,
            "non_foliar_extreme_ratio": non_foliar_ratio
        }

        # Step 4: Decision logic
        if foliar_ratio < self.min_foliar_pixel_ratio:
            # Insufficient foliar pigments
            return LeafVerificationResult(
                is_leaf=False,
                confidence=round(foliar_ratio * 100.0, 1),
                status="REJECTED_NON_LEAF",
                reason=(
                    f"Non-leaf image detected: Foliar chromatic area ({foliar_ratio * 100:.1f}%) "
                    f"is below leaf threshold ({self.min_foliar_pixel_ratio * 100:.1f}%). "
                    "Please upload a clear photograph of plant foliage."
                ),
                metrics=metrics
            )

        if spatial_grad < self.min_spatial_gradient:
            return LeafVerificationResult(
                is_leaf=False,
                confidence=10.0,
                status="REJECTED_NON_LEAF",
                reason="Image lacks organic leaf venation and boundary contours.",
                metrics=metrics
            )

        # Compute heuristic leaf confidence [60.0 to 98.0%]
        # Normalizes foliar ratio between min_ratio and 0.40
        norm_foliar = min(1.0, max(0.0, (foliar_ratio - self.min_foliar_pixel_ratio) / (0.40 - self.min_foliar_pixel_ratio)))
        leaf_conf = 60.0 + (norm_foliar * 38.0)
        if non_foliar_ratio > 0.40:
            leaf_conf = max(40.0, leaf_conf - 30.0)

        return LeafVerificationResult(
            is_leaf=True,
            confidence=round(leaf_conf, 1),
            status="VERIFIED_LEAF",
            reason="Image exhibits confirmed foliar color distribution and natural texture.",
            metrics=metrics
        )
