import os
from pathlib import Path
from typing import Tuple, Optional, Any, Union, List
from PIL import Image, UnidentifiedImageError
import numpy as np
from torchvision import transforms

# Supported image formats for AI Crop Doctor
SUPPORTED_IMAGE_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "BMP"}

# Input dimension limits
MIN_IMAGE_DIMENSION = 32          # Minimum 32x32 pixels to contain discernible leaf texture
MAX_IMAGE_DIMENSION = 10000       # Maximum 10,000x10,000 to prevent decompression bomb denial-of-service
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB file size limit

# Standard ImageNet normalization parameters
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# Letterbox padding policy:
# Neutral medium gray (128, 128, 128) is selected as the default padding color.
# In ImageNet-normalized tensor space where mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225],
# pixel value 128/255 ≈ 0.502 maps to normalized tensor values [0.074, 0.205, 0.427].
# This is near-zero activation, minimizing artificial step-gradient edge responses along canvas borders.
# In contrast, black padding (0, 0, 0) maps to [-2.118, -2.036, -1.804], inducing severe artificial
# high-contrast edge filters at border junctions that corrupt convolutional feature maps.
DEFAULT_LETTERBOX_FILL: Tuple[int, int, int] = (128, 128, 128)


def validate_and_load_image(image_input: Any) -> Tuple[bool, str, Optional[Image.Image]]:
    """
    Robust, safe image validator and loader for AI Crop Doctor.

    Validates:
    1. Input is non-None and is a valid file path, Streamlit UploadedFile, or PIL Image.
    2. File exists and has non-zero size (under 10MB limit).
    3. File is a genuine readable image (protects against corrupt files, truncated streams, bad headers).
    4. Format is supported (JPEG, PNG, WEBP, BMP).
    5. Dimensions are within acceptable limits (min 32x32, max 10,000x10,000).
    6. Decodes pixels safely to RGB without error.
    7. Basic pixel variance check: detects completely blank or solid-color images.

    Returns:
        Tuple of (is_valid: bool, error_message: str, pil_image: Optional[Image.Image])
    """
    if image_input is None:
        return False, "No image uploaded or selected.", None

    try:
        # Case 1: Input is already a PIL Image
        if isinstance(image_input, Image.Image):
            pil_img = image_input

        # Case 2: Input is a filesystem path (str or Path)
        elif isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                return False, f"Image file does not exist: {path.name}", None
            if not path.is_file():
                return False, f"Path is not a regular file: {path.name}", None
            file_size = path.stat().st_size
            if file_size == 0:
                return False, f"Image file is empty (0 bytes): {path.name}", None
            if file_size > MAX_FILE_SIZE_BYTES:
                return False, f"Image file exceeds 10MB size limit ({file_size / (1024 * 1024):.1f}MB).", None

            with Image.open(path) as raw_img:
                raw_format = getattr(raw_img, "format", None)
                pil_img = raw_img.copy()
                if raw_format:
                    pil_img.format = raw_format

        # Case 3: Streamlit UploadedFile or file-like object with read/size
        else:
            if hasattr(image_input, "size"):
                if image_input.size == 0:
                    return False, "Uploaded image file is empty (0 bytes).", None
                if image_input.size > MAX_FILE_SIZE_BYTES:
                    return False, f"Image file exceeds 10MB limit ({image_input.size / (1024 * 1024):.1f}MB).", None

            # Reset pointer if seekable
            if hasattr(image_input, "seek"):
                image_input.seek(0)

            pil_img = Image.open(image_input)

        # Verify format (if available on Image instance)
        img_format = getattr(pil_img, "format", None)
        if img_format and img_format.upper() not in SUPPORTED_IMAGE_FORMATS:
            return False, f"Unsupported image format '{img_format}'. Supported formats: JPG, PNG, WEBP, BMP.", None

        # Check image dimensions
        width, height = pil_img.size
        if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
            return False, f"Image dimensions ({width}x{height}) are too small. Expected at least {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION} pixels.", None
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            return False, f"Image dimensions ({width}x{height}) exceed maximum allowed limit ({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}).", None

        # Force full decoding of pixel stream to catch truncated/corrupted image data
        # Note: Image.open() is lazy; convert('RGB') and load() force pixel verification
        rgb_img = pil_img.convert("RGB")
        rgb_img.load()

        # Basic pixel variance check (detect solid black, solid white, or single-color dummy files)
        img_array = np.array(rgb_img)
        if img_array.std() < 1.0:
            return False, "Image appears completely blank or uniform without discernible visual features.", None

        return True, "", rgb_img

    except UnidentifiedImageError:
        return False, "File is not a recognized image or has corrupted headers.", None
    except (IOError, OSError) as exc:
        return False, f"Corrupted or truncated image data ({exc}).", None
    except Exception as exc:
        return False, f"Failed to load image ({exc}).", None


def letterbox_image(
    image: Union[Image.Image, np.ndarray, str, Path],
    target_size: Tuple[int, int] = (224, 224),
    fill_color: Tuple[int, int, int] = DEFAULT_LETTERBOX_FILL
) -> Image.Image:
    """
    Resize an input image while strictly preserving its original aspect ratio,
    padding symmetrical borders with a neutral fill color (letterboxing/pillarboxing).

    Padding Policy:
        - Default fill: (128, 128, 128) neutral medium gray.
        - Neutral gray maps to ~0.0 in ImageNet normalized space, minimizing artificial
          edge gradients at canvas boundaries.
        - Centered placement: Equal padding is applied to both sides/top-bottom.
        - High-quality bilinear resampling is used for scaling.

    Args:
        image: PIL Image, NumPy array, or filesystem path.
        target_size: Desired (width, height) output dimensions, default (224, 224).
        fill_color: RGB tuple for padding color, default DEFAULT_LETTERBOX_FILL.

    Returns:
        RGB PIL Image of dimensions exactly matching target_size.

    Raises:
        ValueError: If input image is invalid or cannot be decoded.
    """
    # Step 1: Resolve and safely load to RGB PIL Image
    if isinstance(image, Image.Image):
        rgb_img = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        rgb_img = Image.fromarray(image).convert("RGB")
    elif isinstance(image, (str, Path)):
        is_valid, err_msg, loaded = validate_and_load_image(image)
        if not is_valid or loaded is None:
            raise ValueError(f"Invalid image for letterboxing: {err_msg}")
        rgb_img = loaded
    else:
        # File-like object or other input
        is_valid, err_msg, loaded = validate_and_load_image(image)
        if not is_valid or loaded is None:
            raise ValueError(f"Invalid image for letterboxing: {err_msg}")
        rgb_img = loaded

    tw, th = target_size
    ow, oh = rgb_img.size

    # Fast path: already identical dimensions
    if ow == tw and oh == th:
        return rgb_img.copy()

    # Step 2: Compute aspect-preserving scale factor
    scale = min(tw / ow, th / oh)
    nw = max(1, min(tw, int(round(ow * scale))))
    nh = max(1, min(th, int(round(oh * scale))))

    # Step 3: Bilinear resize
    resample_method = getattr(Image, "Resampling", Image).BILINEAR
    resized_img = rgb_img.resize((nw, nh), resample=resample_method)

    # Step 4: Canvas instantiation and centered paste
    canvas = Image.new("RGB", (tw, th), fill_color)
    paste_x = (tw - nw) // 2
    paste_y = (th - nh) // 2
    canvas.paste(resized_img, (paste_x, paste_y))

    return canvas


class LetterboxTransform:
    """
    Torchvision-compatible transform for aspect-ratio preserving letterboxing.
    Accepts PIL Images or file paths and produces padded (target_size) RGB PIL Images.
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        fill_color: Tuple[int, int, int] = DEFAULT_LETTERBOX_FILL
    ):
        self.target_size = target_size
        self.fill_color = fill_color

    def __call__(self, img: Union[Image.Image, np.ndarray, str, Path]) -> Image.Image:
        return letterbox_image(img, target_size=self.target_size, fill_color=self.fill_color)

    def __repr__(self) -> str:
        return f"LetterboxTransform(target_size={self.target_size}, fill_color={self.fill_color})"


def get_preprocessing_transform(
    mode: str = "letterbox",
    target_size: Tuple[int, int] = (224, 224),
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    fill_color: Tuple[int, int, int] = DEFAULT_LETTERBOX_FILL
) -> transforms.Compose:
    """
    Factory function returning torchvision Compose transform for inference/validation.

    Supported Modes:
        - 'letterbox': Preserves aspect ratio, pads neutral borders to target_size, then normalizes.
        - 'baseline' / 'squish': Anamorphic direct resize to target_size, then normalizes.

    Args:
        mode: Preprocessing mode string ('letterbox', 'baseline', or 'squish').
        target_size: Output (width, height) tuple, default (224, 224).
        mean: Normalization mean (default: ImageNet [0.485, 0.456, 0.406]).
        std: Normalization std (default: ImageNet [0.229, 0.224, 0.225]).
        fill_color: Padding RGB tuple for letterbox mode.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    norm_mean = mean if mean is not None else IMAGENET_MEAN
    norm_std = std if std is not None else IMAGENET_STD

    clean_mode = str(mode).strip().lower()

    if clean_mode == "letterbox":
        return transforms.Compose([
            LetterboxTransform(target_size=target_size, fill_color=fill_color),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])
    elif clean_mode in ("baseline", "squish", "direct"):
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])
    else:
        raise ValueError(
            f"Unsupported preprocessing mode '{mode}'. Choose 'letterbox' or 'baseline'."
        )

