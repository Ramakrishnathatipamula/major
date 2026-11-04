"""
Crop Registry module for AI Crop Doctor.
Manages multi-crop specifications, supported disease models, and support status.
"""

from .crop_registry import (
    CropSpec,
    CropRegistry,
    get_crop_registry,
)

__all__ = [
    "CropSpec",
    "CropRegistry",
    "get_crop_registry",
]
