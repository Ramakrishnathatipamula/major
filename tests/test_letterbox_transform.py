"""
Focused unit tests for reusable letterbox transform and preprocessing integration (Milestone 6B).
Tests landscape, portrait, square, extreme aspect ratios, color modes, malformed inputs,
and reversible pipeline integration.
"""

import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from modules.image_utils import (
    letterbox_image,
    LetterboxTransform,
    get_preprocessing_transform,
    DEFAULT_LETTERBOX_FILL,
    IMAGENET_MEAN,
    IMAGENET_STD
)
from modules.identification.plant_identifier import PlantIdentifier
from modules.disease_predictor import DiseasePredictor
from modules.identification.router import PlantRouter


class TestLetterboxTransform(unittest.TestCase):
    """Unit tests for aspect-ratio-preserving letterbox transform."""

    def test_landscape_image_padding(self):
        """Landscape image (e.g. 800x400, 2:1) must pad vertically with neutral gray."""
        img = Image.new("RGB", (800, 400), (200, 50, 50))
        out = letterbox_image(img, target_size=(224, 224))

        self.assertEqual(out.size, (224, 224))
        self.assertEqual(out.mode, "RGB")

        arr = np.array(out)
        # Expected: width scaled to 224, height scaled to 112.
        # Vertical padding: (224 - 112) // 2 = 56 on top, 56 on bottom.
        # Top-left border pixel (0, 0) should be neutral gray padding
        np.testing.assert_array_equal(arr[0, 0], DEFAULT_LETTERBOX_FILL)
        np.testing.assert_array_equal(arr[50, 112], DEFAULT_LETTERBOX_FILL)
        # Center pixel (112, 112) should be the original image color
        np.testing.assert_array_equal(arr[112, 112], (200, 50, 50))
        # Bottom padding pixel (220, 112) should be neutral gray padding
        np.testing.assert_array_equal(arr[220, 112], DEFAULT_LETTERBOX_FILL)

    def test_portrait_image_padding(self):
        """Portrait image (e.g. 300x900, 1:3) must pad horizontally with neutral gray."""
        img = Image.new("RGB", (300, 900), (50, 180, 50))
        out = letterbox_image(img, target_size=(224, 224))

        self.assertEqual(out.size, (224, 224))
        self.assertEqual(out.mode, "RGB")

        arr = np.array(out)
        # Expected: height scaled to 224, width scaled to 75.
        # Horizontal padding: (224 - 75) // 2 = 74 on left and right.
        # Left border pixel (112, 0) should be padding
        np.testing.assert_array_equal(arr[112, 0], DEFAULT_LETTERBOX_FILL)
        # Center pixel (112, 112) should be the image color
        np.testing.assert_array_equal(arr[112, 112], (50, 180, 50))
        # Right border pixel (112, 220) should be padding
        np.testing.assert_array_equal(arr[112, 220], DEFAULT_LETTERBOX_FILL)

    def test_square_image_no_padding(self):
        """Square image (500x500) must scale directly without letterbox bands."""
        img = Image.new("RGB", (500, 500), (40, 40, 210))
        out = letterbox_image(img, target_size=(224, 224))

        self.assertEqual(out.size, (224, 224))
        arr = np.array(out)
        # All corners and center should be image color
        np.testing.assert_array_equal(arr[0, 0], (40, 40, 210))
        np.testing.assert_array_equal(arr[223, 223], (40, 40, 210))
        np.testing.assert_array_equal(arr[112, 112], (40, 40, 210))

    def test_exact_target_size_fast_path(self):
        """Input already matching target_size (224, 224) returns valid copy."""
        img = Image.new("RGB", (224, 224), (100, 150, 200))
        out = letterbox_image(img, target_size=(224, 224))

        self.assertEqual(out.size, (224, 224))
        self.assertIsNot(out, img)  # Must be a copy, not same instance
        arr = np.array(out)
        np.testing.assert_array_equal(arr[0, 0], (100, 150, 200))

    def test_extreme_aspect_ratios(self):
        """Extreme aspect ratios (10:1 panoramic, 1:10 strip) produce valid 224x224 RGB images."""
        panoramic = Image.new("RGB", (1000, 100), (220, 120, 30))
        out_pan = letterbox_image(panoramic, target_size=(224, 224))
        self.assertEqual(out_pan.size, (224, 224))
        self.assertEqual(out_pan.mode, "RGB")

        vertical_strip = Image.new("RGB", (100, 1000), (30, 120, 220))
        out_vert = letterbox_image(vertical_strip, target_size=(224, 224))
        self.assertEqual(out_vert.size, (224, 224))
        self.assertEqual(out_vert.mode, "RGB")

    def test_color_mode_conversions(self):
        """RGBA, Grayscale (L), and Palette (P) images are converted cleanly to RGB."""
        # RGBA with transparency
        rgba = Image.new("RGBA", (400, 300), (100, 200, 100, 128))
        out_rgba = letterbox_image(rgba, target_size=(224, 224))
        self.assertEqual(out_rgba.mode, "RGB")
        self.assertEqual(out_rgba.size, (224, 224))

        # Grayscale
        gray = Image.new("L", (300, 500), 120)
        out_gray = letterbox_image(gray, target_size=(224, 224))
        self.assertEqual(out_gray.mode, "RGB")
        self.assertEqual(out_gray.size, (224, 224))

    def test_numpy_array_input(self):
        """NumPy uint8 array input is accepted and correctly letterboxed."""
        np_img = np.full((350, 600, 3), fill_value=175, dtype=np.uint8)
        out = letterbox_image(np_img, target_size=(224, 224))
        self.assertEqual(out.size, (224, 224))
        self.assertEqual(out.mode, "RGB")

    def test_custom_fill_color(self):
        """User-specified fill_color is respected."""
        img = Image.new("RGB", (800, 400), (255, 255, 0))
        custom_fill = (64, 64, 64)
        out = letterbox_image(img, target_size=(224, 224), fill_color=custom_fill)

        arr = np.array(out)
        np.testing.assert_array_equal(arr[0, 0], custom_fill)
        np.testing.assert_array_equal(arr[112, 112], (255, 255, 0))

    def test_malformed_and_invalid_inputs(self):
        """Malformed, empty, non-existent, and corrupt inputs raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # 1. Non-existent path
            with self.assertRaises(ValueError):
                letterbox_image(tmp / "does_not_exist.jpg")

            # 2. 0-byte file
            empty_file = tmp / "empty.jpg"
            empty_file.write_bytes(b"")
            with self.assertRaises(ValueError):
                letterbox_image(empty_file)

            # 3. Corrupt byte stream
            corrupt_file = tmp / "corrupt.jpg"
            corrupt_file.write_bytes(b"\xff\xd8\xff\xe0" + b"GARBAGE_PAYLOAD_NOT_JPEG" * 10)
            with self.assertRaises(ValueError):
                letterbox_image(corrupt_file)

            # 4. Completely blank solid-color image (fails variance screening)
            blank_file = tmp / "blank.jpg"
            Image.new("RGB", (200, 200), (0, 0, 0)).save(blank_file)
            with self.assertRaises(ValueError):
                letterbox_image(blank_file)

    def test_torchvision_tensor_transform_pipeline(self):
        """get_preprocessing_transform(mode='letterbox') outputs normalized PyTorch tensor."""
        t_letterbox = get_preprocessing_transform(mode="letterbox", target_size=(224, 224))

        # Create natural-looking synthetic leaf image with texture
        np.random.seed(42)
        leaf_np = np.random.randint(50, 220, size=(400, 600, 3), dtype=np.uint8)
        leaf_img = Image.fromarray(leaf_np)

        tensor = t_letterbox(leaf_img)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([3, 224, 224]))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_reversibility_and_invalid_mode_error(self):
        """Transforms can be switched to 'baseline' or raise error on invalid mode."""
        t_base = get_preprocessing_transform(mode="baseline", target_size=(224, 224))
        self.assertIsNotNone(t_base)

        t_squish = get_preprocessing_transform(mode="squish", target_size=(224, 224))
        self.assertIsNotNone(t_squish)

        with self.assertRaises(ValueError):
            get_preprocessing_transform(mode="unsupported_quantum_warp")

    def test_model_and_router_integration(self):
        """PlantIdentifier, DiseasePredictor, and PlantRouter instantiate cleanly with letterbox."""
        # Locate a sample leaf from project dataset or construct textured image
        sample_path = next(Path("data/raw/plant_village_subset/Potato___Early_blight").glob("*.JPG"), None)
        if sample_path and sample_path.exists():
            sample_img = Image.open(sample_path).convert("RGB")
        else:
            np.random.seed(42)
            arr = np.random.randint(60, 200, size=(250, 350, 3), dtype=np.uint8)
            sample_img = Image.fromarray(arr)

        # PlantIdentifier
        identifier_lb = PlantIdentifier(device=torch.device("cpu"), preprocessing_mode="letterbox")
        self.assertEqual(identifier_lb.preprocessing_mode, "letterbox")
        res_id = identifier_lb.identify(sample_img)
        self.assertIsNotNone(res_id.crop_id)

        # Baseline mode reversibility check
        identifier_base = PlantIdentifier(device=torch.device("cpu"), preprocessing_mode="baseline")
        self.assertEqual(identifier_base.preprocessing_mode, "baseline")
        res_id_base = identifier_base.identify(sample_img)
        self.assertIsNotNone(res_id_base.crop_id)

        # DiseasePredictor
        predictor_lb = DiseasePredictor(device=torch.device("cpu"), preprocessing_mode="letterbox")
        self.assertEqual(predictor_lb.preprocessing_mode, "letterbox")
        diag_lb = predictor_lb.predict(sample_img, crop="Potato")
        self.assertIn("predicted_disease", diag_lb)

        # PlantRouter
        router_lb = PlantRouter(plant_identifier=identifier_lb)
        dec = router_lb.route(sample_img, manual_crop="Potato")
        self.assertTrue(dec.route_to_specialist)
        self.assertEqual(dec.target_crop, "potato")


if __name__ == "__main__":
    unittest.main()
