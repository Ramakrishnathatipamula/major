import sys
import json
import hashlib
import unittest
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import settings
from modules.dataset import load_plant_dataset, get_class_names
from modules.dataset_paths import resolve_dataset_path


class TestDatasetAndModelIntegrity(unittest.TestCase):
    """
    Integrity verification for Phase 1 dataset cleanup:
    - Verifies synthetic sample quarantine isolation
    - Verifies active training dataset loader excludes archived samples
    - Verifies model checkpoint remains untouched
    - Verifies machine-readable dataset inventory
    """

    EXPECTED_CHECKPOINT_HASH = "5257bda513c482eaae1dea6af4df29eb"
    EXPECTED_CHECKPOINT_SIZE = 44791371
    ACTIVE_CLASSES = [
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy"
    ]

    def test_archived_synthetic_samples_outside_active_training_directories(self):
        """Archived synthetic samples must reside outside active training directories."""
        archive_dir = settings.DATA_DIR / "archived_synthetic_samples"
        self.assertTrue(archive_dir.exists(), f"Archive directory missing: {archive_dir}")

        manifest_path = archive_dir / "manifest.json"
        self.assertTrue(manifest_path.exists(), f"Quarantine manifest missing: {manifest_path}")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        self.assertEqual(manifest.get("total_quarantined"), 15)
        quarantined_files = manifest.get("quarantined_files", [])
        self.assertEqual(len(quarantined_files), 15)

        # Confirm every archived sample exists in archive directory and has valid hash
        for item in quarantined_files:
            archived_path = resolve_dataset_path(item["archived_path"])
            self.assertTrue(archived_path.exists(), f"Archived file missing: {archived_path}")
            file_hash = hashlib.md5(archived_path.read_bytes()).hexdigest()
            self.assertEqual(file_hash, item["md5_hash"])

        # Confirm ZERO synthetic files remain in active training directory
        active_dataset_dir = settings.RAW_DATA_DIR / "plant_village_subset"
        for cls in self.ACTIVE_CLASSES:
            cls_dir = active_dataset_dir / cls
            self.assertTrue(cls_dir.exists(), f"Active class directory missing: {cls_dir}")
            synthetic_remnants = list(cls_dir.glob("sample_*.jpg"))
            self.assertEqual(
                synthetic_remnants,
                [],
                f"Active directory '{cls}' still contains synthetic samples: {synthetic_remnants}"
            )

            # Confirm all remaining images are real images
            real_images = list(cls_dir.glob("real_*.JPG"))
            self.assertEqual(len(real_images), 100, f"Expected 100 real images in {cls}, found {len(real_images)}")

    def test_training_dataset_loader_excludes_archived_samples(self):
        """Active dataset loader must load exactly 300 real images without any synthetic samples."""
        active_dataset_dir = settings.RAW_DATA_DIR / "plant_village_subset"
        dataset = load_plant_dataset(active_dataset_dir)

        # Total images loaded must be exactly 300
        self.assertEqual(len(dataset), 300)
        self.assertEqual(get_class_names(dataset), self.ACTIVE_CLASSES)

        # Verify none of the samples in the PyTorch dataset are synthetic or archived
        for img_path, label_idx in dataset.samples:
            path_obj = Path(img_path)
            self.assertTrue(path_obj.name.startswith("real_"), f"Unexpected file in dataset: {path_obj.name}")
            self.assertFalse(path_obj.name.startswith("sample_"), f"Synthetic file leaked into dataset: {path_obj.name}")
            self.assertNotIn("archived_synthetic_samples", str(path_obj))

    def test_existing_potato_checkpoint_untouched(self):
        """Primary and backup model checkpoints must match exact byte size and MD5 hash."""
        primary_ckpt = settings.BASE_DIR / "models" / "disease" / "potato_disease_model.pth"
        backup_ckpt = settings.SAVED_MODELS_DIR / "potato_disease_model.pth"

        self.assertTrue(primary_ckpt.exists(), f"Primary checkpoint missing: {primary_ckpt}")
        self.assertTrue(backup_ckpt.exists(), f"Backup checkpoint missing: {backup_ckpt}")

        self.assertEqual(primary_ckpt.stat().st_size, self.EXPECTED_CHECKPOINT_SIZE)
        self.assertEqual(backup_ckpt.stat().st_size, self.EXPECTED_CHECKPOINT_SIZE)

        primary_md5 = hashlib.md5(primary_ckpt.read_bytes()).hexdigest()
        backup_md5 = hashlib.md5(backup_ckpt.read_bytes()).hexdigest()

        self.assertEqual(primary_md5, self.EXPECTED_CHECKPOINT_HASH, "Primary checkpoint hash changed!")
        self.assertEqual(backup_md5, self.EXPECTED_CHECKPOINT_HASH, "Backup checkpoint hash changed!")

    def test_dataset_inventory_json_validity(self):
        """Verify machine-readable dataset inventory file structure and metrics."""
        inventory_path = settings.DATA_DIR / "dataset_inventory.json"
        self.assertTrue(inventory_path.exists(), f"Inventory missing: {inventory_path}")

        with open(inventory_path, "r") as f:
            inv = json.load(f)

        self.assertEqual(inv["scan_summary"]["total_active_project_images"], 300)
        self.assertEqual(inv["scan_summary"]["total_quarantined_synthetic_images"], 15)
        self.assertEqual(inv["scan_summary"]["total_locally_cached_plantvillage_images"], 54305)
        self.assertEqual(inv["scan_summary"]["total_unique_crops_in_cache"], 14)
        self.assertEqual(inv["scan_summary"]["total_unique_classes_in_cache"], 38)
        self.assertTrue(inv["scan_summary"]["is_rice_available_locally"])


if __name__ == "__main__":
    unittest.main()
