import json
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.dataset_paths import (
    PROJECT_ROOT,
    get_plantvillage_root,
    resolve_dataset_path,
    to_portable_path,
)


class TestDatasetPathPortability(unittest.TestCase):
    def test_project_paths_are_stored_relative(self):
        sample = ROOT_DIR / "data" / "raw" / "plant_village_subset" / "Potato___healthy" / "real_001.JPG"
        self.assertEqual(to_portable_path(sample), "data/raw/plant_village_subset/Potato___healthy/real_001.JPG")

    def test_home_paths_use_home_prefix(self):
        sample = Path.home() / ".cache" / "example" / "image.jpg"
        self.assertEqual(to_portable_path(sample), "~/.cache/example/image.jpg")

    def test_relative_and_tilde_paths_resolve_to_absolute(self):
        relative = resolve_dataset_path("data/raw/.gitkeep")
        tilde = resolve_dataset_path("~/.cache")
        self.assertTrue(relative.is_absolute())
        self.assertEqual(relative, (PROJECT_ROOT / "data" / "raw" / ".gitkeep").resolve())
        self.assertTrue(tilde.is_absolute())
        self.assertEqual(tilde, (Path.home() / ".cache").resolve())

    def test_manifest_paths_are_portable_and_resolvable(self):
        manifests = [
            "data/tomato/splits/test.json",
            "data/identification/splits/test.json",
            "data/field_eval/manifests/field_benchmark.json",
            "data/field_eval/manifests/field_adaptation_potato_train.json",
            "data/archived_synthetic_samples/manifest.json",
        ]

        for manifest_name in manifests:
            manifest_path = ROOT_DIR / manifest_name
            if not manifest_path.exists():
                self.skipTest(f"Manifest not found: {manifest_path}")

            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                records = data
                path_key = "path"
            elif "images" in data:
                records = data["images"]
                path_key = "path"
            else:
                records = data["quarantined_files"]
                path_key = "archived_path"

            self.assertTrue(records, f"Manifest is empty: {manifest_name}")
            raw_path = records[0][path_key]
            self.assertNotIn("/Users/", raw_path, f"Machine-specific path in {manifest_name}")
            self.assertTrue(resolve_dataset_path(raw_path).exists(), f"Unresolvable path in {manifest_name}")

    def test_plantvillage_root_default_is_not_hardcoded(self):
        import modules.dataset_paths as dataset_paths

        source = Path(dataset_paths.__file__).read_text(encoding="utf-8")
        self.assertNotIn("/Users/kittu", source)
        root = get_plantvillage_root()
        self.assertTrue(root.is_absolute())


if __name__ == "__main__":
    unittest.main()
