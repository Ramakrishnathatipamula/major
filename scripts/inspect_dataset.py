import sys
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import settings

CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

def inspect_folder(base_path: Path, title: str):
    print(f"========================================")
    print(f" {title} ")
    print(f" Base Path: {base_path}")
    print(f"========================================\n")
    
    if not base_path.exists():
        print(f"  ✗ Path does not exist: {base_path}\n")
        return
        
    total_images = 0
    for cls in CLASSES:
        # Search recursively or directly
        matching_dirs = list(base_path.rglob(cls))
        if not matching_dirs:
            print(f"  ✗ Class '{cls}': Directory NOT found.")
            continue
            
        cls_dir = matching_dirs[0]
        image_files = [f for f in cls_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        
        # Verify images with PIL
        valid_count = 0
        corrupt_count = 0
        sample_paths = []
        
        for idx, img_path in enumerate(image_files):
            try:
                with Image.open(img_path) as img:
                    img.verify() # Verify image integrity
                valid_count += 1
                if idx < 3:
                    sample_paths.append(img_path)
            except Exception:
                corrupt_count += 1
                
        total_images += valid_count
        print(f"  📁 Class: {cls}")
        print(f"     - Image Count: {valid_count} valid image files (Corrupt: {corrupt_count})")
        print(f"     - Sample Files:")
        for p in sample_paths:
            print(f"       • {p}")
        print()
        
    print(f"  Total Valid Images Inspected: {total_images}\n")

def main():
    # 1. Inspect Project Extracted Dataset
    inspect_folder(settings.RAW_DATA_DIR / "plant_village_subset", "Project Local Subset Inspection")
    
    # 2. Inspect KaggleHub Raw Cache Dataset
    kaggle_cache_path = Path.home() / ".cache/kagglehub/datasets/mohitsingh1804/plantvillage/versions/1"
    inspect_folder(kaggle_cache_path, "KaggleHub Raw Dataset Cache Inspection")

if __name__ == "__main__":
    main()
