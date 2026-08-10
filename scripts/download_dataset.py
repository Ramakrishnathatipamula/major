import sys
import shutil
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import kagglehub
from config import settings

TARGET_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

MAX_IMAGES_PER_CLASS = 100  # Clean, fast subset size for training

def download_and_extract_potato_subset():
    print("========================================")
    print(" Downloading PlantVillage via KaggleHub ")
    print("========================================\n", flush=True)
    
    # 1. Download PlantVillage dataset via KaggleHub
    print("Downloading dataset from KaggleHub ('mohitsingh1804/plantvillage')...", flush=True)
    download_path_str = kagglehub.dataset_download("mohitsingh1804/plantvillage")
    download_path = Path(download_path_str)
    print(f"  ✓ Raw dataset downloaded to: {download_path}\n", flush=True)
    
    # 2. Target destination folder
    target_base_dir = settings.RAW_DATA_DIR / "plant_village_subset"
    
    # Clean previous synthetic sample files if present
    if target_base_dir.exists():
        shutil.rmtree(target_base_dir)
    target_base_dir.mkdir(parents=True, exist_ok=True)
    
    print("Extracting 3 Potato disease classes into project directory...", flush=True)
    total_copied = 0
    
    # Search recursively for matching class directories in downloaded dataset
    for cls in TARGET_CLASSES:
        cls_target_dir = target_base_dir / cls
        cls_target_dir.mkdir(parents=True, exist_ok=True)
        
        # Find matching directory in downloaded path
        matching_dirs = list(download_path.rglob(cls))
        if not matching_dirs:
            print(f"  ⚠ Warning: Could not find class directory '{cls}' in downloaded path.", flush=True)
            continue
            
        source_cls_dir = matching_dirs[0]
        # Get all image files (.jpg, .jpeg, .png, .JPG)
        image_files = [f for f in source_cls_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        
        # Select up to MAX_IMAGES_PER_CLASS images
        selected_files = image_files[:MAX_IMAGES_PER_CLASS]
        
        for idx, img_file in enumerate(selected_files):
            dest_file = cls_target_dir / f"real_{idx+1:03d}{img_file.suffix}"
            shutil.copy2(img_file, dest_file)
            total_copied += 1
            
        print(f"  ✓ Extracted {len(selected_files)} real images into: data/raw/plant_village_subset/{cls}", flush=True)
        
    print("\n========================================")
    print(f" 🎉 Successfully installed {total_copied} REAL images across {len(TARGET_CLASSES)} classes!")
    print(f" Dataset Location: {target_base_dir}")
    print("========================================\n", flush=True)

if __name__ == "__main__":
    download_and_extract_potato_subset()
