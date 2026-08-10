import sys
from pathlib import Path
from PIL import Image, ImageDraw

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import settings

CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

def create_sample_leaf_image(class_name: str, index: int, target_dir: Path):
    """Generate a clean sample leaf image for testing."""
    # Create a 256x256 image with green background resembling a leaf
    img = Image.new("RGB", (256, 256), color=(34, 139, 34)) # Forest Green
    draw = ImageDraw.Draw(img)
    
    # Draw leaf shape
    draw.ellipse([30, 30, 226, 226], fill=(50, 160, 50), outline=(20, 100, 20))
    
    # Add disease spots depending on the class
    if class_name == "Potato___Early_blight":
        # Early blight: small brown spots with concentric ring pattern
        for x, y in [(80, 90), (140, 150), (170, 90), (100, 180)]:
            draw.ellipse([x-15, y-15, x+15, y+15], fill=(139, 69, 19)) # Brown
            draw.ellipse([x-8, y-8, x+8, y+8], fill=(160, 82, 45))
    elif class_name == "Potato___Late_blight":
        # Late blight: dark water-soaked lesions with dark centers
        for x, y in [(100, 120), (160, 130)]:
            draw.ellipse([x-30, y-25, x+30, y+25], fill=(40, 40, 40)) # Dark Grey/Black
            draw.ellipse([x-20, y-15, x+20, y+15], fill=(100, 100, 50)) # Yellowish lesion border
            
    # Save image
    file_path = target_dir / f"sample_{index+1:02d}.jpg"
    img.save(file_path, "JPEG")
    return file_path

def main():
    print("Creating sample PlantVillage Potato dataset subset...")
    dataset_base = settings.RAW_DATA_DIR / "plant_village_subset"
    dataset_base.mkdir(parents=True, exist_ok=True)
    
    total_created = 0
    for cls in CLASSES:
        cls_dir = dataset_base / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(5):
            create_sample_leaf_image(cls, i, cls_dir)
            total_created += 1
            
        print(f"  ✓ Created 5 sample images in: {cls_dir.relative_to(settings.BASE_DIR)}")
        
    print(f"\n🎉 Successfully created {total_created} sample images across {len(CLASSES)} classes!")
    print(f"Dataset root: {dataset_base}")

if __name__ == "__main__":
    main()
