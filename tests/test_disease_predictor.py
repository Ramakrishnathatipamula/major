import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import settings
from modules.disease_predictor import DiseasePredictor

def main():
    print("========================================")
    print(" AI Crop Doctor - Inference Module Test ")
    print("========================================\n", flush=True)
    
    # 1. Initialize predictor
    predictor = DiseasePredictor()
    print("✓ Successfully loaded DiseasePredictor model checkpoint.\n", flush=True)
    
    # 2. Select ONE sample image from real dataset
    sample_dir = settings.RAW_DATA_DIR / "plant_village_subset" / "Potato___Early_blight"
    sample_images = list(sample_dir.glob("*.[jJ][pP]*[gG]"))
    
    if not sample_images:
        print("✗ Error: No sample image found in data/raw/plant_village_subset/Potato___Early_blight")
        return
        
    sample_img_path = sample_images[0]
    print(f"Testing inference on real image: {sample_img_path.relative_to(settings.BASE_DIR)}\n", flush=True)
    
    # 3. Run prediction
    result = predictor.predict(sample_img_path)
    
    # 4. Print output in exact user requested format
    print(f"Predicted disease: {result['predicted_disease']}")
    print(f"Confidence: {result['confidence']:.2f}%\n")
    print("Probabilities:")
    probs = result["probabilities"]
    print(f"Early Blight: {probs.get('Early Blight', 0.0):.2f}%")
    print(f"Late Blight: {probs.get('Late Blight', 0.0):.2f}%")
    print(f"Healthy: {probs.get('Healthy', 0.0):.2f}%")
    
    print("\n========================================")
    print(" 🎉 Disease Predictor Test Passed!")
    print("========================================", flush=True)

if __name__ == "__main__":
    main()
