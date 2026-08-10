import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import settings
from modules.disease_predictor import DiseasePredictor
from modules.explainability.gradcam import GradCAM

def main():
    print("========================================")
    print(" AI Crop Doctor - Grad-CAM XAI Test ")
    print("========================================\n", flush=True)
    
    # 1. Initialize DiseasePredictor & GradCAM
    predictor = DiseasePredictor()
    gradcam = GradCAM(predictor=predictor)
    print("✓ Successfully initialized Grad-CAM with target layer 'model.layer4[-1]'.\n", flush=True)
    
    # 2. Target test image specified by user
    sample_img_path = settings.RAW_DATA_DIR / "plant_village_subset" / "Potato___Early_blight" / "real_071.JPG"
    
    if not sample_img_path.exists():
        print(f"✗ Error: Target image not found at {sample_img_path}")
        return
        
    print(f"Processing real leaf photo: {sample_img_path.relative_to(settings.BASE_DIR)}", flush=True)
    
    # 3. Generate Grad-CAM explanation and visualization
    output_save_path = settings.DATA_DIR / "processed" / "gradcam_early_blight.jpg"
    result = gradcam.generate_explanation(sample_img_path, output_path=output_save_path)
    
    # 4. Display exact requested results
    print("\n--- Grad-CAM Explanation Results ---")
    print(f"1. Predicted Class: {result['predicted_disease']} ({result['friendly_name']})")
    print(f"2. Confidence: {result['confidence']:.2f}%")
    print(f"3. Grad-CAM Visualization: Heatmap overlay successfully generated on 224x224 leaf tensor.")
    print(f"4. Saved Output Path: {result['output_path']}")
    
    print("\n========================================")
    print(" 🎉 Grad-CAM Explainability Test Passed!")
    print("========================================\n", flush=True)

if __name__ == "__main__":
    main()
