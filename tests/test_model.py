import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from config import settings
from modules.model import build_resnet_model, load_trained_model, predict_image, get_device

CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

def test_model_module():
    print("========================================")
    print(" AI Crop Doctor - Model Module Test ")
    print("========================================\n", flush=True)
    
    device = get_device()
    print(f"[1/4] Hardware Device: '{device}'", flush=True)
    
    # 1. Test ResNet-18 model architecture construction
    print(f"[2/4] Constructing ResNet-18 architecture...", flush=True)
    model = build_resnet_model(num_classes=3, pretrained=False)
    print(f"  ✓ Model instantiated. Output layer linear shape: {model.fc}", flush=True)
    
    # 2. Test forward pass with dummy tensor
    print(f"[3/4] Testing dummy batch forward pass tensor shapes...", flush=True)
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(dummy_input)
        
    assert output.shape == torch.Size([2, 3]), f"Unexpected output shape: {output.shape}"
    print(f"  ✓ Dummy forward pass output shape: {output.shape} (Batch Size x Num Classes)", flush=True)
    
    # 3. Test loading saved checkpoint if present
    print(f"[4/4] Checking trained model checkpoint availability...", flush=True)
    checkpoint_path = settings.BASE_DIR / "models" / "disease" / "potato_disease_model.pth"
    
    if checkpoint_path.exists():
        loaded_model = load_trained_model(checkpoint_path, num_classes=3, device=device)
        print(f"  ✓ Successfully loaded trained model from: {checkpoint_path.relative_to(settings.BASE_DIR)}", flush=True)
        
        # Test single image prediction on first available real leaf image
        sample_img_dir = settings.RAW_DATA_DIR / "plant_village_subset" / "Potato___healthy"
        sample_imgs = list(sample_img_dir.glob("*.[jJ][pP]*[gG]"))
        
        if sample_imgs:
            test_img_path = sample_imgs[0]
            pred_label, confidence, probs = predict_image(loaded_model, test_img_path, CLASSES, device=device)
            print(f"  ✓ Sample Prediction on '{test_img_path.name}':", flush=True)
            print(f"    • Predicted Class: {pred_label}", flush=True)
            print(f"    • Confidence: {confidence:.2f}%", flush=True)
            print(f"    • Probabilities: {probs}", flush=True)
    else:
        print(f"  ℹ Model checkpoint not found yet at '{checkpoint_path.relative_to(settings.BASE_DIR)}'. Run training to generate checkpoint.", flush=True)
        
    print("\n========================================")
    print(" 🎉 Model Architecture Test Passed!")
    print("========================================", flush=True)

if __name__ == "__main__":
    test_model_module()
