import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from config import settings
from modules.dataset import (
    load_plant_dataset, 
    get_class_names, 
    get_class_to_idx, 
    create_data_loader,
    get_transforms
)

EXPECTED_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

def test_dataset_pipeline():
    print("========================================")
    print(" AI Crop Doctor - Dataset Module Test ")
    print("========================================\n", flush=True)
    
    dataset_dir = settings.RAW_DATA_DIR / "plant_village_subset"
    
    # 1. Verify dataset folder exists
    print(f"[1/5] Verifying dataset directory path...", flush=True)
    assert dataset_dir.exists(), f"Dataset directory missing: {dataset_dir}"
    print(f"  ✓ Found dataset folder at: {dataset_dir.relative_to(settings.BASE_DIR)}", flush=True)
    
    # 2. Verify subdirectories exist for each class
    for cls in EXPECTED_CLASSES:
        cls_path = dataset_dir / cls
        assert cls_path.exists(), f"Missing class directory: {cls_path}"
        num_files = len([f for f in cls_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        print(f"  ✓ Subdirectory '{cls}' contains {num_files} real leaf images.", flush=True)
        
    # 3. Load dataset using PyTorch ImageFolder
    print(f"\n[2/5] Loading PyTorch ImageFolder...", flush=True)
    transforms_dict = get_transforms(image_size=(224, 224))
    dataset = load_plant_dataset(dataset_dir, transform=transforms_dict["val"])
    print(f"  ✓ Successfully loaded {len(dataset)} total images.", flush=True)
    
    # 4. Verify class names & integer mapping
    print(f"\n[3/5] Verifying class names & label mapping...", flush=True)
    detected_classes = get_class_names(dataset)
    class_to_idx = get_class_to_idx(dataset)
    
    assert detected_classes == EXPECTED_CLASSES, f"Class mismatch. Detected: {detected_classes}"
    print(f"  ✓ Detected classes: {detected_classes}", flush=True)
    print(f"  ✓ Label integer mapping: {class_to_idx}", flush=True)
    
    # 5. Check sample item, image loading & tensor shape
    print(f"\n[4/5] Testing single image loading & tensor transformations...", flush=True)
    sample_img_tensor, sample_label_idx = dataset[0]
    expected_shape = torch.Size([3, 224, 224])
    
    assert sample_img_tensor.shape == expected_shape, f"Shape mismatch: {sample_img_tensor.shape}"
    assert isinstance(sample_label_idx, int), "Label index must be an integer"
    
    label_name = detected_classes[sample_label_idx]
    print(f"  ✓ Sample 0 image tensor shape: {sample_img_tensor.shape} (Channels x Height x Width)", flush=True)
    print(f"  ✓ Sample 0 label index: {sample_label_idx} ('{label_name}')", flush=True)
    
    # 6. Test PyTorch DataLoader batch extraction
    print(f"\n[5/5] Testing PyTorch DataLoader mini-batch extraction...", flush=True)
    batch_size = 4
    loader = create_data_loader(dataset, batch_size=batch_size, shuffle=False)
    
    batch_imgs, batch_labels = next(iter(loader))
    expected_batch_shape = torch.Size([batch_size, 3, 224, 224])
    
    assert batch_imgs.shape == expected_batch_shape, f"Batch shape mismatch: {batch_imgs.shape}"
    assert len(batch_labels) == batch_size, f"Batch labels size mismatch: {len(batch_labels)}"
    
    print(f"  ✓ DataLoader batch images shape: {batch_imgs.shape} (Batch Size x Channels x H x W)", flush=True)
    print(f"  ✓ DataLoader batch labels tensor: {batch_labels.tolist()}", flush=True)
    
    print("\n========================================")
    print(" 🎉 All Dataset Tests Passed Successfully!")
    print("========================================", flush=True)

if __name__ == "__main__":
    test_dataset_pipeline()
