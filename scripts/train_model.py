import sys
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import settings
from modules.model import build_resnet_model, get_device
from modules.dataset import IMAGENET_MEAN, IMAGENET_STD

def get_augmented_transforms():
    """Returns training transforms with data augmentation and standard val/test transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return train_transform, eval_transform

class TransformedDataset(torch.utils.data.Dataset):
    """Custom wrapper to apply dataset-specific transforms to subset samples."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def train_and_evaluate_model():
    print("========================================")
    print(" AI Crop Doctor - ResNet Model Training ")
    print("========================================\n", flush=True)
    
    device = get_device()
    print(f"[1/6] Hardware Compute Device: '{device}'", flush=True)
    
    # 1. Dataset Loading
    dataset_dir = settings.RAW_DATA_DIR / "plant_village_subset"
    print(f"[2/6] Loading raw dataset from: {dataset_dir.relative_to(settings.BASE_DIR)}", flush=True)
    
    # Load un-transformed ImageFolder to enable custom split transforms
    raw_dataset = datasets.ImageFolder(root=str(dataset_dir))
    class_names = raw_dataset.classes
    print(f"  - Detected Classes ({len(class_names)}): {class_names}", flush=True)
    print(f"  - Total Images: {len(raw_dataset)}", flush=True)
    
    # 2. Stratified / Seeded Train, Val, Test Split (70%, 15%, 15%)
    generator = torch.Generator().manual_seed(42)
    total_size = len(raw_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_subset, val_subset, test_subset = random_split(
        raw_dataset, [train_size, val_size, test_size], generator=generator
    )
    print(f"  - Split Sizes -> Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}", flush=True)
    
    # 3. Apply Transformations
    train_tf, eval_tf = get_augmented_transforms()
    train_dataset = TransformedDataset(train_subset, transform=train_tf)
    val_dataset = TransformedDataset(val_subset, transform=eval_tf)
    test_dataset = TransformedDataset(test_subset, transform=eval_tf)
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Model Initialization
    print(f"\n[3/6] Initializing Pretrained ResNet-18 Model...", flush=True)
    model = build_resnet_model(num_classes=len(class_names), pretrained=True)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    
    # 5. Training Loop
    epochs = 10
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_val_acc = 0.0
    
    print(f"\n[4/6] Starting Training Loop ({epochs} Epochs)...", flush=True)
    print("-" * 65, flush=True)
    print(f"{'Epoch':<7} | {'Train Loss':<11} | {'Train Acc (%)':<13} | {'Val Loss':<10} | {'Val Acc (%)':<11}", flush=True)
    print("-" * 65, flush=True)
    
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = (running_corrects / len(train_dataset)) * 100.0
        
        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data).item()
                
        epoch_val_loss = val_running_loss / len(val_dataset)
        epoch_val_acc = (val_running_corrects / len(val_dataset)) * 100.0
        
        print(f"{epoch:<7} | {epoch_train_loss:<11.4f} | {epoch_train_acc:<13.2f} | {epoch_val_loss:<10.4f} | {epoch_val_acc:<11.2f}", flush=True)
        
        # Deep copy best model checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
    print("-" * 65, flush=True)
    print(f"  ✓ Best Validation Loss: {best_val_loss:.4f} | Best Validation Accuracy: {best_val_acc:.2f}%\n", flush=True)
    
    # 6. Save Best Checkpoint
    model_dir = settings.BASE_DIR / "models" / "disease"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = model_dir / "potato_disease_model.pth"
    
    # Also save to saved_models directory for backup
    backup_dir = settings.SAVED_MODELS_DIR
    backup_save_path = backup_dir / "potato_disease_model.pth"
    
    torch.save(best_model_wts, model_save_path)
    torch.save(best_model_wts, backup_save_path)
    print(f"[5/6] Saved best trained checkpoint to:", flush=True)
    print(f"  - Primary: {model_save_path.relative_to(settings.BASE_DIR)}", flush=True)
    print(f"  - Backup:  {backup_save_path.relative_to(settings.BASE_DIR)}", flush=True)
    
    # 7. Evaluate on Unseen Test Dataset
    print(f"\n[6/6] Evaluating Best Checkpoint on Unseen Test Set ({len(test_dataset)} images)...", flush=True)
    model.load_state_dict(best_model_wts)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    test_accuracy = (np.sum(all_preds == all_targets) / len(all_targets)) * 100.0
    print(f"  ✓ Overall Test Set Accuracy: {test_accuracy:.2f}%\n", flush=True)
    
    print("Classification Report (Precision, Recall, F1-Score):")
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4), flush=True)
    
    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:")
    print(f"Classes order: {class_names}")
    print(cm, flush=True)
    
    print("\n========================================")
    print(" 🎉 Model Training & Evaluation Complete!")
    print("========================================", flush=True)

if __name__ == "__main__":
    train_and_evaluate_model()
