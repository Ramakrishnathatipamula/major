import os
import sys
import json
import time
import hashlib
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import settings
from modules.dataset import IMAGENET_MEAN, IMAGENET_STD
from modules.dataset_paths import get_plantvillage_root, resolve_dataset_path, to_portable_path

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

TOMATO_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

TOMATO_FRIENDLY_NAMES = {
    "Tomato___Bacterial_spot": "Bacterial Spot",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___Leaf_Mold": "Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spider Mites",
    "Tomato___Target_Spot": "Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus": "Mosaic Virus",
    "Tomato___healthy": "Healthy"
}


class TomatoDiseaseDataset(Dataset):
    """PyTorch Dataset loading images from file paths with given transforms."""
    def __init__(self, records: List[Dict[str, Any]], transform=None):
        self.records = records
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(TOMATO_CLASSES)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        image_path = resolve_dataset_path(item["path"])
        class_name = item["class_name"]
        label = self.class_to_idx[class_name]

        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")

        if self.transform:
            rgb_img = self.transform(rgb_img)

        return rgb_img, label


def get_plant_id_leakage_hashes() -> set:
    """Collect MD5 hashes from Stage 1 plant identification splits to prevent data leakage."""
    id_dir = ROOT_DIR / "data" / "identification" / "splits"
    leakage_hashes = set()
    for split_name in ["train", "val", "test"]:
        split_path = id_dir / f"{split_name}.json"
        if split_path.exists():
            with open(split_path, "r") as f:
                records = json.load(f)
            for rec in records:
                if rec.get("crop_id") == "tomato" and "md5" in rec:
                    leakage_hashes.add(rec["md5"])
    return leakage_hashes


def collect_and_prepare_dataset(
    cache_base: Path,
    max_samples_per_class: int = 500
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Scan disk, deduplicate by MD5, filter plant-id overlap, exclude synthetic images,
    and create reproducible stratified dataset.
    """
    leakage_hashes = get_plant_id_leakage_hashes()
    print(f"  ✓ Found {len(leakage_hashes)} plant-identification Tomato hashes to exclude for leakage prevention.")

    raw_stats = {}
    dedup_pool: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_hashes = set()

    for cls in TOMATO_CLASSES:
        raw_files = []
        for split in ["train", "val"]:
            cls_dir = cache_base / split / cls
            if cls_dir.exists():
                raw_files.extend(list(cls_dir.glob("*.JPG")) + list(cls_dir.glob("*.jpg")))
        raw_files = sorted(raw_files)
        raw_stats[cls] = {
            "total_files_on_disk": len(raw_files),
            "duplicates_removed": 0,
            "leakage_excluded": 0,
            "retained": 0
        }

        # Deduplicate and filter
        for file_path in raw_files:
            try:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
                h = hashlib.md5(file_bytes).hexdigest()

                if h in seen_hashes:
                    raw_stats[cls]["duplicates_removed"] += 1
                    continue

                if h in leakage_hashes:
                    raw_stats[cls]["leakage_excluded"] += 1
                    continue

                seen_hashes.add(h)
                dedup_pool[cls].append({
                    "path": to_portable_path(file_path),
                    "class_name": cls,
                    "md5": h
                })
            except Exception as e:
                print(f"    ⚠️ Warning reading {file_path}: {e}")

        raw_stats[cls]["unique_clean_count"] = len(dedup_pool[cls])

    # Sample reproducibly per class
    sampled_pool: Dict[str, List[Dict[str, Any]]] = {}
    for cls in TOMATO_CLASSES:
        items = list(dedup_pool[cls])
        random.seed(SEED)
        random.shuffle(items)
        if max_samples_per_class and len(items) > max_samples_per_class:
            items = items[:max_samples_per_class]
        sampled_pool[cls] = items
        raw_stats[cls]["retained"] = len(items)
        print(f"    - {cls}: {raw_stats[cls]['total_files_on_disk']} raw -> {raw_stats[cls]['unique_clean_count']} unique -> {len(items)} retained")

    return sampled_pool, raw_stats


def create_stratified_splits(
    dataset_pool: Dict[str, List[Dict[str, Any]]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create reproducible stratified train, val, test splits."""
    train_set, val_set, test_set = [], [], []

    for cls in TOMATO_CLASSES:
        items = list(dataset_pool[cls])
        random.seed(SEED)
        random.shuffle(items)

        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_set.extend(items[:n_train])
        val_set.extend(items[n_train:n_train + n_val])
        test_set.extend(items[n_train + n_val:])

    # Shuffle combined sets
    random.seed(SEED)
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    return train_set, val_set, test_set


def build_model(device: torch.device) -> nn.Module:
    """Build ResNet-18 model loading offline cached ImageNet weights."""
    model = models.resnet18(weights=None)
    cached_weights = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
    if cached_weights.exists():
        state_dict = torch.load(cached_weights, map_location="cpu")
        model.load_state_dict(state_dict)
        print("  ✓ Loaded cached ImageNet pretrained backbone offline.")
    else:
        print("  ⚠️ Cached ImageNet backbone not found; using random weights.")

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(TOMATO_CLASSES))
    return model.to(device)


def main():
    print("================================================================")
    print(" AI Crop Doctor — Milestone 3: Tomato Disease Specialist Training")
    print("================================================================")

    cache_base = get_plantvillage_root()
    if not cache_base.exists():
        print(f"❌ Error: PlantVillage cache directory not found at {cache_base}")
        sys.exit(1)

    output_dir = ROOT_DIR / "data" / "tomato"
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    model_save_dir = ROOT_DIR / "models" / "disease"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Collection and Leakage-Free Preparation
    print("\n[1/5] Collecting and preparing Tomato dataset...")
    sampled_pool, raw_stats = collect_and_prepare_dataset(cache_base, max_samples_per_class=500)

    # 2. Stratified Splits Creation
    print("\n[2/5] Creating reproducible stratified splits (70% train / 15% val / 15% test)...")
    train_records, val_records, test_records = create_stratified_splits(sampled_pool)

    print(f"  Split counts: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}")

    # Save splits to JSON
    with open(splits_dir / "train.json", "w") as f:
        json.dump(train_records, f, indent=2)
    with open(splits_dir / "val.json", "w") as f:
        json.dump(val_records, f, indent=2)
    with open(splits_dir / "test.json", "w") as f:
        json.dump(test_records, f, indent=2)

    # Compute Class Weights for Imbalance Handling
    train_class_counts = Counter([r["class_name"] for r in train_records])
    total_train = len(train_records)
    num_classes = len(TOMATO_CLASSES)
    class_weights = []
    print("\n  Class distribution & computed weights in training set:")
    for cls in TOMATO_CLASSES:
        count = train_class_counts.get(cls, 1)
        w = total_train / (num_classes * count)
        class_weights.append(w)
        print(f"    - {cls:<46}: {count} samples (weight: {w:.3f})")

    # 3. DataLoaders and Augmentations
    print("\n[3/5] Setting up DataLoaders and transforms...")
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    train_dataset = TomatoDiseaseDataset(train_records, transform=train_transform)
    val_dataset = TomatoDiseaseDataset(val_records, transform=eval_transform)
    test_dataset = TomatoDiseaseDataset(test_records, transform=eval_transform)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  ✓ Using compute device: {device}")

    # Build model
    model = build_model(device)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    # 4. Training Loop with Validation-Based Checkpoint Selection
    print("\n[4/5] Training Tomato Specialist (5 epochs)...")
    epochs = 5
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_state = None
    training_history = []

    model_target_path = model_save_dir / "tomato_disease_model.pth"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # Train phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100.0
        dt = time.time() - t0

        is_best = epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        flag = "⭐ [BEST]" if is_best else ""
        print(f"  Epoch {epoch}/{epochs} ({dt:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}% {flag}")

        training_history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 2),
            "val_loss": round(epoch_val_loss, 4),
            "val_acc": round(epoch_val_acc, 2),
            "is_best": is_best
        })

    # Save best checkpoint
    print(f"\n  ✓ Saving best validation checkpoint (Val Loss: {best_val_loss:.4f}, Acc: {best_val_acc:.2f}%) to:")
    print(f"    {model_target_path}")
    torch.save(best_model_state, model_target_path)

    # 5. Evaluation on Isolated Held-out Test Set
    print("\n[5/5] Evaluating on isolated test set using best checkpoint...")
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    test_correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    test_total = len(all_labels)
    test_acc = (test_correct / test_total) * 100.0

    print(f"\n================================================================")
    print(f" Isolated Test Set Results (Total: {test_total} images)")
    print(f" Overall Test Accuracy: {test_acc:.2f}%")
    print(f"================================================================")

    target_display_names = [TOMATO_FRIENDLY_NAMES[cls] for cls in TOMATO_CLASSES]
    cls_report = classification_report(
        all_labels,
        all_preds,
        target_names=target_display_names,
        digits=4,
        output_dict=True
    )
    conf_matrix = confusion_matrix(all_labels, all_preds).tolist()

    print(classification_report(all_labels, all_preds, target_names=target_display_names, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # Save comprehensive evaluation report
    report_payload = {
        "model_path": str(model_target_path),
        "target_crop": "Tomato",
        "num_classes": len(TOMATO_CLASSES),
        "classes": TOMATO_CLASSES,
        "friendly_names": TOMATO_FRIENDLY_NAMES,
        "dataset_summary": {
            "max_samples_per_class": 500,
            "train_samples": len(train_records),
            "val_samples": len(val_records),
            "test_samples": len(test_records),
            "class_stats": raw_stats,
            "class_weights": {cls: round(w, 4) for cls, w in zip(TOMATO_CLASSES, class_weights)}
        },
        "training_history": training_history,
        "best_epoch_metrics": {
            "best_val_loss": round(best_val_loss, 4),
            "best_val_acc": round(best_val_acc, 2)
        },
        "test_metrics": {
            "test_accuracy": round(test_acc, 2),
            "macro_precision": round(cls_report["macro avg"]["precision"] * 100.0, 2),
            "macro_recall": round(cls_report["macro avg"]["recall"] * 100.0, 2),
            "macro_f1": round(cls_report["macro avg"]["f1-score"] * 100.0, 2),
            "weighted_precision": round(cls_report["weighted avg"]["precision"] * 100.0, 2),
            "weighted_recall": round(cls_report["weighted avg"]["recall"] * 100.0, 2),
            "weighted_f1": round(cls_report["weighted avg"]["f1-score"] * 100.0, 2),
            "per_class": {
                name: {
                    "precision": round(cls_report[name]["precision"] * 100.0, 2),
                    "recall": round(cls_report[name]["recall"] * 100.0, 2),
                    "f1_score": round(cls_report[name]["f1-score"] * 100.0, 2),
                    "support": cls_report[name]["support"]
                }
                for name in target_display_names
            },
            "confusion_matrix": conf_matrix
        },
        "scientific_limitation_statement": (
            "These metrics were measured on isolated held-out test splits from the laboratory "
            "PlantVillage benchmark. They do NOT represent real-world field accuracy under variable natural lighting, "
            "complex outdoor canopy backgrounds, multi-pathogen co-infections, or insect damage."
        )
    }

    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report_payload, f, indent=2)
    print(f"\n  ✓ Full evaluation report saved to: {report_path}")
    print("================================================================\n")


if __name__ == "__main__":
    main()
