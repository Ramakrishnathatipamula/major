import os
import sys
import json
import time
import hashlib
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import settings
from modules.dataset import IMAGENET_MEAN, IMAGENET_STD
from modules.dataset_paths import get_plantvillage_root, resolve_dataset_path, to_portable_path
from modules.identification.plant_identifier import (
    PLANT_SPECIES_CLASSES,
    PLANT_SPECIES_DISPLAY_NAMES,
)

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

# Wheat is not present in PlantVillage. These folder-name rules map a public
# field dataset into the canonical Wheat classes used by the specialist model.
WHEAT_SOURCE_DIR = ROOT_DIR / "data" / "wheat" / "raw"
WHEAT_FOLDER_CLASS_RULES = [
    ("septoria", "Wheat___Septoria_leaf_blotch"),
    ("stripe", "Wheat___Stripe_rust"),
    ("yellow", "Wheat___Stripe_rust"),
    ("healthy", "Wheat___healthy"),
]

# Canonical crop mapping from PlantVillage directory prefix
PLANTVILLAGE_CROP_MAP = {
    "Apple": "apple",
    "Blueberry": "blueberry",
    "Cherry_(including_sour)": "cherry",
    "Corn_(maize)": "maize",
    "Grape": "grape",
    "Orange": "orange",
    "Peach": "peach",
    "Pepper,_bell": "bell_pepper",
    "Potato": "potato",
    "Raspberry": "raspberry",
    "Soybean": "soybean",
    "Squash": "squash",
    "Strawberry": "strawberry",
    "Tomato": "tomato"
}


class PlantIdentificationDataset(Dataset):
    """PyTorch Dataset loading images from manifest paths with transforms."""
    def __init__(self, records: List[Dict[str, Any]], transform=None):
        self.records = records
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(PLANT_SPECIES_CLASSES)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        image_path = resolve_dataset_path(item["path"])
        crop_id = item["crop_id"]
        label = self.class_to_idx[crop_id]

        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")

        if self.transform:
            rgb_img = self.transform(rgb_img)

        return rgb_img, label


def _wheat_class_from_path(path: Path) -> str:
    """Map a Wheat dataset path to a canonical class using folder-name tokens."""
    relative_parts = [part.casefold() for part in path.relative_to(WHEAT_SOURCE_DIR).parts[:-1]]
    joined = " ".join(relative_parts)
    for token, class_name in WHEAT_FOLDER_CLASS_RULES:
        if token in joined:
            return class_name
    return ""


def collect_wheat_samples(
    crop_samples: Dict[str, List[Dict[str, Any]]],
    seen_hashes: set,
    samples_per_crop: int = 120,
) -> Dict[str, int]:
    """Collect Wheat images from the separately downloaded field dataset."""
    stats: Dict[str, int] = defaultdict(int)
    if not WHEAT_SOURCE_DIR.exists():
        print(f"  ⚠️ Wheat source directory not found: {WHEAT_SOURCE_DIR}")
        return stats

    image_paths = []
    for pattern in ("**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.JPG", "**/*.JPEG", "**/*.PNG"):
        image_paths.extend(WHEAT_SOURCE_DIR.glob(pattern))
    image_paths = sorted(set(image_paths))

    candidates_by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for image_path in image_paths:
        class_name = _wheat_class_from_path(image_path)
        if not class_name:
            continue
        try:
            with open(image_path, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            candidates_by_class[class_name].append(
                {
                    "path": to_portable_path(image_path),
                    "crop_id": "wheat",
                    "disease_class": class_name,
                    "md5": h,
                }
            )
            stats[class_name] += 1
        except Exception:
            continue

    # Round-robin across classes so the 120-image cap does not collapse onto
    # the largest Wheat class.
    for class_name in candidates_by_class:
        random.seed(SEED)
        random.shuffle(candidates_by_class[class_name])
    selected: List[Dict[str, Any]] = []
    indices = {class_name: 0 for class_name in candidates_by_class}
    while len(selected) < samples_per_crop:
        added = False
        for class_name in sorted(candidates_by_class):
            idx = indices[class_name]
            if idx < len(candidates_by_class[class_name]):
                selected.append(candidates_by_class[class_name][idx])
                indices[class_name] += 1
                added = True
                if len(selected) >= samples_per_crop:
                    break
        if not added:
            break

    crop_samples["wheat"] = selected
    stats["retained"] = len(selected)
    return stats


def collect_and_deduplicate_dataset(
    cache_dir: Path,
    samples_per_crop: int = 120
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Collect and deduplicate real leaf images per crop across all 15 species.
    Uses round-robin sampling across healthy and disease subdirectories so the plant
    identifier learns invariant plant species features rather than single-disease shortcuts.
    """
    crop_subdirs = defaultdict(list)
    for split in ["train", "val"]:
        split_dir = cache_dir / split
        if not split_dir.exists():
            continue
        for d in sorted(split_dir.iterdir()):
            if d.is_dir():
                raw_crop = d.name.split("___")[0]
                if raw_crop in PLANTVILLAGE_CROP_MAP:
                    crop_subdirs[PLANTVILLAGE_CROP_MAP[raw_crop]].append(d)

    seen_hashes = set()
    crop_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for crop_id, subdirs in sorted(crop_subdirs.items()):
        subdir_images = []
        for sd in subdirs:
            imgs = sorted(list(sd.glob("*.JPG")) + list(sd.glob("*.jpg")))
            subdir_images.append((sd.name, imgs))

        idx = 0
        while len(crop_samples[crop_id]) < samples_per_crop:
            added_in_round = 0
            for sd_name, imgs in subdir_images:
                if idx < len(imgs):
                    img_path = imgs[idx]
                    try:
                        with open(img_path, "rb") as f:
                            h = hashlib.md5(f.read()).hexdigest()
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            crop_samples[crop_id].append({
                                "path": to_portable_path(img_path),
                                "crop_id": crop_id,
                                "disease_class": sd_name,
                                "md5": h
                            })
                            added_in_round += 1
                        if len(crop_samples[crop_id]) >= samples_per_crop:
                            break
                    except Exception:
                        continue
            idx += 1
            if added_in_round == 0:
                break

    wheat_stats = collect_wheat_samples(crop_samples, seen_hashes, samples_per_crop)
    if wheat_stats:
        print(f"  ✓ Wheat source: {dict(wheat_stats)}")

    return crop_samples


def create_stratified_splits(
    crop_samples: Dict[str, List[Dict[str, Any]]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create reproducible, stratified train/validation/test splits.
    Ensures held-out test split is strictly isolated.
    """
    train_set = []
    val_set = []
    test_set = []

    for crop_id in sorted(crop_samples.keys()):
        items = list(crop_samples[crop_id])
        random.seed(SEED)
        random.shuffle(items)

        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_set.extend(items[:n_train])
        val_set.extend(items[n_train:n_train + n_val])
        test_set.extend(items[n_train + n_val:])

    return train_set, val_set, test_set


def build_plant_id_model(num_classes: int = 15) -> nn.Module:
    """Build ResNet-18 model loading offline cached weights."""
    model = models.resnet18(weights=None)
    weights_path = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("  ✓ Loaded cached ImageNet pretrained backbone offline.")
    else:
        print("  ⚠️ Cached weights not found; training from random initialization.")

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train 15-class plant identifier")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=ROOT_DIR / "data" / "identification" / "splits_15class",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=ROOT_DIR / "models" / "identification" / "plant_id_model_15class.pth",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=ROOT_DIR / "data" / "identification" / "evaluation_report_15class.json",
    )
    parser.add_argument("--samples-per-crop", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()

    print("================================================================")
    print(" AI Crop Doctor — Stage 1 Plant Identification Model Training")
    print("================================================================")

    cache_dir = get_plantvillage_root()
    if not cache_dir.exists():
        print(f"❌ Error: PlantVillage cache not found at {cache_dir}")
        sys.exit(1)

    output_dir = ROOT_DIR / "data" / "identification"
    splits_dir = args.split_dir
    splits_dir.mkdir(parents=True, exist_ok=True)
    models_dir = args.output_model.parent
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Collection and Deduplication
    print("\n[1/5] Collecting and deduplicating dataset from cache...")
    crop_samples = collect_and_deduplicate_dataset(cache_dir, samples_per_crop=args.samples_per_crop)
    total_images = sum(len(v) for v in crop_samples.values())
    print(f"  ✓ Sampled {total_images} deduplicated images across {len(crop_samples)} crops.")

    # 2. Stratified Splits
    print("\n[2/5] Creating stratified Train/Validation/Test splits (70/15/15)...")
    train_records, val_records, test_records = create_stratified_splits(crop_samples)
    print(f"  ✓ Train split:      {len(train_records)} images")
    print(f"  ✓ Validation split: {len(val_records)} images")
    print(f"  ✓ Held-out Test:    {len(test_records)} images (ISOLATED)")

    # Save split manifests
    for name, data in [("train", train_records), ("val", val_records), ("test", test_records)]:
        p = splits_dir / f"{name}.json"
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved split manifest: {p.relative_to(ROOT_DIR)}")

    # 3. DataLoaders and Augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    train_loader = DataLoader(PlantIdentificationDataset(train_records, train_transform), batch_size=32, shuffle=True)
    val_loader = DataLoader(PlantIdentificationDataset(val_records, eval_transform), batch_size=32, shuffle=False)
    test_loader = DataLoader(PlantIdentificationDataset(test_records, eval_transform), batch_size=32, shuffle=False)

    # 4. Device and Model Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[3/5] Instantiating ResNet-18 on device: {device}")
    model = build_plant_id_model(num_classes=len(PLANT_SPECIES_CLASSES))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 5. Training Loop
    epochs = args.epochs
    print(f"\n[4/5] Training for {epochs} epochs...")
    best_val_acc = 0.0
    checkpoint_path = args.output_model

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_acc = (train_correct / total_train) * 100.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_acc = (val_correct / total_val) * 100.0
        print(f"  Epoch {epoch}/{epochs} — Train Acc: {epoch_train_acc:.2f}%, Val Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), checkpoint_path)

    elapsed = time.time() - start_time
    print(f"  ✓ Training complete in {elapsed:.1f}s. Checkpoint saved to: {checkpoint_path.relative_to(ROOT_DIR)}")

    # 6. Evaluation on Held-Out Test Set
    print("\n[5/5] Evaluating on Held-Out Test Set (ISOLATED)...")
    # Load best checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    test_correct = 0
    total_test = 0
    per_crop_correct = defaultdict(int)
    per_crop_total = defaultdict(int)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for p, l in zip(preds, labels):
                crop_cls = PLANT_SPECIES_CLASSES[l.item()]
                per_crop_total[crop_cls] += 1
                if p == l:
                    test_correct += 1
                    per_crop_correct[crop_cls] += 1
            total_test += labels.size(0)

    overall_test_acc = (test_correct / total_test) * 100.0
    print(f"  Overall Top-1 Test Accuracy: {overall_test_acc:.2f}% ({test_correct}/{total_test})")

    per_crop_acc = {}
    for crop_cls in PLANT_SPECIES_CLASSES:
        acc = (per_crop_correct[crop_cls] / per_crop_total[crop_cls]) * 100.0 if per_crop_total[crop_cls] > 0 else 0.0
        display = PLANT_SPECIES_DISPLAY_NAMES[crop_cls]
        per_crop_acc[display] = round(acc, 2)
        print(f"    {display:15s}: {acc:6.2f}% ({per_crop_correct[crop_cls]}/{per_crop_total[crop_cls]})")

    # Save evaluation report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_architecture": "ResNet-18",
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT_DIR)),
        "dataset_source": "PlantVillage Local Cache plus public Wheat field dataset",
        "num_classes": len(PLANT_SPECIES_CLASSES),
        "split_counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records)
        },
        "training_time_seconds": round(elapsed, 2),
        "overall_test_accuracy": round(overall_test_acc, 2),
        "per_crop_test_accuracy": per_crop_acc,
        "train_images_per_crop": {
            crop_id: sum(1 for r in train_records if r["crop_id"] == crop_id)
            for crop_id in PLANT_SPECIES_CLASSES
        },
        "wheat_source": {
            "path": "data/wheat/raw",
            "description": "Kaggle olyadgetch/wheat-leaf-dataset (field-acquired, Ethiopia)",
            "classes": {
                cls: sum(1 for r in train_records if r.get("disease_class") == cls)
                for cls in sorted({r.get("disease_class") for r in train_records if r["crop_id"] == "wheat"})
            },
        },
        "rejection_limitations": {
            "closed_set_caveat": "Model is trained on 15 cultivated agricultural species. Out-of-distribution species require confidence threshold rejection and LeafVerifier screening.",
            "provisional_thresholds": "Confidence thresholds (60% routing, 40% rejection) are provisional and must be verified on wild field photos."
        }
    }

    report_path = args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  ✓ Saved evaluation report to: {report_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
