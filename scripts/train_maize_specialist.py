"""
Train a Maize (Corn) disease specialist from the cached PlantVillage Corn classes.

Safety / integrity rules:
- New checkpoint only. Existing Potato/Tomato checkpoints are never modified.
- MD5 de-duplication prevents identical images crossing splits.
- Images used by the Stage 1 plant-identifier splits are excluded to keep the
  identifier's held-out test set independent from specialist training.
- Metrics are laboratory PlantVillage metrics, not field accuracy.
"""

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.dataset import IMAGENET_MEAN, IMAGENET_STD
from modules.dataset_paths import (
    get_plantvillage_root,
    resolve_dataset_path,
    to_portable_path,
)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MAIZE_CLASSES = [
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
]

MAIZE_FRIENDLY_NAMES = {
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Gray Leaf Spot",
    "Corn_(maize)___Common_rust_": "Common Rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Northern Leaf Blight",
    "Corn_(maize)___healthy": "Healthy",
}


class MaizeDiseaseDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], transform=None):
        self.records = records
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(MAIZE_CLASSES)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        item = self.records[idx]
        with Image.open(resolve_dataset_path(item["path"])) as img:
            rgb_img = img.convert("RGB")
        if self.transform:
            rgb_img = self.transform(rgb_img)
        return rgb_img, self.class_to_idx[item["class_name"]]


def get_identifier_leakage_hashes(crop_ids: Tuple[str, ...] = ("maize",)) -> set:
    """Collect MD5 hashes used by Stage 1 splits for the given crop IDs."""
    leakage_hashes = set()
    split_dir = ROOT_DIR / "data" / "identification" / "splits"
    for split_name in ("train", "val", "test"):
        split_path = split_dir / f"{split_name}.json"
        if not split_path.exists():
            continue
        with open(split_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        for rec in records:
            if rec.get("crop_id") in crop_ids and rec.get("md5"):
                leakage_hashes.add(rec["md5"])
    return leakage_hashes


def collect_and_prepare_dataset(
    cache_base: Path,
    max_samples_per_class: int = 500,
    leakage_hashes: set = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    leakage_hashes = leakage_hashes or set()
    print(f"  ✓ Excluding {len(leakage_hashes)} plant-identifier Maize hashes from specialist training.")

    raw_stats: Dict[str, Any] = {}
    dedup_pool: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_hashes = set()

    for cls in MAIZE_CLASSES:
        raw_files: List[Path] = []
        for split in ("train", "val"):
            cls_dir = cache_base / split / cls
            if cls_dir.exists():
                raw_files.extend(list(cls_dir.glob("*.JPG")) + list(cls_dir.glob("*.jpg")))
        raw_files = sorted(raw_files)
        raw_stats[cls] = {
            "total_files_on_disk": len(raw_files),
            "duplicates_removed": 0,
            "identifier_leakage_excluded": 0,
            "retained": 0,
        }

        for file_path in raw_files:
            try:
                with open(file_path, "rb") as f:
                    h = hashlib.md5(f.read()).hexdigest()
                if h in seen_hashes:
                    raw_stats[cls]["duplicates_removed"] += 1
                    continue
                if h in leakage_hashes:
                    raw_stats[cls]["identifier_leakage_excluded"] += 1
                    continue
                seen_hashes.add(h)
                dedup_pool[cls].append(
                    {
                        "path": to_portable_path(file_path),
                        "class_name": cls,
                        "md5": h,
                    }
                )
            except Exception as exc:
                print(f"    ⚠️ Warning reading {file_path}: {exc}")

        items = list(dedup_pool[cls])
        random.seed(SEED)
        random.shuffle(items)
        if max_samples_per_class and len(items) > max_samples_per_class:
            items = items[:max_samples_per_class]
        dedup_pool[cls] = items
        raw_stats[cls]["unique_clean_count"] = len(items)
        raw_stats[cls]["retained"] = len(items)
        print(
            f"    - {cls}: {raw_stats[cls]['total_files_on_disk']} raw -> "
            f"{len(items)} retained"
        )

    return dedup_pool, raw_stats


def create_stratified_splits(
    dataset_pool: Dict[str, List[Dict[str, Any]]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_set: List[Dict[str, Any]] = []
    val_set: List[Dict[str, Any]] = []
    test_set: List[Dict[str, Any]] = []

    for cls in MAIZE_CLASSES:
        items = list(dataset_pool[cls])
        random.seed(SEED)
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_set.extend(items[:n_train])
        val_set.extend(items[n_train:n_train + n_val])
        test_set.extend(items[n_train + n_val:])

    random.seed(SEED)
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)
    return train_set, val_set, test_set


def build_model(device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    cached_weights = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
    if cached_weights.exists():
        model.load_state_dict(torch.load(cached_weights, map_location="cpu"))
        print("  ✓ Loaded cached ImageNet pretrained backbone offline.")
    else:
        print("  ⚠️ Cached ImageNet backbone not found; using random initialization.")
    model.fc = nn.Linear(model.fc.in_features, len(MAIZE_CLASSES))
    return model.to(device)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, List[int], List[int]]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    accuracy = (correct / len(all_labels) * 100.0) if all_labels else 0.0
    return accuracy, all_preds, all_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Maize disease specialist")
    parser.add_argument(
        "--output-model",
        type=Path,
        default=ROOT_DIR / "models" / "disease" / "maize_disease_model.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "data" / "maize",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--max-samples-per-class", type=int, default=500)
    args = parser.parse_args()

    print("================================================================")
    print(" AI Crop Doctor — Maize Disease Specialist Training")
    print("================================================================")

    cache_base = get_plantvillage_root()
    if not cache_base.exists():
        print(f"❌ Error: PlantVillage cache directory not found at {cache_base}")
        sys.exit(1)

    output_dir = args.output_dir
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Collecting and preparing Maize dataset...")
    leakage_hashes = get_identifier_leakage_hashes(("maize",))
    sampled_pool, raw_stats = collect_and_prepare_dataset(
        cache_base,
        max_samples_per_class=args.max_samples_per_class,
        leakage_hashes=leakage_hashes,
    )

    print("\n[2/5] Creating reproducible stratified splits (70/15/15)...")
    train_records, val_records, test_records = create_stratified_splits(sampled_pool)
    print(
        f"  Split counts: Train={len(train_records)}, "
        f"Val={len(val_records)}, Test={len(test_records)}"
    )
    for name, records in (("train", train_records), ("val", val_records), ("test", test_records)):
        with open(splits_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    train_counts = Counter(r["class_name"] for r in train_records)
    total_train = len(train_records)
    class_weights = []
    print("\n  Class distribution and weights:")
    for cls in MAIZE_CLASSES:
        count = train_counts.get(cls, 1)
        weight = total_train / (len(MAIZE_CLASSES) * count)
        class_weights.append(weight)
        print(f"    - {cls:<58}: {count} samples (weight {weight:.3f})")

    print("\n[3/5] Setting up DataLoaders and transforms...")
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    batch_size = 64
    train_loader = DataLoader(
        MaizeDiseaseDataset(train_records, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        MaizeDiseaseDataset(val_records, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        MaizeDiseaseDataset(test_records, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  ✓ Using compute device: {device}")

    model = build_model(device)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n[4/5] Training Maize specialist ({args.epochs} epochs)...")
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = None
    history: List[Dict[str, Any]] = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
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
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

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
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_val_loss = val_loss / val_total
        epoch_train_acc = train_correct / train_total * 100.0
        epoch_val_acc = val_correct / val_total * 100.0
        is_best = epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"  Epoch {epoch}/{args.epochs} ({time.time() - t0:.1f}s) | "
            f"Train Loss {epoch_train_loss:.4f}, Acc {epoch_train_acc:.2f}% | "
            f"Val Loss {epoch_val_loss:.4f}, Acc {epoch_val_acc:.2f}%"
            f"{' ⭐' if is_best else ''}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(epoch_train_loss, 4),
                "train_acc": round(epoch_train_acc, 2),
                "val_loss": round(epoch_val_loss, 4),
                "val_acc": round(epoch_val_acc, 2),
                "is_best": is_best,
            }
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    torch.save(best_state, args.output_model)
    print(f"\n  ✓ Best checkpoint saved to: {args.output_model}")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print("\n[5/5] Evaluating on isolated held-out test split...")
    test_acc, all_preds, all_labels = evaluate(model, test_loader, device)
    target_display_names = [MAIZE_FRIENDLY_NAMES[cls] for cls in MAIZE_CLASSES]
    cls_report = classification_report(
        all_labels,
        all_preds,
        target_names=target_display_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    conf_matrix = confusion_matrix(all_labels, all_preds).tolist()

    print(f"  Overall Test Accuracy: {test_acc:.2f}%")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=target_display_names,
        digits=4,
        zero_division=0,
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_path": str(args.output_model.relative_to(ROOT_DIR))
        if args.output_model.is_relative_to(ROOT_DIR)
        else str(args.output_model),
        "target_crop": "Maize",
        "num_classes": len(MAIZE_CLASSES),
        "classes": MAIZE_CLASSES,
        "friendly_names": MAIZE_FRIENDLY_NAMES,
        "dataset_source": "PlantVillage cached Corn_(maize) classes",
        "dataset_summary": {
            "max_samples_per_class": args.max_samples_per_class,
            "identifier_leakage_hashes_excluded": len(leakage_hashes),
            "train_samples": len(train_records),
            "val_samples": len(val_records),
            "test_samples": len(test_records),
            "class_stats": raw_stats,
            "class_weights": {
                cls: round(weight, 4) for cls, weight in zip(MAIZE_CLASSES, class_weights)
            },
        },
        "training_history": history,
        "training_time_seconds": round(time.time() - started, 2),
        "best_epoch_metrics": {
            "best_val_loss": round(best_val_loss, 4),
            "best_val_acc": round(best_val_acc, 2),
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
                    "support": cls_report[name]["support"],
                }
                for name in target_display_names
            },
            "confusion_matrix": conf_matrix,
        },
        "scientific_limitation_statement": (
            "These are isolated held-out PlantVillage laboratory metrics, not field accuracy. "
            "Natural lighting, complex backgrounds, co-infections, and insect damage can change performance."
        ),
    }
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ Evaluation report saved to: {report_path}")
    print("================================================================\n")


if __name__ == "__main__":
    main()
