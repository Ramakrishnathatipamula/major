"""
evaluate_plant_identifier.py
============================
Evaluation-only script for the Plant Species Identification model (Model 1).

STRICT CONSTRAINTS:
  - Does NOT retrain, fine-tune, or touch any checkpoint.
  - Loads checkpoint from the path passed by the caller (default: the 15-class
    plant_id_model_15class.pth, read-only).
  - Evaluates only on the pre-built, frozen test split passed by the caller.
  - If any required file is missing, the script exits with a clear error message
    rather than fabricating metrics.
  - Saves a full metrics JSON and a labelled confusion-matrix image.
"""

import json
import sys
import time
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ── project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.dataset import IMAGENET_MEAN, IMAGENET_STD
from modules.dataset_paths import resolve_dataset_path
from modules.identification.plant_identifier import (
    PLANT_SPECIES_CLASSES,
    PLANT_SPECIES_DISPLAY_NAMES,
)

# ── paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = ROOT_DIR / "models" / "identification" / "plant_id_model_15class.pth"
TEST_MANIFEST   = ROOT_DIR / "data" / "identification" / "splits_15class" / "test.json"
OUT_DIR         = ROOT_DIR / "data" / "identification"
REPORT_PATH     = OUT_DIR / "eval_report_full_15class.json"
CM_IMAGE_PATH   = OUT_DIR / "confusion_matrix_15class.png"

NUM_CLASSES = len(PLANT_SPECIES_CLASSES)
BATCH_SIZE  = 32

# ── dataset ───────────────────────────────────────────────────────────────────
class ManifestDataset(Dataset):
    """Load images from a JSON manifest produced by train_plant_identifier.py."""

    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(PLANT_SPECIES_CLASSES)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        label = self.class_to_idx[item["crop_id"]]
        with Image.open(resolve_dataset_path(item["path"])) as img:
            rgb = img.convert("RGB")
        if self.transform:
            rgb = self.transform(rgb)
        return rgb, label, item["crop_id"]


# ── helpers ───────────────────────────────────────────────────────────────────
def compute_metrics_from_cm(cm: np.ndarray, class_names):
    """Return per-class and aggregate precision / recall / F1 from a confusion matrix."""
    n = cm.shape[0]
    prec, rec, f1, support = [], [], [], []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        s  = int(cm[i, :].sum())
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        prec.append(float(p)); rec.append(float(r))
        f1.append(float(f));    support.append(s)

    total   = int(cm.sum())
    correct = int(cm.diagonal().sum())
    acc     = correct / total if total > 0 else 0.0

    weights = np.array(support) / sum(support)
    macro_p  = float(np.mean(prec))
    macro_r  = float(np.mean(rec))
    macro_f  = float(np.mean(f1))
    wt_p     = float(np.dot(weights, prec))
    wt_r     = float(np.dot(weights, rec))
    wt_f     = float(np.dot(weights, f1))

    per_class = {
        class_names[i]: {
            "precision": round(prec[i] * 100, 4),
            "recall":    round(rec[i]  * 100, 4),
            "f1_score":  round(f1[i]   * 100, 4),
            "support":   support[i],
        }
        for i in range(n)
    }
    return dict(
        accuracy=round(acc * 100, 4),
        total_samples=total,
        correct=correct,
        macro_precision=round(macro_p * 100, 4),
        macro_recall=round(macro_r * 100, 4),
        macro_f1=round(macro_f * 100, 4),
        weighted_precision=round(wt_p * 100, 4),
        weighted_recall=round(wt_r * 100, 4),
        weighted_f1=round(wt_f * 100, 4),
        per_class=per_class,
    )


def save_confusion_matrix_image(cm: np.ndarray, class_names, out_path: Path):
    """Render a colour-coded confusion matrix and save it as a PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("  ⚠ matplotlib not installed — skipping confusion matrix image.")
        return

    display_names = [PLANT_SPECIES_DISPLAY_NAMES.get(c, c) for c in class_names]
    n = len(display_names)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=display_names,
        yticklabels=display_names,
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Plant Species Identification — Confusion Matrix\n(Test Split)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(int(cm[i, j])),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Confusion matrix image saved: {out_path.relative_to(ROOT_DIR)}")


# ── guard checks ──────────────────────────────────────────────────────────────
def preflight_check():
    missing = []
    if not CHECKPOINT_PATH.exists():
        missing.append(f"Checkpoint not found:   {CHECKPOINT_PATH}")
    if not TEST_MANIFEST.exists():
        missing.append(f"Test manifest not found: {TEST_MANIFEST}")
    if missing:
        print("FATAL — Cannot proceed. Required files are missing:")
        for m in missing:
            print(f"  ✗ {m}")
        sys.exit(1)

    # Verify all image paths in the manifest exist
    with open(TEST_MANIFEST) as f:
        records = json.load(f)

    bad_paths = [r["path"] for r in records if not resolve_dataset_path(r["path"]).exists()]
    if bad_paths:
        print(f"FATAL — {len(bad_paths)} image paths in test manifest are missing on disk.")
        print("  First 5 missing:")
        for p in bad_paths[:5]:
            print(f"    {p}")
        sys.exit(1)

    return records


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    global CHECKPOINT_PATH, TEST_MANIFEST, REPORT_PATH, CM_IMAGE_PATH  # noqa: PLW0603

    parser = argparse.ArgumentParser(description="Evaluate the plant identifier without retraining")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--manifest", type=Path, default=TEST_MANIFEST)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--confusion-matrix", type=Path, default=CM_IMAGE_PATH)
    args = parser.parse_args()
    CHECKPOINT_PATH = args.checkpoint
    TEST_MANIFEST = args.manifest
    REPORT_PATH = args.report
    CM_IMAGE_PATH = args.confusion_matrix

    print("=" * 70)
    print(" Plant Species Identification — Evaluation-Only Script (Model 1)")
    print("=" * 70)
    print(f"  Checkpoint : {CHECKPOINT_PATH.relative_to(ROOT_DIR)}")
    print(f"  Test split : {TEST_MANIFEST.relative_to(ROOT_DIR)}")
    print()

    # 1. Pre-flight
    print("[1/5] Validating required files and image paths...")
    records = preflight_check()
    print(f"  ✓ Test manifest loaded: {len(records)} records, all image paths verified.")

    # 2. Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n[2/5] Using device: {device}")

    # 3. Model — load checkpoint with strict=True, eval mode only
    print(f"\n[3/5] Loading checkpoint (read-only, no training)...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Freeze all parameters — purely defensive
    for p in model.parameters():
        p.requires_grad = False

    print("  ✓ Checkpoint loaded. Parameters frozen. Model in eval mode.")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Total parameters: {total_params:,}")

    # 4. DataLoader
    print(f"\n[4/5] Building DataLoader from frozen test manifest...")
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    dataset = ManifestDataset(records, transform=eval_transform)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  ✓ Dataset: {len(dataset)} images | Batches: {len(loader)}")

    # 5. Inference
    print(f"\n[5/5] Running inference (no_grad) ...")
    all_preds, all_labels, all_crop_ids = [], [], []

    t0 = time.time()
    with torch.no_grad():
        for batch_imgs, batch_labels, batch_crop_ids in loader:
            batch_imgs = batch_imgs.to(device)
            outputs    = model(batch_imgs)
            preds      = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.tolist())
            all_crop_ids.extend(batch_crop_ids)

    elapsed = time.time() - t0
    print(f"  ✓ Inference complete in {elapsed:.1f}s "
          f"({elapsed / len(dataset) * 1000:.1f} ms/image)")

    # 6. Metrics
    print("\n─── Computing Metrics ──────────────────────────────────────────────")
    all_preds_np  = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    # Confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(all_labels_np, all_preds_np):
        cm[t, p] += 1

    metrics = compute_metrics_from_cm(cm, PLANT_SPECIES_CLASSES)

    # Per-crop breakdown with display names
    per_crop_display = {}
    for idx, cls in enumerate(PLANT_SPECIES_CLASSES):
        display = PLANT_SPECIES_DISPLAY_NAMES.get(cls, cls)
        per_crop_display[display] = metrics["per_class"][cls]

    print(f"\n  Overall Accuracy      : {metrics['accuracy']:.4f}%")
    print(f"  Macro Precision       : {metrics['macro_precision']:.4f}%")
    print(f"  Macro Recall          : {metrics['macro_recall']:.4f}%")
    print(f"  Macro F1              : {metrics['macro_f1']:.4f}%")
    print(f"  Weighted Precision    : {metrics['weighted_precision']:.4f}%")
    print(f"  Weighted Recall       : {metrics['weighted_recall']:.4f}%")
    print(f"  Weighted F1           : {metrics['weighted_f1']:.4f}%")
    print(f"  Total test samples    : {metrics['total_samples']}")
    print(f"  Correct predictions   : {metrics['correct']}")
    print()
    print(f"  {'Crop':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
    print("  " + "─" * 60)
    for crop_cls in PLANT_SPECIES_CLASSES:
        d       = metrics["per_class"][crop_cls]
        display = PLANT_SPECIES_DISPLAY_NAMES.get(crop_cls, crop_cls)
        print(f"  {display:<16} {d['precision']:>9.2f}% {d['recall']:>9.2f}% "
              f"{d['f1_score']:>9.2f}% {d['support']:>8}")

    # 7. Save report JSON
    report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/evaluate_plant_identifier.py",
        "evaluation_mode": "INFERENCE_ONLY — no training, no checkpoint modification",
        "checkpoint_path": str(CHECKPOINT_PATH.relative_to(ROOT_DIR)),
        "test_manifest_path": str(TEST_MANIFEST.relative_to(ROOT_DIR)),
        "model_architecture": "ResNet-18",
        "num_classes": NUM_CLASSES,
        "classes": PLANT_SPECIES_CLASSES,
        "display_names": PLANT_SPECIES_DISPLAY_NAMES,
        "device": str(device),
        "inference_time_seconds": round(elapsed, 2),
        "summary_metrics": {
            "accuracy_pct":           metrics["accuracy"],
            "total_test_samples":     metrics["total_samples"],
            "correct_predictions":    metrics["correct"],
            "macro_precision_pct":    metrics["macro_precision"],
            "macro_recall_pct":       metrics["macro_recall"],
            "macro_f1_pct":           metrics["macro_f1"],
            "weighted_precision_pct": metrics["weighted_precision"],
            "weighted_recall_pct":    metrics["weighted_recall"],
            "weighted_f1_pct":        metrics["weighted_f1"],
        },
        "per_class_metrics": per_crop_display,
        "confusion_matrix": {
            "labels": [PLANT_SPECIES_DISPLAY_NAMES.get(c, c) for c in PLANT_SPECIES_CLASSES],
            "matrix": cm.tolist(),
        },
        "scientific_limitation_statement": (
            "Metrics are from a closed-set PlantVillage lab benchmark. "
            "Real-world field accuracy will differ due to natural lighting variation, "
            "complex outdoor backgrounds, and novel species not present in training."
        ),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  ✓ Full report saved: {REPORT_PATH.relative_to(ROOT_DIR)}")

    # 8. Save confusion matrix image
    save_confusion_matrix_image(cm, PLANT_SPECIES_CLASSES, CM_IMAGE_PATH)

    print()
    print("=" * 70)
    print(" Evaluation complete.")
    print(f"  Report : {REPORT_PATH}")
    print(f"  CM     : {CM_IMAGE_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
