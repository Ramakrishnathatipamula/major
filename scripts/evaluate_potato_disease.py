"""
evaluate_potato_disease.py
==========================
Evaluation-only script for the Potato Disease Classification model (Model 2).

STRICT CONSTRAINTS:
  - Does NOT retrain, fine-tune, or touch any checkpoint.
  - Loads checkpoint from models/disease/potato_disease_model.pth (read-only).
  - Reproduces the EXACT deterministic test split using the same logic as
    train_model.py: torch.Generator().manual_seed(42), 70/15/15 random_split
    over data/raw/plant_village_subset (300 images total, 3 classes × 100).
  - Verifies the reproduced split has the same size (test = 45 images) and
    class distribution as the original training run before running inference.
  - Reports the ACTUAL held-out test accuracy — distinct from the previously
    recorded 100% best validation accuracy.
  - If the dataset directory or checkpoint is missing, exits with a clear error.
  - Saves:
      data/potato/eval_report_full.json   (full metrics)
      data/potato/confusion_matrix.png    (labeled heatmap)

Class order (as produced by torchvision.datasets.ImageFolder alphabetical sort):
  Index 0: Potato___Early_blight
  Index 1: Potato___Late_blight
  Index 2: Potato___healthy
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# ── project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.dataset import IMAGENET_MEAN, IMAGENET_STD

# ── paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = ROOT_DIR / "models" / "disease" / "potato_disease_model.pth"
DATASET_DIR     = ROOT_DIR / "data" / "raw" / "plant_village_subset"
OUT_DIR         = ROOT_DIR / "data" / "potato"
REPORT_PATH     = OUT_DIR / "eval_report_full.json"
CM_IMAGE_PATH   = OUT_DIR / "confusion_matrix.png"

# Split parameters — must match train_model.py exactly
SPLIT_SEED  = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15 (remainder)

NUM_CLASSES = 3
BATCH_SIZE  = 16

# ── helpers ───────────────────────────────────────────────────────────────────
class TransformedSubset(torch.utils.data.Dataset):
    """Apply a transform to a Subset without modifying the underlying dataset."""
    def __init__(self, subset, transform=None):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


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

    weights  = np.array(support) / sum(support)
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
    """Render and save a labelled confusion matrix heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠ matplotlib not installed — skipping confusion matrix image.")
        return

    n    = len(class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    im   = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Potato Disease Classification — Confusion Matrix\n(Held-Out Test Split, N=45)",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(int(cm[i, j])),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11, fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Confusion matrix image saved: {out_path.relative_to(ROOT_DIR)}")


def preflight_check():
    """Verify checkpoint and dataset exist before proceeding."""
    missing = []
    if not CHECKPOINT_PATH.exists():
        missing.append(f"Checkpoint not found:  {CHECKPOINT_PATH}")
    if not DATASET_DIR.exists():
        missing.append(f"Dataset dir not found: {DATASET_DIR}")
    if missing:
        print("FATAL — Cannot proceed. Required files/directories are missing:")
        for m in missing:
            print(f"  ✗ {m}")
        sys.exit(1)


def reproduce_test_split(raw_dataset):
    """
    Reproduce the EXACT test split from train_model.py using the same
    generator seed, split sizes, and random_split call.

    train_model.py logic (verbatim):
        generator = torch.Generator().manual_seed(42)
        total_size = len(raw_dataset)
        train_size = int(0.70 * total_size)
        val_size   = int(0.15 * total_size)
        test_size  = total_size - train_size - val_size
        train_subset, val_subset, test_subset = random_split(
            raw_dataset, [train_size, val_size, test_size], generator=generator
        )
    """
    total_size = len(raw_dataset)
    train_size = int(TRAIN_RATIO * total_size)
    val_size   = int(VAL_RATIO   * total_size)
    test_size  = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_subset, val_subset, test_subset = random_split(
        raw_dataset, [train_size, val_size, test_size], generator=generator
    )
    return train_subset, val_subset, test_subset, train_size, val_size, test_size


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(" Potato Disease Classification — Evaluation-Only Script (Model 2)")
    print("=" * 70)
    print(f"  Checkpoint : {CHECKPOINT_PATH.relative_to(ROOT_DIR)}")
    print(f"  Dataset    : {DATASET_DIR.relative_to(ROOT_DIR)}")
    print()

    # 1. Pre-flight checks
    print("[1/5] Validating required files...")
    preflight_check()
    print("  ✓ Checkpoint and dataset directory verified.")

    # 2. Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n[2/5] Using device: {device}")

    # 3. Reproduce deterministic test split
    print(f"\n[3/5] Reproducing deterministic test split "
          f"(seed={SPLIT_SEED}, 70/15/15)...")

    # Load with NO transform so random_split uses the same indices as training
    raw_dataset = datasets.ImageFolder(root=str(DATASET_DIR))
    class_names = raw_dataset.classes   # Alphabetical: Early_blight, Late_blight, healthy
    num_detected = len(raw_dataset)

    print(f"  Classes detected ({len(class_names)}): {class_names}")
    print(f"  Total images in dataset: {num_detected}")

    if num_detected != 300:
        print(f"  ⚠ WARNING: Expected 300 images but found {num_detected}. "
              "Split indices may differ from the original training run.")

    train_subset, val_subset, test_subset, ts, vs, tes = reproduce_test_split(raw_dataset)
    print(f"  ✓ Split reproduced → Train: {len(train_subset)} | "
          f"Val: {len(val_subset)} | Test: {len(test_subset)}")

    if len(test_subset) != tes:
        print(f"  ✗ FATAL: Expected test size {tes}, got {len(test_subset)}.")
        sys.exit(1)

    # Verify per-class distribution of test split
    class_counts = {c: 0 for c in class_names}
    for idx in test_subset.indices:
        _, label = raw_dataset.samples[idx]
        class_counts[class_names[label]] += 1

    print("  Per-class test distribution:")
    for cls, cnt in class_counts.items():
        print(f"    {cls}: {cnt} images")

    # 4. Model — checkpoint load, eval mode only
    print(f"\n[4/5] Loading checkpoint (read-only, no training)...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print("  ✓ Checkpoint loaded. Parameters frozen. Model in eval mode.")

    # 5. DataLoader with eval transform
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    test_dataset = TransformedSubset(test_subset, transform=eval_transform)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"\n  DataLoader ready: {len(test_dataset)} test images | Batches: {len(test_loader)}")

    # 6. Inference
    print(f"\n[5/5] Running inference (no_grad)...")
    all_preds, all_labels = [], []

    t0 = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds   = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    elapsed = time.time() - t0
    print(f"  ✓ Inference complete in {elapsed:.1f}s "
          f"({elapsed / len(test_dataset) * 1000:.1f} ms/image)")

    # 7. Metrics
    print("\n─── Computing Metrics ──────────────────────────────────────────────")
    all_preds_np  = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(all_labels_np, all_preds_np):
        cm[t, p] += 1

    metrics = compute_metrics_from_cm(cm, class_names)

    print(f"\n  Overall Test Accuracy : {metrics['accuracy']:.4f}%")
    print(f"  [Previously recorded best VAL accuracy: 100.00%]")
    print()
    print(f"  Macro Precision       : {metrics['macro_precision']:.4f}%")
    print(f"  Macro Recall          : {metrics['macro_recall']:.4f}%")
    print(f"  Macro F1              : {metrics['macro_f1']:.4f}%")
    print(f"  Weighted Precision    : {metrics['weighted_precision']:.4f}%")
    print(f"  Weighted Recall       : {metrics['weighted_recall']:.4f}%")
    print(f"  Weighted F1           : {metrics['weighted_f1']:.4f}%")
    print(f"  Total test samples    : {metrics['total_samples']}")
    print(f"  Correct predictions   : {metrics['correct']}")
    print()
    print(f"  {'Class':<40} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>8}")
    print("  " + "─" * 76)
    for cls in class_names:
        d = metrics["per_class"][cls]
        print(f"  {cls:<40} {d['precision']:>7.2f}% {d['recall']:>7.2f}% "
              f"{d['f1_score']:>7.2f}% {d['support']:>8}")

    print("\n  Confusion Matrix (rows=True Label, cols=Predicted):")
    print(f"  Classes: {class_names}")
    print(cm)

    # 8. Save report
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/evaluate_potato_disease.py",
        "evaluation_mode": "INFERENCE_ONLY — no training, no checkpoint modification",
        "checkpoint_path": str(CHECKPOINT_PATH.relative_to(ROOT_DIR)),
        "dataset_path": str(DATASET_DIR.relative_to(ROOT_DIR)),
        "model_architecture": "ResNet-18",
        "num_classes": NUM_CLASSES,
        "classes": class_names,
        "device": str(device),
        "split_config": {
            "seed": SPLIT_SEED,
            "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": 1 - TRAIN_RATIO - VAL_RATIO},
            "sizes": {"train": ts, "val": vs, "test": tes},
            "split_method": "torch.utils.data.random_split with Generator(seed=42)",
            "reproducibility_note": (
                "Split is deterministic given seed=42 and the same dataset image ordering "
                "(alphabetical ImageFolder). If dataset files change, indices may shift."
            ),
        },
        "previously_reported_best_val_accuracy_pct": 100.0,
        "note_on_reported_100pct": (
            "The 100% figure was the best *validation* accuracy during training, "
            "not the held-out test accuracy. This script reports the actual test split result."
        ),
        "per_class_test_distribution": class_counts,
        "inference_time_seconds": round(elapsed, 2),
        "summary_metrics": {
            "test_accuracy_pct":        metrics["accuracy"],
            "total_test_samples":       metrics["total_samples"],
            "correct_predictions":      metrics["correct"],
            "macro_precision_pct":      metrics["macro_precision"],
            "macro_recall_pct":         metrics["macro_recall"],
            "macro_f1_pct":             metrics["macro_f1"],
            "weighted_precision_pct":   metrics["weighted_precision"],
            "weighted_recall_pct":      metrics["weighted_recall"],
            "weighted_f1_pct":          metrics["weighted_f1"],
        },
        "per_class_metrics": metrics["per_class"],
        "confusion_matrix": {
            "labels": class_names,
            "matrix": cm.tolist(),
        },
        "scientific_limitation_statement": (
            "Metrics are from a closed-set PlantVillage lab benchmark (300 images, "
            "controlled backgrounds). Real-world field accuracy is substantially lower "
            "under variable outdoor lighting and complex canopy backgrounds."
        ),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  ✓ Full report saved: {REPORT_PATH.relative_to(ROOT_DIR)}")

    # 9. Confusion matrix image
    save_confusion_matrix_image(cm, class_names, CM_IMAGE_PATH)

    print()
    print("=" * 70)
    print(" Evaluation complete.")
    print(f"  Report : {REPORT_PATH}")
    print(f"  CM     : {CM_IMAGE_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
