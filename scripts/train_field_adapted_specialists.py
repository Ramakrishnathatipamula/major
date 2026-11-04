"""
Milestone 6C: Controlled Training of Experimental Field-Adapted Disease Specialists.

Trains separate experimental field-adapted models:
  - models/disease/potato_field_adapted_model.pth
  - models/disease/tomato_field_adapted_model.pth

CRITICAL CONSTRAINTS:
1. Zero overwrite: Original checkpoints remain immutable baselines.
2. Leakage-safe: Uses ONLY the curated PlantDoc train manifests; all 11 test duplicates excluded.
3. Aspect-preserving: Uses Letterbox preprocessing during training and validation.
4. Records training history, validation metrics, and cryptographic SHA256 hashes.
"""

import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import settings
from modules.dataset_paths import resolve_dataset_path
from modules.model import build_resnet_model, get_device
from modules.image_utils import get_preprocessing_transform, IMAGENET_MEAN, IMAGENET_STD
from modules.disease_predictor import DEFAULT_CLASSES as POTATO_CLASSES, TOMATO_CLASSES

MANIFESTS_DIR = ROOT_DIR / "data" / "field_eval" / "manifests"
MODELS_DIR = ROOT_DIR / "models" / "disease"
REPORTS_DIR = ROOT_DIR / "data" / "field_eval" / "reports"

SEED = 42
torch.manual_seed(SEED)


class FieldManifestDataset(Dataset):
    """Loads images from a curated JSON manifest with letterbox transforms."""
    def __init__(self, manifest_path: Path, class_list: List[str], transform):
        with open(manifest_path) as f:
            data = json.load(f)
        self.items = data["images"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_list)}
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = resolve_dataset_path(item["path"])
        mapped_disease = item["mapped_disease"]
        label_idx = self.class_to_idx[mapped_disease]

        with Image.open(img_path) as raw:
            img = raw.convert("RGB")

        tensor = self.transform(img)
        return tensor, label_idx


def train_specialist(
    crop_name: str,
    train_manifest: Path,
    val_manifest: Path,
    base_checkpoint: Path,
    out_checkpoint: Path,
    class_list: List[str],
    epochs: int = 4,
    lr: float = 1e-4,
    batch_size: int = 16,
    device: torch.device = None
) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print(f"Training Experimental Field-Adapted Specialist: {crop_name.upper()}")
    print(f"{'='*70}")

    if device is None:
        device = get_device()
    print(f"Device: {device}")
    print(f"Base Checkpoint: {base_checkpoint.name}")
    print(f"Output Target:   {out_checkpoint.name}")

    # Use aspect-ratio preserving letterbox transform
    train_transform = get_preprocessing_transform(mode="letterbox", target_size=(224, 224))
    val_transform = get_preprocessing_transform(mode="letterbox", target_size=(224, 224))

    train_ds = FieldManifestDataset(train_manifest, class_list, train_transform)
    val_ds = FieldManifestDataset(val_manifest, class_list, val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Load baseline model
    model = build_resnet_model(num_classes=len(class_list), pretrained=False)
    state = torch.load(base_checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    # Freeze earlier layers, fine-tune layer3, layer4 and fc
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Initial validation accuracy before adaptation
    model.eval()
    init_correct = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            init_correct += (preds == y).sum().item()
    init_val_acc = (init_correct / len(val_ds) * 100.0) if len(val_ds) > 0 else 0.0
    print(f"Initial Baseline Validation Accuracy on Field Val: {init_val_acc:.2f}% ({init_correct}/{len(val_ds)})")

    history = []
    best_acc = init_val_acc
    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_correct = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            train_correct += (outputs.argmax(dim=1) == y).sum().item()

        epoch_loss = running_loss / len(train_ds)
        epoch_train_acc = (train_correct / len(train_ds) * 100.0)

        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                val_correct += (preds == y).sum().item()
        val_acc = (val_correct / len(val_ds) * 100.0)

        print(f"Epoch [{epoch}/{epochs}] Loss: {epoch_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_loss, 4),
            "train_acc": round(epoch_train_acc, 2),
            "val_acc": round(val_acc, 2)
        })

        if val_acc >= best_acc:
            best_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"Training completed in {time.time()-t0:.1f}s. Best Val Acc: {best_acc:.2f}%")

    # Save to NEW experimental checkpoint
    torch.save(best_weights, out_checkpoint)
    print(f"Saved experimental model to: {out_checkpoint}")

    # Compute SHA-256
    checkpoint_bytes = out_checkpoint.read_bytes()
    sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    md5 = hashlib.md5(checkpoint_bytes).hexdigest()
    print(f"Checkpoint SHA256: {sha256}")

    return {
        "crop": crop_name,
        "base_checkpoint": str(base_checkpoint),
        "output_checkpoint": str(out_checkpoint),
        "checkpoint_sha256": sha256,
        "checkpoint_md5": md5,
        "init_val_acc": round(init_val_acc, 2),
        "best_val_acc": round(best_acc, 2),
        "epochs": epochs,
        "history": history
    }


def main():
    device = torch.device("cpu")

    base_potato = MODELS_DIR / "potato_disease_model.pth"
    out_potato = MODELS_DIR / "potato_field_adapted_model.pth"

    base_tomato = MODELS_DIR / "tomato_disease_model.pth"
    out_tomato = MODELS_DIR / "tomato_field_adapted_model.pth"

    results = {}

    # 1. Train Field-Adapted Potato Specialist
    potato_res = train_specialist(
        crop_name="potato",
        train_manifest=MANIFESTS_DIR / "field_adaptation_potato_train.json",
        val_manifest=MANIFESTS_DIR / "field_adaptation_potato_val.json",
        base_checkpoint=base_potato,
        out_checkpoint=out_potato,
        class_list=POTATO_CLASSES,
        epochs=4,
        lr=1e-4,
        batch_size=16,
        device=device
    )
    results["potato"] = potato_res

    # 2. Train Field-Adapted Tomato Specialist
    tomato_res = train_specialist(
        crop_name="tomato",
        train_manifest=MANIFESTS_DIR / "field_adaptation_tomato_train.json",
        val_manifest=MANIFESTS_DIR / "field_adaptation_tomato_val.json",
        base_checkpoint=base_tomato,
        out_checkpoint=out_tomato,
        class_list=TOMATO_CLASSES,
        epochs=3,
        lr=1e-4,
        batch_size=16,
        device=device
    )
    results["tomato"] = tomato_res

    # Save summary
    out_report = REPORTS_DIR / "field_adapted_training_summary.json"
    with open(out_report, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved training summary to: {out_report}")


if __name__ == "__main__":
    main()
