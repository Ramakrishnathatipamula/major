"""
Compute and save class feature prototypes for PlantIdentifier.

CRITICAL INTEGRITY REQUIREMENT:
Prototypes are derived ONLY from data/identification/splits/train.json.
Validation and test images are NEVER used to generate prototypes.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.identification.plant_identifier import (
    PLANT_SPECIES_CLASSES,
    PLANT_SPECIES_DISPLAY_NAMES
)
from modules.model import build_resnet_model, get_device
from modules.dataset import IMAGENET_MEAN, IMAGENET_STD
from modules.dataset_paths import resolve_dataset_path


def compute_and_save_prototypes(
    model_path: Path = ROOT_DIR / "models" / "identification" / "plant_id_model_15class.pth",
    train_manifest: Path = ROOT_DIR / "data" / "identification" / "splits_15class" / "train.json",
    output_path: Path = ROOT_DIR / "models" / "identification" / "crop_prototypes_15class.pt"
) -> Path:
    device = get_device()
    print(f"Loading PlantIdentifier checkpoint from: {model_path} on {device}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not train_manifest.exists():
        raise FileNotFoundError(f"Train manifest not found: {train_manifest}")

    model = build_resnet_model(num_classes=len(PLANT_SPECIES_CLASSES), pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Feature extractor: everything up to avgpool
    modules = list(model.children())
    backbone = nn.Sequential(*modules[:-1]).to(device)
    backbone.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    with open(train_manifest, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    print(f"Computing class prototypes strictly from {len(train_data)} train samples...")
    t0 = time.time()

    class_embeddings = {cls: [] for cls in PLANT_SPECIES_CLASSES}

    with torch.no_grad():
        for idx, item in enumerate(train_data):
            crop = item["crop_id"]
            if crop not in class_embeddings:
                continue
            img_path = resolve_dataset_path(item["path"])
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            feat = torch.flatten(backbone(tensor), 1)
            feat = feat / torch.norm(feat, p=2, dim=1, keepdim=True)
            class_embeddings[crop].append(feat.cpu())

    prototypes = {}
    counts = {}
    for cls in PLANT_SPECIES_CLASSES:
        if not class_embeddings[cls]:
            raise ValueError(f"No embeddings found for class {cls}")
        feats = torch.cat(class_embeddings[cls], dim=0) # [N_c, 512]
        mean_feat = torch.mean(feats, dim=0, keepdim=True)
        norm_proto = mean_feat / torch.norm(mean_feat, p=2, dim=1, keepdim=True)
        prototypes[cls] = norm_proto.squeeze(0) # [512]
        counts[cls] = len(class_embeddings[cls])

    proto_matrix = torch.stack([prototypes[cls] for cls in PLANT_SPECIES_CLASSES], dim=0)  # [num_classes, 512]

    # Verify prototype properties
    sim_matrix = torch.mm(proto_matrix, proto_matrix.t())
    diag = torch.diag(sim_matrix)
    off_diag = sim_matrix - torch.eye(len(PLANT_SPECIES_CLASSES)) * 2.0
    max_off_diag = off_diag.max().item()

    payload = {
        "classes": PLANT_SPECIES_CLASSES,
        "display_names": PLANT_SPECIES_DISPLAY_NAMES,
        "proto_matrix": proto_matrix,  # [num_classes, 512]
        "prototypes": {cls: prototypes[cls] for cls in PLANT_SPECIES_CLASSES},
        "sample_counts": counts,
        "total_train_samples": len(train_data),
        "feature_dim": proto_matrix.shape[1],
        "created_at": datetime.utcnow().isoformat() + "Z",
        "derived_from": str(train_manifest),
        "validation_excluded": True,
        "test_excluded": True,
        "max_inter_class_similarity": round(max_off_diag, 4)
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    elapsed = time.time() - t0
    print(f"Prototypes saved to {output_path} in {elapsed:.2f}s.")
    print(f"Matrix shape: {proto_matrix.shape}, Max off-diagonal cosine sim: {max_off_diag:.4f}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute plant-identifier feature prototypes")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT_DIR / "models" / "identification" / "plant_id_model_15class.pth",
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=ROOT_DIR / "data" / "identification" / "splits_15class" / "train.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "models" / "identification" / "crop_prototypes_15class.pt",
    )
    args = parser.parse_args()
    compute_and_save_prototypes(
        model_path=args.model,
        train_manifest=args.train_manifest,
        output_path=args.output,
    )
