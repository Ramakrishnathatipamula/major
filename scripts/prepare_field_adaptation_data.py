"""
Milestone 6C: Prepare leakage-safe field adaptation splits and conduct data-quality audit.

Audits PlantDoc training images:
1. Detects exact MD5 duplicates within train partition.
2. Identifies and excludes cross-split duplicates with the official test partition (N=11).
3. Audits cross-class label contradictions.
4. Curates leakage-safe train/val manifests for Potato and Tomato field adaptation.

Outputs:
- data/field_eval/reports/label_quality_audit.json
- data/field_eval/manifests/field_adaptation_potato_train.json
- data/field_eval/manifests/field_adaptation_potato_val.json
- data/field_eval/manifests/field_adaptation_tomato_train.json
- data/field_eval/manifests/field_adaptation_tomato_val.json
"""

import sys
import json
import hashlib
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

BENCHMARK_MANIFEST = ROOT_DIR / "data" / "field_eval" / "manifests" / "field_benchmark.json"
REPORTS_DIR = ROOT_DIR / "data" / "field_eval" / "reports"
MANIFESTS_DIR = ROOT_DIR / "data" / "field_eval" / "manifests"

SEED = 42
random.seed(SEED)


def compute_md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    with open(BENCHMARK_MANIFEST) as f:
        manifest = json.load(f)

    all_images = manifest["images"]
    train_images = [img for img in all_images if img["plantdoc_split"] == "train"]
    test_images = [img for img in all_images if img["plantdoc_split"] == "test"]

    print(f"Loaded {len(train_images)} train images and {len(test_images)} test images.")

    # 1. Cross-split duplicate detection (MD5 hash check)
    test_hashes = {img["md5"]: img for img in test_images}
    cross_split_duplicates = []
    for img in train_images:
        h = img["md5"]
        if h in test_hashes:
            test_match = test_hashes[h]
            cross_split_duplicates.append({
                "hash": h,
                "train_path": img["relative_path"],
                "train_class": img["plantdoc_class"],
                "test_path": test_match["relative_path"],
                "test_class": test_match["plantdoc_class"],
                "is_label_contradiction": img["plantdoc_class"] != test_match["plantdoc_class"]
            })

    print(f"\n[Audit 1] Cross-split duplicates with official test set: {len(cross_split_duplicates)}")
    cross_dup_hashes = {d["hash"] for d in cross_split_duplicates}

    # 2. Internal duplicates within train split
    train_hash_to_imgs = defaultdict(list)
    for img in train_images:
        train_hash_to_imgs[img["md5"]].append(img)

    internal_duplicates = []
    for h, imgs in train_hash_to_imgs.items():
        if len(imgs) > 1:
            classes = [img["plantdoc_class"] for img in imgs]
            internal_duplicates.append({
                "hash": h,
                "count": len(imgs),
                "paths": [img["relative_path"] for img in imgs],
                "classes": classes,
                "is_label_contradiction": len(set(classes)) > 1
            })

    print(f"[Audit 2] Internal hash duplicates within train split: {len(internal_duplicates)}")

    # 3. Label mapping and class distribution analysis
    potato_classes = {
        "Potato leaf early blight": "Potato___Early_blight",
        "Potato leaf late blight": "Potato___Late_blight"
    }
    tomato_classes = {
        "Tomato Early blight leaf": "Tomato___Early_blight",
        "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
        "Tomato leaf late blight": "Tomato___Late_blight",
        "Tomato mold leaf": "Tomato___Leaf_Mold",
        "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
        "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
        "Tomato leaf": "Tomato___healthy",
        "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite"
    }

    # Clean Potato and Tomato candidates (excluding cross-split test duplicates and internal exact duplicates)
    clean_potato_candidates = []
    clean_tomato_candidates = []

    seen_hashes = set()
    excluded_for_test_leakage = 0
    excluded_internal_duplicate = 0

    for img in train_images:
        h = img["md5"]
        cls = img["plantdoc_class"]

        # Leakage protection: never include cross-split test duplicates
        if h in cross_dup_hashes:
            excluded_for_test_leakage += 1
            continue

        # Deduplication protection: keep only one instance of identical images
        if h in seen_hashes:
            excluded_internal_duplicate += 1
            continue
        seen_hashes.add(h)

        if cls in potato_classes:
            clean_potato_candidates.append(img)
        elif cls in tomato_classes:
            # Exclude classes with severe data insufficiency (<5 images)
            if cls == "Tomato two spotted spider mites leaf":
                continue
            clean_tomato_candidates.append(img)

    print(f"\n[Audit 3] Usable clean candidates after deduplication & leakage filtering:")
    print(f"  Potato clean candidates: {len(clean_potato_candidates)} (excluded {excluded_for_test_leakage} test dups)")
    print(f"  Tomato clean candidates: {len(clean_tomato_candidates)}")

    # Potato class counts
    potato_counts = Counter(img["plantdoc_class"] for img in clean_potato_candidates)
    print("\nPotato clean candidate distribution:")
    for k, v in potato_counts.items():
        print(f"  {k:35s} -> {potato_classes[k]}: {v}")

    # Tomato class counts
    tomato_counts = Counter(img["plantdoc_class"] for img in clean_tomato_candidates)
    print("\nTomato clean candidate distribution:")
    for k, v in tomato_counts.items():
        print(f"  {k:35s} -> {tomato_classes[k]}: {v}")

    # 4. Stratified Train / Validation Split (80% Train, 20% Val)
    def split_stratified(candidates: List[Dict[str, Any]], val_ratio: float = 0.20) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        by_class = defaultdict(list)
        for c in candidates:
            by_class[c["plantdoc_class"]].append(c)

        train_split = []
        val_split = []
        for cls, items in by_class.items():
            random.shuffle(items)
            val_size = max(1, int(round(len(items) * val_ratio)))
            val_split.extend(items[:val_size])
            train_split.extend(items[val_size:])

        return train_split, val_split

    potato_train, potato_val = split_stratified(clean_potato_candidates, val_ratio=0.20)
    tomato_train, tomato_val = split_stratified(clean_tomato_candidates, val_ratio=0.20)

    print(f"\n[Manifest Creation]")
    print(f"  Potato: Train={len(potato_train)}, Val={len(potato_val)} (Total={len(potato_train)+len(potato_val)})")
    print(f"  Tomato: Train={len(tomato_train)}, Val={len(tomato_val)} (Total={len(tomato_train)+len(tomato_val)})")

    # Save manifests
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFESTS_DIR / "field_adaptation_potato_train.json", "w") as f:
        json.dump({"images": potato_train, "class_counts": dict(Counter(i["plantdoc_class"] for i in potato_train))}, f, indent=2)
    with open(MANIFESTS_DIR / "field_adaptation_potato_val.json", "w") as f:
        json.dump({"images": potato_val, "class_counts": dict(Counter(i["plantdoc_class"] for i in potato_val))}, f, indent=2)

    with open(MANIFESTS_DIR / "field_adaptation_tomato_train.json", "w") as f:
        json.dump({"images": tomato_train, "class_counts": dict(Counter(i["plantdoc_class"] for i in tomato_train))}, f, indent=2)
    with open(MANIFESTS_DIR / "field_adaptation_tomato_val.json", "w") as f:
        json.dump({"images": tomato_val, "class_counts": dict(Counter(i["plantdoc_class"] for i in tomato_val))}, f, indent=2)

    # Save comprehensive audit report
    audit_report = {
        "timestamp": json.dumps(None),
        "total_train_images": len(train_images),
        "total_test_images": len(test_images),
        "cross_split_duplicates_with_test": {
            "count": len(cross_split_duplicates),
            "records": cross_split_duplicates,
            "leakage_remedy": "Strictly excluded from proposed field adaptation training and validation partitions."
        },
        "internal_train_duplicates": {
            "count": len(internal_duplicates),
            "records": internal_duplicates,
            "remedy": "Deduplicated; contradictory pairs logged."
        },
        "label_mapping_status": {
            "potato": {
                "Potato leaf early blight": {"maps_to": "Potato___Early_blight", "status": "CONFIRMED_NAME_MATCH", "note": "Contains label noise: images of Late blight and Tomato mixed in."},
                "Potato leaf late blight": {"maps_to": "Potato___Late_blight", "status": "CONFIRMED_NAME_MATCH", "note": "Contains duplicates with Tomato Late blight."},
                "Potato___healthy": {"status": "COMPLETELY_MISSING", "count": 0, "note": "Zero healthy potato leaf images exist in PlantDoc."}
            },
            "tomato": {
                "Tomato Early blight leaf": {"maps_to": "Tomato___Early_blight", "status": "CONFIRMED_MATCH"},
                "Tomato leaf bacterial spot": {"maps_to": "Tomato___Bacterial_spot", "status": "CONFIRMED_MATCH"},
                "Tomato leaf late blight": {"maps_to": "Tomato___Late_blight", "status": "CONFIRMED_MATCH"},
                "Tomato mold leaf": {"maps_to": "Tomato___Leaf_Mold", "status": "CONFIRMED_MATCH"},
                "Tomato Septoria leaf spot": {"maps_to": "Tomato___Septoria_leaf_spot", "status": "CONFIRMED_MATCH"},
                "Tomato leaf yellow virus": {"maps_to": "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "status": "CONFIRMED_MATCH"},
                "Tomato leaf mosaic virus": {"maps_to": "Tomato___Tomato_mosaic_virus", "status": "CONFIRMED_MATCH"},
                "Tomato leaf": {"maps_to": "Tomato___healthy", "status": "CONFIRMED_MATCH"},
                "Tomato two spotted spider mites leaf": {"status": "SEVERE_INSUFFICIENCY", "count": 2, "note": "Excluded from adaptation (N=2 insufficient for generalization)."},
                "Tomato___Target_Spot": {"status": "COMPLETELY_MISSING", "count": 0, "note": "Zero target spot images exist in PlantDoc."}
            }
        },
        "curated_manifest_counts": {
            "potato_train": len(potato_train),
            "potato_val": len(potato_val),
            "tomato_train": len(tomato_train),
            "tomato_val": len(tomato_val)
        }
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "label_quality_audit.json", "w") as f:
        json.dump(audit_report, f, indent=2)

    print(f"\nSaved audit report to: {REPORTS_DIR / 'label_quality_audit.json'}")


if __name__ == "__main__":
    from typing import Tuple
    main()
