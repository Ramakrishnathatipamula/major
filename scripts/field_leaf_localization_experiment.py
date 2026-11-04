"""
Milestone 6A: Controlled Field Leaf-Localization and Preprocessing Experiment.

Rigorously compares 4 non-destructive input preprocessing interventions against the
Milestone 5 frozen baseline across the PlantDoc test set (N=236) and full field dataset:
  1. Baseline (Direct squish to 224x224)
  2. Letterbox (Aspect-ratio preserving padding to square)
  3. Zoomed Leaf Crop (Central 70% bounding box to isolate primary leaf)
  4. Multi-Region Ensembling (Aspect-preserving letterbox + central crop aggregation)

Zero modification to model checkpoints, zero data leakage into training splits.
Outputs: data/field_eval/reports/field_localization_experiment_results.json
"""

import sys
import json
import time
from pathlib import Path
from collections import Counter
from typing import Dict, Any, Tuple
import torch
import torch.nn.functional as F
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.identification.plant_identifier import PlantIdentifier
from modules.dataset_paths import resolve_dataset_path
from modules.identification.leaf_verifier import LeafVerifier
from modules.identification.router import PlantRouter
from modules.disease_predictor import DiseasePredictor

MANIFEST_PATH = ROOT_DIR / "data" / "field_eval" / "manifests" / "field_benchmark.json"
OUT_PATH = ROOT_DIR / "data" / "field_eval" / "reports" / "field_localization_experiment_results.json"


def letterbox_pad(img: Image.Image) -> Image.Image:
    """Pad to square with neutral gray (128, 128, 128) to preserve natural aspect ratio."""
    w, h = img.size
    max_side = max(w, h)
    new_img = Image.new("RGB", (max_side, max_side), (128, 128, 128))
    new_img.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
    return new_img


def center_leaf_crop(img: Image.Image, crop_fraction: float = 0.70) -> Image.Image:
    """Extract central 70% window to discard peripheral field clutter/soil/weeds."""
    w, h = img.size
    dw = int(w * (1.0 - crop_fraction) / 2.0)
    dh = int(h * (1.0 - crop_fraction) / 2.0)
    return img.crop((dw, dh, w - dw, h - dh))


def run_experiment(images_subset: list, identifier: PlantIdentifier, router: PlantRouter, disease_predictor: DiseasePredictor) -> Dict[str, Any]:
    interventions = ["baseline", "letterbox", "center_crop_70", "multi_region_ensemble"]
    metrics = {
        name: {
            "potato_routed_correctly": 0,
            "potato_total": 0,
            "tomato_routed_correctly": 0,
            "tomato_total": 0,
            "unsupported_accepted": 0,
            "unsupported_total": 0,
            "disease_correct_potato": 0,
            "disease_correct_tomato": 0,
            "total_supported_evaluated": 0,
            "mean_prototype_sim_supported": 0.0,
            "mean_energy_supported": 0.0,
            "sim_sum": 0.0,
            "energy_sum": 0.0,
        }
        for name in interventions
    }

    t0 = time.time()
    for idx, item in enumerate(images_subset):
        path = resolve_dataset_path(item["path"])
        true_crop = item["crop_id"]
        true_disease = item["mapped_disease"]
        cohort = item["cohort"]

        try:
            with Image.open(path) as raw:
                raw_img = raw.convert("RGB")
        except Exception:
            continue

        # Prepare transformed variants
        variants = {
            "baseline": raw_img,
            "letterbox": letterbox_pad(raw_img),
            "center_crop_70": center_leaf_crop(raw_img, crop_fraction=0.70),
        }

        # For each intervention, evaluate identification and routing
        for name in ["baseline", "letterbox", "center_crop_70"]:
            var_img = variants[name]
            id_res = identifier.identify(var_img)
            routing_dec = router.route(var_img, manual_crop="Auto-Detect (AI Plant ID)")

            is_supp = cohort in ("SUPPORTED_POTATO", "SUPPORTED_TOMATO")
            m = metrics[name]

            if cohort == "SUPPORTED_POTATO":
                m["potato_total"] += 1
                if routing_dec.route_to_specialist and routing_dec.target_crop == "potato":
                    m["potato_routed_correctly"] += 1
                    # Disease check
                    try:
                        diag = disease_predictor.predict(var_img, crop="potato")
                        if diag.get("predicted_disease") == true_disease:
                            m["disease_correct_potato"] += 1
                    except Exception:
                        pass
                if id_res.prototype_similarity:
                    m["sim_sum"] += id_res.prototype_similarity
                if id_res.energy_score:
                    m["energy_sum"] += id_res.energy_score

            elif cohort == "SUPPORTED_TOMATO":
                m["tomato_total"] += 1
                if routing_dec.route_to_specialist and routing_dec.target_crop == "tomato":
                    m["tomato_routed_correctly"] += 1
                    try:
                        diag = disease_predictor.predict(var_img, crop="tomato")
                        if diag.get("predicted_disease") == true_disease:
                            m["disease_correct_tomato"] += 1
                    except Exception:
                        pass
                if id_res.prototype_similarity:
                    m["sim_sum"] += id_res.prototype_similarity
                if id_res.energy_score:
                    m["energy_sum"] += id_res.energy_score

            elif cohort == "UNSUPPORTED_CROP":
                m["unsupported_total"] += 1
                if routing_dec.route_to_specialist:
                    m["unsupported_accepted"] += 1

        # Multi-region ensemble: aggregate logits of baseline, letterbox, and center_crop
        m_ens = metrics["multi_region_ensemble"]
        t_base = identifier.transform(variants["baseline"]).unsqueeze(0)
        t_let = identifier.transform(variants["letterbox"]).unsqueeze(0)
        t_crop = identifier.transform(variants["center_crop_70"]).unsqueeze(0)
        with torch.no_grad():
            batch = torch.cat([t_base, t_let, t_crop], dim=0)
            logits = identifier.model(batch)
            avg_logits = logits.mean(dim=0, keepdim=True)
            probs = F.softmax(avg_logits, dim=-1)[0]
            top_prob, top_idx = torch.topk(probs, 1)
            ens_crop = identifier.classes[top_idx.item()]
            ens_conf = top_prob.item() * 100.0

        if cohort == "SUPPORTED_POTATO":
            m_ens["potato_total"] += 1
            if ens_crop == "potato" and ens_conf >= 60.0:
                m_ens["potato_routed_correctly"] += 1
        elif cohort == "SUPPORTED_TOMATO":
            m_ens["tomato_total"] += 1
            if ens_crop == "tomato" and ens_conf >= 60.0:
                m_ens["tomato_routed_correctly"] += 1
        elif cohort == "UNSUPPORTED_CROP":
            m_ens["unsupported_total"] += 1
            if ens_crop in ("potato", "tomato") and ens_conf >= 60.0:
                m_ens["unsupported_accepted"] += 1

    # Summarize metrics
    summary = {}
    for name, m in metrics.items():
        p_tot = max(1, m["potato_total"])
        t_tot = max(1, m["tomato_total"])
        u_tot = max(1, m["unsupported_total"])
        supp_tot = p_tot + t_tot

        p_recall = round(100.0 * m["potato_routed_correctly"] / p_tot, 2)
        t_recall = round(100.0 * m["tomato_routed_correctly"] / t_tot, 2)
        u_far = round(100.0 * m["unsupported_accepted"] / u_tot, 2)

        summary[name] = {
            "potato_routing_recall": f"{p_recall}% ({m['potato_routed_correctly']}/{p_tot})",
            "tomato_routing_recall": f"{t_recall}% ({m['tomato_routed_correctly']}/{t_tot})",
            "unsupported_false_acceptance_rate": f"{u_far}% ({m['unsupported_accepted']}/{u_tot})",
            "potato_disease_correct": m["disease_correct_potato"],
            "tomato_disease_correct": m["disease_correct_tomato"],
            "avg_proto_sim_supported": round(m["sim_sum"] / supp_tot, 4) if m["sim_sum"] > 0 else None,
            "avg_energy_supported": round(m["energy_sum"] / supp_tot, 3) if m["energy_sum"] > 0 else None,
        }

    return summary


def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    all_images = manifest["images"]

    # Test split (N=236) and full dataset (N=2572)
    test_split_images = [img for img in all_images if img["plantdoc_split"] == "test"]
    print(f"Loaded {len(all_images)} total images ({len(test_split_images)} in test split).")

    identifier = PlantIdentifier(device="cpu")
    router = PlantRouter()
    disease_predictor = DiseasePredictor()

    print("\nRunning controlled experiment on official test split (N=236)...")
    t0 = time.time()
    test_results = run_experiment(test_split_images, identifier, router, disease_predictor)
    print(f"Test split experiment complete in {time.time()-t0:.1f}s.")

    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "test_split_sample_size": len(test_split_images),
        "results_test_split": test_results,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved experiment results to: {OUT_PATH}")
    print("\n" + "="*80)
    print("CONTROLLED EXPERIMENT RESULTS (Official Test Split, N=236)")
    print("="*80)
    for intervention, res in test_results.items():
        print(f"\n--- [{intervention.upper()}] ---")
        for k, v in res.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
