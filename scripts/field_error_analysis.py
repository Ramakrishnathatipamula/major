"""
Milestone 6A: Detailed per-image field error analysis.
Extracts categorized failure records from PlantDoc field benchmark WITHOUT modifying
any application code, thresholds, prototypes, or model weights.

Output: data/field_eval/reports/per_image_error_analysis.json
"""

import sys
import json
import time
from pathlib import Path
from collections import Counter, defaultdict
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.identification.leaf_verifier import LeafVerifier
from modules.dataset_paths import resolve_dataset_path
from modules.identification.plant_identifier import PlantIdentifier
from modules.identification.router import PlantRouter
from modules.disease_predictor import DiseasePredictor
from modules.treatment import PotatoTreatmentRecommender

MANIFEST = ROOT_DIR / "data" / "field_eval" / "manifests" / "field_benchmark.json"
OUT_DIR  = ROOT_DIR / "data" / "field_eval" / "reports"

def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)
    images = manifest["images"]
    print(f"Loaded {len(images)} images from manifest.")

    # Init engines (frozen)
    router = PlantRouter()
    disease_predictor = DiseasePredictor()
    treatment = PotatoTreatmentRecommender()

    # Failure category buckets
    cat_A_supported_rejected_by_ood = []   # Genuine supported leaves rejected by open-set gates
    cat_B_supported_wrong_specialist = []  # Supported leaves routed to wrong specialist
    cat_C_routed_wrong_disease = []        # Correctly routed but wrong disease prediction
    cat_D_unsupported_accepted = []        # Unsupported crops incorrectly accepted
    cat_E_leaf_verifier_rejects = []       # Leaf verifier false rejections (on genuine leaves)
    cat_F_correctly_routed_correct = []    # Correctly routed AND correct disease
    all_records = []

    t0 = time.time()
    with torch.inference_mode():
        for idx, item in enumerate(images):
            if (idx + 1) % 500 == 0 or idx == len(images) - 1:
                print(f"  [{idx+1}/{len(images)}] ({time.time()-t0:.1f}s)")

            path = resolve_dataset_path(item["path"])
            true_crop = item["crop_id"]
            true_disease = item["mapped_disease"]
            cohort = item["cohort"]
            split = item["plantdoc_split"]

            try:
                with Image.open(path) as raw:
                    img = raw.convert("RGB")
            except Exception:
                continue

            dec = router.route(img, manual_crop="Auto-Detect (AI Plant ID)")
            leaf = dec.leaf_verification or {}
            pid = dec.plant_identification or {}

            # Disease diagnosis if routed
            disease_pred = None
            disease_conf = None
            disease_correct = None
            if dec.route_to_specialist and true_disease:
                try:
                    diag = disease_predictor.predict(img, crop=dec.target_crop)
                    disease_pred = diag.get("predicted_disease")
                    disease_conf = diag.get("confidence")
                    disease_correct = (disease_pred == true_disease)
                except Exception:
                    pass

            rec = {
                "path": item["relative_path"],
                "plantdoc_class": item["plantdoc_class"],
                "true_crop": true_crop,
                "true_disease": true_disease,
                "cohort": cohort,
                "split": split,
                "leaf_is_leaf": leaf.get("is_leaf", False),
                "leaf_status": leaf.get("status", "UNKNOWN"),
                "leaf_conf": round(leaf.get("confidence", 0.0), 2),
                "plant_id_pred": pid.get("crop_id", "unknown"),
                "plant_id_conf": round(pid.get("confidence", 0.0), 2),
                "plant_id_proto_sim": pid.get("prototype_similarity"),
                "plant_id_proto_crop": pid.get("prototype_crop"),
                "plant_id_energy": pid.get("energy_score"),
                "plant_id_ood_status": pid.get("ood_status", "UNKNOWN"),
                "plant_id_status": pid.get("status", "UNKNOWN"),
                "routing_target": dec.target_crop,
                "routing_status": dec.routing_status,
                "route_to_specialist": dec.route_to_specialist,
                "disease_pred": disease_pred,
                "disease_conf": disease_conf,
                "disease_correct": disease_correct,
            }

            all_records.append(rec)

            # Classify failure category
            is_supported = cohort in ("SUPPORTED_POTATO", "SUPPORTED_TOMATO")
            expected_crop = "potato" if cohort == "SUPPORTED_POTATO" else ("tomato" if cohort == "SUPPORTED_TOMATO" else None)

            # Cat E: Leaf verifier false rejection (of genuine plant leaves)
            if not rec["leaf_is_leaf"]:
                cat_E_leaf_verifier_rejects.append(rec)
                continue

            if is_supported:
                if not dec.route_to_specialist:
                    # Cat A: Supported leaf rejected by OOD/unknown gates
                    cat_A_supported_rejected_by_ood.append(rec)
                elif dec.target_crop != expected_crop:
                    # Cat B: Routed to wrong specialist
                    cat_B_supported_wrong_specialist.append(rec)
                elif disease_correct is False:
                    # Cat C: Right specialist, wrong disease
                    cat_C_routed_wrong_disease.append(rec)
                elif disease_correct is True:
                    # Correct
                    cat_F_correctly_routed_correct.append(rec)
            else:
                # Unsupported crops
                if dec.route_to_specialist:
                    cat_D_unsupported_accepted.append(rec)

    elapsed = time.time() - t0
    print(f"\nInference complete: {len(all_records)} images in {elapsed:.1f}s")

    # Detailed statistics per failure category
    def summarize_cat(records, cat_name):
        if not records:
            return {"count": 0, "category": cat_name}

        statuses = Counter(r["routing_status"] for r in records)
        ood_statuses = Counter(r["plant_id_ood_status"] for r in records)
        pred_crops = Counter(r["plant_id_pred"] for r in records)
        proto_sims = [r["plant_id_proto_sim"] for r in records if r["plant_id_proto_sim"] is not None]
        energies = [r["plant_id_energy"] for r in records if r["plant_id_energy"] is not None]
        confs = [r["plant_id_conf"] for r in records if r["plant_id_conf"] is not None]

        s = {
            "count": len(records),
            "category": cat_name,
            "routing_status_dist": dict(statuses),
            "ood_status_dist": dict(ood_statuses),
            "predicted_crop_dist": dict(pred_crops),
            "proto_sim_stats": {
                "mean": round(sum(proto_sims)/len(proto_sims), 4) if proto_sims else None,
                "min": round(min(proto_sims), 4) if proto_sims else None,
                "max": round(max(proto_sims), 4) if proto_sims else None,
                "below_0.70": sum(1 for x in proto_sims if x < 0.70),
                "0.70_to_0.80": sum(1 for x in proto_sims if 0.70 <= x < 0.80),
                "0.80_to_0.90": sum(1 for x in proto_sims if 0.80 <= x < 0.90),
                "above_0.90": sum(1 for x in proto_sims if x >= 0.90),
            },
            "energy_stats": {
                "mean": round(sum(energies)/len(energies), 3) if energies else None,
                "min": round(min(energies), 3) if energies else None,
                "max": round(max(energies), 3) if energies else None,
                "below_3.5": sum(1 for x in energies if x < 3.5),
                "3.5_to_4.5": sum(1 for x in energies if 3.5 <= x < 4.5),
                "above_4.5": sum(1 for x in energies if x >= 4.5),
            },
            "softmax_conf_stats": {
                "mean": round(sum(confs)/len(confs), 2) if confs else None,
                "min": round(min(confs), 2) if confs else None,
                "max": round(max(confs), 2) if confs else None,
                "below_40": sum(1 for x in confs if x < 40),
                "40_to_60": sum(1 for x in confs if 40 <= x < 60),
                "above_60": sum(1 for x in confs if x >= 60),
            },
            "representative_failures": records[:10],
        }

        # For Cat C: also show true vs predicted disease distribution
        if cat_name.startswith("C"):
            true_vs_pred = Counter((r["true_disease"], r["disease_pred"]) for r in records)
            s["true_vs_pred_dist"] = {f"{t} -> {p}": c for (t, p), c in true_vs_pred.most_common()}

        return s

    # For Cat A, break down further by which gate triggered the rejection
    cat_A_by_gate = defaultdict(list)
    for r in cat_A_supported_rejected_by_ood:
        gate = "UNKNOWN"
        sim = r["plant_id_proto_sim"]
        energy = r["plant_id_energy"]
        conf = r["plant_id_conf"]
        if sim is not None and sim < 0.70:
            gate = "PROTOTYPE_SIM_BELOW_0.70"
        elif energy is not None and energy < 3.5:
            gate = "ENERGY_BELOW_3.5"
        elif conf < 40.0:
            gate = "SOFTMAX_CONF_BELOW_40"
        else:
            gate = "AMBIGUOUS_BUT_NOT_REJECTED"
        # Not rejected by OOD but still not routed? Check if predicted crop != expected
        if r["routing_status"] == "UNSUPPORTED_CROP":
            gate = "MISIDENTIFIED_AS_UNSUPPORTED_CROP"
        elif r["routing_status"] == "UNKNOWN_SPECIES":
            gate = r["plant_id_ood_status"]
        cat_A_by_gate[gate].append(r)

    # Compile output
    output = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_images_evaluated": len(all_records),
        "summary_counts": {
            "A_supported_rejected_by_gates": len(cat_A_supported_rejected_by_ood),
            "B_supported_wrong_specialist": len(cat_B_supported_wrong_specialist),
            "C_correctly_routed_wrong_disease": len(cat_C_routed_wrong_disease),
            "D_unsupported_accepted": len(cat_D_unsupported_accepted),
            "E_leaf_verifier_false_rejections": len(cat_E_leaf_verifier_rejects),
            "F_correctly_routed_correct_disease": len(cat_F_correctly_routed_correct),
            "note": "A+B+C+F = total supported images minus leaf-verifier rejects. D counted from unsupported only."
        },
        "A_supported_rejected_by_gates": summarize_cat(cat_A_supported_rejected_by_ood, "A: Supported leaves rejected by open-set gates"),
        "A_by_rejection_gate": {gate: len(recs) for gate, recs in cat_A_by_gate.items()},
        "B_supported_wrong_specialist": summarize_cat(cat_B_supported_wrong_specialist, "B: Supported leaves routed to wrong specialist"),
        "C_routed_wrong_disease": summarize_cat(cat_C_routed_wrong_disease, "C: Correctly routed, wrong disease prediction"),
        "D_unsupported_accepted": summarize_cat(cat_D_unsupported_accepted, "D: Unsupported crops falsely accepted"),
        "E_leaf_verifier_false_rejections": summarize_cat(cat_E_leaf_verifier_rejects, "E: Leaf verifier false rejections"),
        "F_correctly_routed_correct": summarize_cat(cat_F_correctly_routed_correct, "F: Correctly routed AND correct disease"),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "per_image_error_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved error analysis to: {out_path}")

    # Also save the raw per-image records for further analysis
    raw_path = OUT_DIR / "per_image_records.json"
    with open(raw_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"Saved raw per-image records to: {raw_path}")

    # Print summary
    print("\n" + "="*80)
    print("FAILURE CATEGORY SUMMARY")
    print("="*80)
    for k, v in output["summary_counts"].items():
        print(f"  {k}: {v}")
    print(f"\n  A breakdown by rejection gate: {output['A_by_rejection_gate']}")


if __name__ == "__main__":
    main()
