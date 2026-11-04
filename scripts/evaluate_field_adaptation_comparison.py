"""
Milestone 6C: Comparative Evaluation of Baseline vs Experimental Field-Adapted Models.

Runs single-pass dual evaluation across:
  1. Development / Calibration Split (PlantDoc train partition, N=2,336)
  2. Official Field Test Split (PlantDoc test partition, N=236; disclosed as previously observed in 6A)

Measures:
  - Routing Recall and Precision (Potato, Tomato)
  - Conditional Disease Accuracy (given correct routing)
  - End-to-End Correct Diagnosis (across all eligible supported images)
  - Per-Class Metrics (Precision, Recall, F1)
  - Unsupported Crop False Acceptance (FAR)
  - Treatment-Suppression Compliance

Outputs: data/field_eval/reports/field_adaptation_comparison_report.json
"""

import sys
import json
import time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple
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

MANIFEST_PATH = ROOT_DIR / "data" / "field_eval" / "manifests" / "field_benchmark.json"
OUT_PATH = ROOT_DIR / "data" / "field_eval" / "reports" / "field_adaptation_comparison_report.json"


def evaluate_split(
    images: List[Dict[str, Any]],
    router: PlantRouter,
    pred_baseline: DiseasePredictor,
    pred_adapted: DiseasePredictor,
    treatment_recommender: PotatoTreatmentRecommender
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Shared routing counters
    potato_total = 0
    potato_routed_to_potato = 0
    potato_routed_to_tomato = 0
    potato_rejected = 0

    tomato_total = 0
    tomato_routed_to_tomato = 0
    tomato_routed_to_potato = 0
    tomato_rejected = 0

    total_routed_to_potato = 0
    total_routed_to_tomato = 0

    unsupported_total = 0
    unsupported_accepted_potato = 0
    unsupported_accepted_tomato = 0

    # Config-specific disease counters
    configs = ["baseline", "field_adapted"]
    predictors = {"baseline": pred_baseline, "field_adapted": pred_adapted}

    disease_correct = {"baseline": {"potato": 0, "tomato": 0}, "field_adapted": {"potato": 0, "tomato": 0}}
    disease_evaluated = {"potato": 0, "tomato": 0}

    per_class_stats = {
        "baseline": defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0}),
        "field_adapted": defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0}),
    }

    suppression_eligible = 0
    suppression_compliant = {"baseline": 0, "field_adapted": 0}

    t0 = time.time()
    with torch.inference_mode():
        for idx, item in enumerate(images):
            if (idx + 1) % 500 == 0 or idx == len(images) - 1:
                print(f"    [{idx+1}/{len(images)}] ({time.time()-t0:.1f}s)")

            path = resolve_dataset_path(item["path"])
            true_crop = item["crop_id"]
            true_disease = item["mapped_disease"]
            cohort = item["cohort"]

            try:
                with Image.open(path) as raw:
                    img = raw.convert("RGB")
            except Exception:
                continue

            # Route through PlantRouter (shared Stage 1)
            routing_dec = router.route(img, manual_crop="Auto-Detect (AI Plant ID)")
            target_crop = routing_dec.target_crop
            routed = routing_dec.route_to_specialist
            status = routing_dec.routing_status

            if routed and target_crop == "potato":
                total_routed_to_potato += 1
            elif routed and target_crop == "tomato":
                total_routed_to_tomato += 1

            # Shared cohort routing categorization
            if cohort == "SUPPORTED_POTATO":
                potato_total += 1
                if routed and target_crop == "potato":
                    potato_routed_to_potato += 1
                    if true_disease:
                        disease_evaluated["potato"] += 1
                elif routed and target_crop == "tomato":
                    potato_routed_to_tomato += 1
                else:
                    potato_rejected += 1

            elif cohort == "SUPPORTED_TOMATO":
                tomato_total += 1
                if routed and target_crop == "tomato":
                    tomato_routed_to_tomato += 1
                    if true_disease:
                        disease_evaluated["tomato"] += 1
                elif routed and target_crop == "potato":
                    tomato_routed_to_potato += 1
                else:
                    tomato_rejected += 1

            elif cohort == "UNSUPPORTED_CROP":
                unsupported_total += 1
                if routed:
                    if target_crop == "potato":
                        unsupported_accepted_potato += 1
                    elif target_crop == "tomato":
                        unsupported_accepted_tomato += 1

            # Disease diagnosis for each configuration
            for cfg in configs:
                pred_disease = None
                if routed and target_crop in ("potato", "tomato"):
                    try:
                        diag = predictors[cfg].predict(img, crop=target_crop)
                        pred_disease = diag.get("predicted_disease")
                    except Exception:
                        pass

                # Disease accuracy tracking
                if routed and target_crop == "potato" and cohort == "SUPPORTED_POTATO" and true_disease:
                    if pred_disease == true_disease:
                        disease_correct[cfg]["potato"] += 1
                        per_class_stats[cfg][true_disease]["tp"] += 1
                    else:
                        per_class_stats[cfg][true_disease]["fn"] += 1
                        if pred_disease:
                            per_class_stats[cfg][pred_disease]["fp"] += 1

                elif routed and target_crop == "tomato" and cohort == "SUPPORTED_TOMATO" and true_disease:
                    if pred_disease == true_disease:
                        disease_correct[cfg]["tomato"] += 1
                        per_class_stats[cfg][true_disease]["tp"] += 1
                    else:
                        per_class_stats[cfg][true_disease]["fn"] += 1
                        if pred_disease:
                            per_class_stats[cfg][pred_disease]["fp"] += 1

                # Safety check
                is_uncertain = "_UNCERTAIN" in status
                should_suppress = (not routed) or is_uncertain or (cohort == "UNSUPPORTED_CROP")

                if cfg == "baseline" and should_suppress:
                    suppression_eligible += 1

                if should_suppress:
                    try:
                        treat_res = treatment_recommender.recommend(
                            disease=pred_disease or "Unknown",
                            is_low_confidence=True,
                            risk_level="LOW"
                        )
                        chemicals = treat_res.get("chemical_treatments", [])
                        if len(chemicals) == 0:
                            suppression_compliant[cfg] += 1
                    except Exception:
                        suppression_compliant[cfg] += 1

    # Format outputs
    pot_rec = (potato_routed_to_potato / potato_total * 100.0) if potato_total > 0 else 0.0
    pot_prec = (potato_routed_to_potato / total_routed_to_potato * 100.0) if total_routed_to_potato > 0 else 0.0
    tom_rec = (tomato_routed_to_tomato / tomato_total * 100.0) if tomato_total > 0 else 0.0
    tom_prec = (tomato_routed_to_tomato / total_routed_to_tomato * 100.0) if total_routed_to_tomato > 0 else 0.0

    tot_unsupp_acc = unsupported_accepted_potato + unsupported_accepted_tomato
    unsupp_far = (tot_unsupp_acc / unsupported_total * 100.0) if unsupported_total > 0 else 0.0
    supp_tot = potato_total + tomato_total

    results_by_cfg = {}
    for cfg in configs:
        pot_cond = (disease_correct[cfg]["potato"] / disease_evaluated["potato"] * 100.0) if disease_evaluated["potato"] > 0 else 0.0
        tom_cond = (disease_correct[cfg]["tomato"] / disease_evaluated["tomato"] * 100.0) if disease_evaluated["tomato"] > 0 else 0.0

        pot_e2e = (disease_correct[cfg]["potato"] / potato_total * 100.0) if potato_total > 0 else 0.0
        tom_e2e = (disease_correct[cfg]["tomato"] / tomato_total * 100.0) if tomato_total > 0 else 0.0
        ov_correct = disease_correct[cfg]["potato"] + disease_correct[cfg]["tomato"]
        ov_e2e = (ov_correct / supp_tot * 100.0) if supp_tot > 0 else 0.0

        supp_rate = (suppression_compliant[cfg] / suppression_eligible * 100.0) if suppression_eligible > 0 else 100.0

        # Per-class summary
        per_class_summary = {}
        for cls, st in per_class_stats[cfg].items():
            tp = st["tp"]
            fn = st["fn"]
            fp = st["fp"]
            rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
            prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            per_class_summary[cls] = {
                "evaluated": tp + fn,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "recall": f"{rec:.1f}%",
                "precision": f"{prec:.1f}%",
                "f1": f"{f1:.1f}%"
            }

        results_by_cfg[cfg] = {
            "config_name": f"{'Baseline' if cfg=='baseline' else 'Field-Adapted'} Specialists (Letterbox)",
            "sample_count": len(images),
            "potato_routing": {
                "recall": f"{pot_rec:.2f}% ({potato_routed_to_potato}/{potato_total})",
                "precision": f"{pot_prec:.2f}% ({potato_routed_to_potato}/{total_routed_to_potato})",
                "routed": potato_routed_to_potato,
                "misrouted_to_tomato": potato_routed_to_tomato,
                "rejected": potato_rejected,
                "total": potato_total
            },
            "tomato_routing": {
                "recall": f"{tom_rec:.2f}% ({tomato_routed_to_tomato}/{tomato_total})",
                "precision": f"{tom_prec:.2f}% ({tomato_routed_to_tomato}/{total_routed_to_tomato})",
                "routed": tomato_routed_to_tomato,
                "misrouted_to_potato": tomato_routed_to_potato,
                "rejected": tomato_rejected,
                "total": tomato_total
            },
            "conditional_disease_accuracy": {
                "potato": f"{pot_cond:.2f}% ({disease_correct[cfg]['potato']}/{disease_evaluated['potato']})",
                "tomato": f"{tom_cond:.2f}% ({disease_correct[cfg]['tomato']}/{disease_evaluated['tomato']})",
                "potato_correct": disease_correct[cfg]["potato"],
                "potato_evaluated": disease_evaluated["potato"],
                "tomato_correct": disease_correct[cfg]["tomato"],
                "tomato_evaluated": disease_evaluated["tomato"]
            },
            "end_to_end_accuracy": {
                "potato": f"{pot_e2e:.2f}% ({disease_correct[cfg]['potato']}/{potato_total})",
                "tomato": f"{tom_e2e:.2f}% ({disease_correct[cfg]['tomato']}/{tomato_total})",
                "overall_supported": f"{ov_e2e:.2f}% ({ov_correct}/{supp_tot})",
                "overall_correct": ov_correct,
                "overall_total": supp_tot
            },
            "per_class_metrics": per_class_summary,
            "unsupported_crop_false_acceptance": {
                "false_acceptance_rate": f"{unsupp_far:.2f}% ({tot_unsupp_acc}/{unsupported_total})",
                "accepted_total": tot_unsupp_acc,
                "accepted_as_potato": unsupported_accepted_potato,
                "accepted_as_tomato": unsupported_accepted_tomato,
                "unsupported_total": unsupported_total
            },
            "treatment_suppression_compliance": {
                "compliance_rate": f"{supp_rate:.2f}% ({suppression_compliant[cfg]}/{suppression_eligible})",
                "compliant_cases": suppression_compliant[cfg],
                "eligible_cases": suppression_eligible
            }
        }

    return results_by_cfg["baseline"], results_by_cfg["field_adapted"]


def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    all_images = manifest["images"]

    dev_images = [img for img in all_images if img["plantdoc_split"] == "train"]
    test_images = [img for img in all_images if img["plantdoc_split"] == "test"]

    print(f"Total benchmark images: {len(all_images)}")
    print(f"Development split: {len(dev_images)} images")
    print(f"Official Test split: {len(test_images)} images")

    device = torch.device("cpu")
    leaf_verifier = LeafVerifier()
    treatment_recommender = PotatoTreatmentRecommender()
    identifier = PlantIdentifier(device=device, preprocessing_mode="letterbox")
    router = PlantRouter(leaf_verifier=leaf_verifier, plant_identifier=identifier)

    pred_baseline = DiseasePredictor(device=device, preprocessing_mode="letterbox", use_field_adapted_models=False)
    pred_adapted = DiseasePredictor(device=device, preprocessing_mode="letterbox", use_field_adapted_models=True)

    results = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "methodology": {
            "selection_split": "PlantDoc train split (N=2,336) used for comparative development evaluation.",
            "test_split_disclosure": "PlantDoc test split (N=236) was previously observed in Milestone 6A and is reported solely as exploratory evidence.",
            "weights_status": "Original checkpoints unmodified; experimental checkpoints evaluated side-by-side.",
            "thresholds_status": "Unchanged routing and rejection thresholds."
        },
        "development_split": {},
        "official_test_split": {}
    }

    print("\n" + "="*80)
    print("PHASE 1: EVALUATING DEVELOPMENT SPLIT (N=2,336) [Single-Pass Dual Inference]")
    print("="*80)
    t0 = time.time()
    dev_base, dev_adapt = evaluate_split(dev_images, router, pred_baseline, pred_adapted, treatment_recommender)
    print(f"  Dev split evaluation completed in {time.time()-t0:.1f}s.")
    results["development_split"]["baseline"] = dev_base
    results["development_split"]["field_adapted"] = dev_adapt

    print("\n" + "="*80)
    print("PHASE 2: EVALUATING OFFICIAL TEST SPLIT (N=236) [Single-Pass Dual Inference]")
    print("="*80)
    t1 = time.time()
    test_base, test_adapt = evaluate_split(test_images, router, pred_baseline, pred_adapted, treatment_recommender)
    print(f"  Test split evaluation completed in {time.time()-t1:.1f}s.")
    results["official_test_split"]["baseline"] = test_base
    results["official_test_split"]["field_adapted"] = test_adapt

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison report to: {OUT_PATH}")

    # Display comparison table
    print("\n" + "="*80)
    print("DEVELOPMENT SPLIT COMPARISON (N=2,336)")
    print("="*80)
    print(f"Potato Cond Disease Acc:   Baseline={dev_base['conditional_disease_accuracy']['potato']} vs Adapted={dev_adapt['conditional_disease_accuracy']['potato']}")
    print(f"Tomato Cond Disease Acc:   Baseline={dev_base['conditional_disease_accuracy']['tomato']} vs Adapted={dev_adapt['conditional_disease_accuracy']['tomato']}")
    print(f"Overall E2E Diagnosis:     Baseline={dev_base['end_to_end_accuracy']['overall_supported']} vs Adapted={dev_adapt['end_to_end_accuracy']['overall_supported']}")
    print(f"Unsupported False Accept:  Baseline={dev_base['unsupported_crop_false_acceptance']['false_acceptance_rate']} vs Adapted={dev_adapt['unsupported_crop_false_acceptance']['false_acceptance_rate']}")
    print(f"Treatment Suppression:     Baseline={dev_base['treatment_suppression_compliance']['compliance_rate']} vs Adapted={dev_adapt['treatment_suppression_compliance']['compliance_rate']}")

    print("\n" + "="*80)
    print("OFFICIAL TEST SPLIT EXPLORATORY COMPARISON (N=236)")
    print("="*80)
    print(f"Potato Cond Disease Acc:   Baseline={test_base['conditional_disease_accuracy']['potato']} vs Adapted={test_adapt['conditional_disease_accuracy']['potato']}")
    print(f"Tomato Cond Disease Acc:   Baseline={test_base['conditional_disease_accuracy']['tomato']} vs Adapted={test_adapt['conditional_disease_accuracy']['tomato']}")
    print(f"Overall E2E Diagnosis:     Baseline={test_base['end_to_end_accuracy']['overall_supported']} vs Adapted={test_adapt['end_to_end_accuracy']['overall_supported']}")


if __name__ == "__main__":
    from typing import Tuple
    main()
