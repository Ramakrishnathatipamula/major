import sys
import json
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.intelligence import CropDoctorIntelligenceLayer

def main():
    print("=========================================================================")
    print("AI CROP DOCTOR — INTEGRATED INTELLIGENCE LAYER DEMONSTRATION")
    print("=========================================================================\n")

    late_blight_dir = ROOT_DIR / "data" / "raw" / "plant_village_subset" / "Potato___Late_blight"
    image_candidates = list(late_blight_dir.glob("*.JPG")) + list(late_blight_dir.glob("*.jpg"))
    if not image_candidates:
        print(f"Error: No sample images found in {late_blight_dir}")
        sys.exit(1)

    sample_image_path = image_candidates[0]

    latitude = 44.9778   # Minneapolis / Midwest Potato Region
    longitude = -93.2650

    layer = CropDoctorIntelligenceLayer()

    print(f"Analyzing Sample Leaf Image: {sample_image_path.relative_to(ROOT_DIR)}")
    print(f"Location Coordinates: Lat {latitude}, Lon {longitude}\n")

    # 1. BASELINE DIAGNOSIS (Without What-If scenario)
    print("-------------------------------------------------------------------------")
    print("1. RUNNING PRIMARY INTEGRATED DIAGNOSTIC PIPELINE")
    print("-------------------------------------------------------------------------")
    baseline_result = layer.analyze(
        image_path=str(sample_image_path),
        latitude=latitude,
        longitude=longitude,
        crop="Potato",
        scenario_params=None
    )

    payload_base = baseline_result.to_dict()

    print("\n--- DIAGNOSIS & EXPLAINABILITY ---")
    print(f" • Crop Detected     : {payload_base['DIAGNOSIS']['crop']}")
    print(f" • Disease Diagnosis : {payload_base['DIAGNOSIS']['disease']}")
    print(f" • Confidence        : {payload_base['DIAGNOSIS']['confidence'] * 100:.1f}%")
    print(f" • Low Confidence    : {payload_base['DIAGNOSIS']['is_low_confidence']}")
    print(f" • Grad-CAM Heatmap  : {payload_base['EXPLAINABILITY']['gradcam_output_path']}")

    print("\n--- MICROCLIMATE ENVIRONMENT & RISK ---")
    print(f" • Weather Available : {payload_base['ENVIRONMENT']['weather_available']}")
    print(f" • Risk Level        : {payload_base['RISK']['risk_level']}")
    print("\n--- RECOMMENDED TREATMENT / MEDICINE (CIBRC / ICAR-CPRI / TNAU) ---")
    treatment = payload_base.get("TREATMENT", {})
    print(f" • Treatment Status   : {treatment.get('status')}")
    print(f" • Clinical Summary   : {treatment.get('summary_message')}")
    print(f" • Chemical Options   : {len(treatment.get('chemical_treatments', []))} verified options")
    for opt in treatment.get("chemical_treatments", []):
        print(f"   - {opt.get('active_ingredient')} {opt.get('formulation')} ({opt.get('category')})")
        print(f"     Rate: {opt.get('application_rate')}")
        print(f"     Timing: {opt.get('timing_and_frequency')}")
    print(f" • Cultural Measures  : {len(treatment.get('non_chemical_control_measures', []))} cultural practices")
    print(f" • Safety Precautions : {len(treatment.get('safety_and_environmental_precautions', []))} safety rules")

    print("\n--- GROUNDED RAG KNOWLEDGE EXPLANATION ---")
    print(f" • Generated Query   : \"{payload_base['KNOWLEDGE']['generated_query']}\"")
    print(f" • Grounded Status   : {payload_base['KNOWLEDGE']['grounding_status']}")
    print(f" • Grounded Answer   :\n{payload_base['KNOWLEDGE']['grounded_answer']}\n")

    print("--- VERIFIED EXTENSION SOURCES RETURNED ---")
    for src in payload_base['KNOWLEDGE']['retrieved_sources']:
        print(f" • {src.get('source_organization')} | {src.get('document_title')}")
        print(f"   URL: {src.get('url_reference')}")
        print(f"   Similarity: {src.get('similarity_score')}\n")

    # 2. WHAT-IF SCENARIO DEMONSTRATION
    print("=========================================================================")
    print("2. WHAT-IF SCENARIO SENSITIVITY DEMONSTRATION (HYPOTHETICAL)")
    print("=========================================================================")
    scenario_params = {"override_relative_humidity_pct": 95.0, "delta_temperature_c": 2.0}
    print(f"Scenario Overrides Applied: Relative Humidity = 95.0%, Temperature +2.0°C\n")

    scenario_result = layer.analyze(
        image_path=str(sample_image_path),
        latitude=latitude,
        longitude=longitude,
        crop="Potato",
        scenario_params=scenario_params
    )

    payload_scen = scenario_result.to_dict()
    what_if = payload_scen["WHAT_IF"]

    print("-------------------------------------------------------------------------")
    print("BASELINE  ──►  SCENARIO  ──►  RISK CHANGE  ──►  HYPOTHETICAL RAG EXPLANATION")
    print("-------------------------------------------------------------------------")
    print(f" • BASELINE RISK LEVEL : {what_if['baseline_risk_level']} (Index: {what_if['baseline_risk_index']}%)")
    print(f" • SCENARIO RISK LEVEL : {what_if['scenario_risk_level']} (Index: {what_if['scenario_risk_index']}%)")
    print(f" • RISK INDEX DELTA    : +{what_if['risk_index_delta']}%")
    print(f" • RISK LEVEL CHANGED  : {what_if['risk_level_changed']}")
    print(f" • DISCLAIMER          : {what_if['disclaimer']}\n")

    print("--- HYPOTHETICAL SCENARIO RAG KNOWLEDGE QUERY & ANSWER ---")
    print(f" • Scenario RAG Query  : \"{payload_scen['KNOWLEDGE']['generated_query']}\"")
    print(f" • Grounded Answer     :\n{payload_scen['KNOWLEDGE']['grounded_answer']}\n")

    print("=========================================================================")
    print("                        MANDATORY SYSTEM DISCLAIMERS                     ")
    print("=========================================================================")
    for disc in payload_scen["DISCLAIMERS"]:
        print(f" ⚠️ {disc}")
    print("=========================================================================")

if __name__ == "__main__":
    main()
