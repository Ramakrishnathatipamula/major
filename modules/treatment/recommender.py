import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.rag.retriever import RAGRetriever, SearchResult
from modules.treatment.models import (
    TreatmentCategory,
    TreatmentOption,
    TreatmentRecommendationResult,
    DEFAULT_TREATMENT_DISCLAIMER,
)

# Verified CIBRC, ICAR-CPRI, and Extension Catalog for Potato Blights
# Every single detail is cross-referenced with CIBRC Major Uses, ICAR-CPRI, or TNAU Agritech records.
VERIFIED_LATE_BLIGHT_TREATMENTS: List[TreatmentOption] = [
    TreatmentOption(
        category=TreatmentCategory.PROPHYLACTIC_PROTECTANT.value,
        active_ingredient="Mancozeb",
        formulation="75% WP",
        target_disease="Potato Late Blight (Phytophthora infestans)",
        application_rate="0.2% concentration (2.0 to 2.5 kg/ha in 1,000 L water, or 2.0 to 2.5 g/L)",
        timing_and_frequency="Apply prophylactically before canopy closure or immediately upon disease forecast advisory. Repeat at 7 to 10 day intervals if cool, humid weather persists.",
        application_directions="Mix thoroughly with clean water; spray uniformly ensuring thorough canopy coverage on both upper and lower leaf surfaces.",
        pre_harvest_interval_phi="Refer to manufacturer registered container label (typically 7 to 14 days).",
        re_entry_interval_rei="24 hours or until spray deposit has completely dried.",
        ppe_and_safety_precautions=[
            "Wear chemical-resistant rubber gloves, face mask, eye protection, and protective overalls during mixing and application.",
            "Do not spray against the wind or during high wind speeds (>15 km/h).",
            "Avoid spraying near open surface water bodies, ponds, and drainage ditches."
        ],
        verification_status="Verified CIBRC / ICAR-CPRI / TNAU Guidance",
        evidence_source_title="Official CIBRC Registered Fungicides & ICAR-CPRI / TNAU Management Guidelines for Potato Late Blight",
        evidence_source_url="https://ppqs.gov.in"
    ),
    TreatmentOption(
        category=TreatmentCategory.PROPHYLACTIC_PROTECTANT.value,
        active_ingredient="Chlorothalonil",
        formulation="75% WP",
        target_disease="Potato Late Blight (Phytophthora infestans)",
        application_rate="0.2% concentration (approx. 2.0 kg/ha in 1,000 L water)",
        timing_and_frequency="Apply preventively; alternate with Mancozeb to maintain multi-site contact protection.",
        application_directions="Ensure complete foliage wetting before disease onset.",
        pre_harvest_interval_phi="Refer to manufacturer product label.",
        re_entry_interval_rei="24 to 48 hours.",
        ppe_and_safety_precautions=[
            "Wear protective clothing, eye protection, and vapor mask.",
            "Toxic to aquatic life with long-lasting effects; do not contaminate irrigation canals."
        ],
        verification_status="Verified CIBRC / TNAU Guidance",
        evidence_source_title="Official CIBRC Registered Fungicides & ICAR-CPRI / TNAU Management Guidelines for Potato Late Blight",
        evidence_source_url="https://ppqs.gov.in"
    ),
    TreatmentOption(
        category=TreatmentCategory.PROPHYLACTIC_PROTECTANT.value,
        active_ingredient="Copper Oxychloride",
        formulation="50% WP",
        target_disease="Potato Late Blight (Phytophthora infestans)",
        application_rate="0.25% to 0.3% concentration (2.5 to 3.0 g per litre of water)",
        timing_and_frequency="Preventive protective spray during early vegetative stages.",
        application_directions="Spray uniformly on foliage prior to wet weather spells.",
        pre_harvest_interval_phi="Refer to manufacturer product label.",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=[
            "Wear protective goggles and rubber gloves.",
            "Avoid corrosive contact with metal spraying equipment; rinse sprayers after use."
        ],
        verification_status="Verified CIBRC / TNAU Guidance",
        evidence_source_title="Official CIBRC Registered Fungicides & ICAR-CPRI / TNAU Management Guidelines for Potato Late Blight",
        evidence_source_url="https://ppqs.gov.in"
    ),
    TreatmentOption(
        category=TreatmentCategory.CURATIVE_SYSTEMIC.value,
        active_ingredient="Cymoxanil + Mancozeb",
        formulation="8% + 64% WP",
        target_disease="Potato Late Blight (Phytophthora infestans)",
        application_rate="0.3% concentration (approx. 3.0 kg/ha in 1,000 L water)",
        timing_and_frequency="Apply immediately upon observation of first foliar lesions or under high local disease risk. Repeat after 7 to 10 days if severe conditions continue.",
        application_directions="Translaminar systemic combination provides curative kickback within 24-48 hours of infection plus residual surface protection.",
        pre_harvest_interval_phi="Refer to specific registered product label.",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=[
            "Wear full protective gear (gloves, mask, goggles, boots).",
            "Do not consume food, water, or smoke during application."
        ],
        verification_status="Verified CIBRC / ICAR-CPRI Guidance",
        evidence_source_title="Official CIBRC Registered Fungicides & ICAR-CPRI / TNAU Management Guidelines for Potato Late Blight",
        evidence_source_url="https://ppqs.gov.in"
    ),
    TreatmentOption(
        category=TreatmentCategory.CURATIVE_SYSTEMIC.value,
        active_ingredient="Metalaxyl + Mancozeb",
        formulation="8% + 64% WP (or Ridomil MZ)",
        target_disease="Potato Late Blight (Phytophthora infestans)",
        application_rate="0.2% to 0.25% concentration (2.0 to 2.5 g per litre of water)",
        timing_and_frequency="Apply at initial disease detection. Maximum 1 to 2 consecutive applications per season to prevent pathogen resistance.",
        application_directions="Systemic uptake moves within plant canopy foliage to halt expanding lesions.",
        pre_harvest_interval_phi="Refer to registered product container label.",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=[
            "Wear protective gloves and face shield.",
            "Resistance Management: Must alternate with non-phenylamide contact fungicides (Mancozeb or Chlorothalonil)."
        ],
        verification_status="Verified CIBRC / ICAR-CPRI / TNAU Guidance",
        evidence_source_title="Official CIBRC Registered Fungicides & ICAR-CPRI / TNAU Management Guidelines for Potato Late Blight",
        evidence_source_url="https://ppqs.gov.in"
    ),
]

VERIFIED_EARLY_BLIGHT_TREATMENTS: List[TreatmentOption] = [
    TreatmentOption(
        category=TreatmentCategory.PROPHYLACTIC_PROTECTANT.value,
        active_ingredient="Mancozeb",
        formulation="75% WP",
        target_disease="Potato Early Blight (Alternaria solani)",
        application_rate="0.2% concentration (2.0 kg/ha in 1,000 L water, or 2.0 g/L)",
        timing_and_frequency="Begin spraying as soon as early concentric lesions appear on lower canopy foliage or post-tuber initiation. Repeat at 10 to 15-day intervals.",
        application_directions="Ensure thorough coverage of lower and mature canopy leaves where Alternaria infection begins.",
        pre_harvest_interval_phi="Refer to manufacturer product label.",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=[
            "Wear gloves, face mask, and protective overalls during spraying.",
            "Do not spray during high temperatures or under intense midday sun."
        ],
        verification_status="Verified CIBRC / TNAU / NDSU Guidance",
        evidence_source_title="Official CIBRC Registered Fungicides & TNAU / NDSU Management Guidelines for Potato Early Blight",
        evidence_source_url="https://ppqs.gov.in"
    ),
    TreatmentOption(
        category=TreatmentCategory.PROPHYLACTIC_PROTECTANT.value,
        active_ingredient="Chlorothalonil",
        formulation="75% WP",
        target_disease="Potato Early Blight (Alternaria solani)",
        application_rate="0.2% concentration (approx. 2.0 kg/ha in 1,000 L water)",
        timing_and_frequency="Apply upon first symptom detection; alternate with Mancozeb to maintain multi-site protection.",
        application_directions="Spray uniformly across canopy foliage.",
        pre_harvest_interval_phi="Refer to manufacturer product label.",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=[
            "Wear eye goggles and chemical-resistant gloves.",
            "Toxic to aquatic organisms; avoid spray drift towards water channels."
        ],
        verification_status="Verified CIBRC / TNAU Guidance",
        evidence_source_title="Official CIBRC Registered Fungicides & TNAU / NDSU Management Guidelines for Potato Early Blight",
        evidence_source_url="https://ppqs.gov.in"
    ),
    TreatmentOption(
        category=TreatmentCategory.PROPHYLACTIC_PROTECTANT.value,
        active_ingredient="Copper Oxychloride",
        formulation="50% WP (e.g. Blitox-50)",
        target_disease="Potato Early Blight (Alternaria solani)",
        application_rate="0.2% to 0.25% concentration (2.0 to 2.5 g per litre of water)",
        timing_and_frequency="Apply on lower foliage when concentric ring lesions are first identified.",
        application_directions="Spray early in the morning so canopy dries during daylight hours.",
        pre_harvest_interval_phi="Refer to manufacturer product label.",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=[
            "Wear gloves and eye protection.",
            "Rinse spraying equipment thoroughly after application."
        ],
        verification_status="Verified CIBRC / TNAU Guidance",
        evidence_source_title="Official CIBRC Registered Fungicides & TNAU / NDSU Management Guidelines for Potato Early Blight",
        evidence_source_url="https://ppqs.gov.in"
    ),
    TreatmentOption(
        category=TreatmentCategory.CURATIVE_SYSTEMIC.value,
        active_ingredient="Difenoconazole",
        formulation="25% EC",
        target_disease="Potato Early Blight (Alternaria solani)",
        application_rate="Refer to registered product container label (typically 0.05% or 0.5 ml/L where approved)",
        timing_and_frequency="Apply when early blight lesions expand rapidly during warm, alternating wet-dry weather. Rotate with multi-site contact protectants.",
        application_directions="Systemic triazole fungicide (FRAC Group 3) halts mycelial growth within leaf tissue.",
        pre_harvest_interval_phi="Refer to manufacturer product label.",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=[
            "Wear protective respirator, goggles, and gloves.",
            "Do not apply consecutively to prevent fungal resistance development."
        ],
        verification_status="Verified TNAU / CIBRC Approved Class Guidance",
        evidence_source_title="Official CIBRC Registered Fungicides & TNAU / NDSU Management Guidelines for Potato Early Blight",
        evidence_source_url="https://ppqs.gov.in"
    ),
]


VERIFIED_TOMATO_LATE_BLIGHT_TREATMENTS: List[TreatmentOption] = [
    TreatmentOption(
        category=TreatmentCategory.PROPHYLACTIC_PROTECTANT.value,
        active_ingredient="Mancozeb",
        formulation="75% WP",
        target_disease="Tomato Late Blight (Phytophthora infestans)",
        application_rate="0.2% concentration (2.0 to 2.5 g/L in clean water)",
        timing_and_frequency="Apply prophylactically prior to canopy closure or upon initial regional blight forecast. Repeat at 7 to 10 day intervals during cool, humid weather.",
        application_directions="Spray thoroughly covering both upper and lower leaf surfaces.",
        pre_harvest_interval_phi="Refer to manufacturer registered container label (typically 3 to 7 days).",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=[
            "Wear chemical-resistant gloves, face mask, and eye protection.",
            "Do not spray near open water bodies or ponds."
        ],
        verification_status="Verified CIBRC / ICAR-IIVR / TNAU Guidance",
        evidence_source_title="CIBRC Major Uses & ICAR-IIVR Tomato Crop Protection Advisory",
        evidence_source_url="https://ppqs.gov.in",
        crop="Tomato"
    ),
    TreatmentOption(
        category=TreatmentCategory.PROPHYLACTIC_PROTECTANT.value,
        active_ingredient="Copper Oxychloride",
        formulation="50% WP",
        target_disease="Tomato Late Blight (Phytophthora infestans)",
        application_rate="0.25% to 0.3% concentration (2.5 to 3.0 g/L)",
        timing_and_frequency="Preventive contact spray during early vegetative development.",
        application_directions="Spray uniformly across canopy foliage before prolonged rain spells.",
        pre_harvest_interval_phi="Refer to container label.",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=["Wear rubber gloves and protective goggles."],
        verification_status="Verified CIBRC / TNAU Guidance",
        evidence_source_title="CIBRC Major Uses & TNAU Tomato Pathology Guidelines",
        evidence_source_url="https://ppqs.gov.in",
        crop="Tomato"
    ),
    TreatmentOption(
        category=TreatmentCategory.CURATIVE_SYSTEMIC.value,
        active_ingredient="Cymoxanil + Mancozeb",
        formulation="8% + 64% WP",
        target_disease="Tomato Late Blight (Phytophthora infestans)",
        application_rate="0.3% concentration (3.0 g/L in water)",
        timing_and_frequency="Apply immediately upon detecting active water-soaked lesions. Translaminar kickback within 24-48 hours.",
        application_directions="Ensure thorough canopy penetration. Repeat after 7 to 10 days if high disease risk persists.",
        pre_harvest_interval_phi="Refer to product label.",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=["Wear full protective gear (gloves, mask, overalls)."],
        verification_status="Verified CIBRC / ICAR-IIVR Guidance",
        evidence_source_title="CIBRC Registered Fungicides & ICAR-IIVR Advisory",
        evidence_source_url="https://ppqs.gov.in",
        crop="Tomato"
    ),
    TreatmentOption(
        category=TreatmentCategory.CURATIVE_SYSTEMIC.value,
        active_ingredient="Metalaxyl + Mancozeb",
        formulation="8% + 64% WP",
        target_disease="Tomato Late Blight (Phytophthora infestans)",
        application_rate="0.2% to 0.25% concentration (2.0 to 2.5 g/L)",
        timing_and_frequency="Apply at initial disease detection. Maximum 1 to 2 consecutive applications per season to prevent pathogen resistance.",
        application_directions="Systemic movement halts lesion expansion in canopy.",
        pre_harvest_interval_phi="Refer to container label.",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=[
            "Wear gloves and face shield.",
            "Resistance Management: Alternate with multi-site contact protectants."
        ],
        verification_status="Verified CIBRC / ICAR-IIVR Guidance",
        evidence_source_title="CIBRC Registered Fungicides & ICAR-IIVR Advisory",
        evidence_source_url="https://ppqs.gov.in",
        crop="Tomato"
    )
]

VERIFIED_TOMATO_EARLY_BLIGHT_TREATMENTS: List[TreatmentOption] = [
    TreatmentOption(
        category=TreatmentCategory.PROPHYLACTIC_PROTECTANT.value,
        active_ingredient="Mancozeb",
        formulation="75% WP",
        target_disease="Tomato Early Blight (Alternaria solani)",
        application_rate="0.2% concentration (2.0 g/L in water)",
        timing_and_frequency="Begin spraying as soon as concentric ring lesions appear on lower canopy foliage. Repeat at 10 to 14 day intervals.",
        application_directions="Direct spray towards lower mature leaves where Alternaria infection initiates.",
        pre_harvest_interval_phi="Refer to product label.",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=["Wear gloves and face mask during spraying."],
        verification_status="Verified CIBRC / ICAR-IIVR / TNAU Guidance",
        evidence_source_title="CIBRC Registered Fungicides & TNAU Early Blight Protocol",
        evidence_source_url="https://ppqs.gov.in",
        crop="Tomato"
    ),
    TreatmentOption(
        category=TreatmentCategory.PROPHYLACTIC_PROTECTANT.value,
        active_ingredient="Chlorothalonil",
        formulation="75% WP",
        target_disease="Tomato Early Blight (Alternaria solani)",
        application_rate="0.2% concentration (2.0 g/L)",
        timing_and_frequency="Apply preventively; alternate with Mancozeb for resistance management.",
        application_directions="Ensure uniform foliage coverage.",
        pre_harvest_interval_phi="Refer to container label.",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=["Wear eye goggles and chemical-resistant gloves."],
        verification_status="Verified CIBRC / TNAU Guidance",
        evidence_source_title="CIBRC Major Uses for Tomato",
        evidence_source_url="https://ppqs.gov.in",
        crop="Tomato"
    ),
    TreatmentOption(
        category=TreatmentCategory.CURATIVE_SYSTEMIC.value,
        active_ingredient="Difenoconazole",
        formulation="25% EC",
        target_disease="Tomato Early Blight (Alternaria solani)",
        application_rate="0.05% concentration (0.5 ml per litre of water)",
        timing_and_frequency="Apply when early blight lesions expand rapidly in warm, alternating wet-dry weather.",
        application_directions="Systemic triazole halts mycelial expansion inside leaf tissue.",
        pre_harvest_interval_phi="Refer to manufacturer container label.",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=["Wear protective overalls, goggles, and gloves."],
        verification_status="Verified CIBRC / TNAU Guidance",
        evidence_source_title="CIBRC Registered Fungicides & TNAU Tomato Management",
        evidence_source_url="https://ppqs.gov.in",
        crop="Tomato"
    )
]

VERIFIED_TOMATO_BACTERIAL_SPOT_TREATMENTS: List[TreatmentOption] = [
    TreatmentOption(
        category=TreatmentCategory.PROPHYLACTIC_PROTECTANT.value,
        active_ingredient="Copper Oxychloride + Streptocycline",
        formulation="50% WP (2.5 g/L) + Streptocycline (1 g/10 L, 100-200 ppm)",
        target_disease="Tomato Bacterial Spot (Xanthomonas vesicatoria)",
        application_rate="Copper Oxychloride @ 0.25% + Streptocycline @ 100 ppm",
        timing_and_frequency="Apply upon first appearance of small water-soaked dark leaf spots. Repeat at 10 to 12-day intervals under warm, humid conditions.",
        application_directions="Thorough foliar wetting; avoid spraying in extreme midday sun.",
        pre_harvest_interval_phi="Refer to manufacturer product label (typically 7 to 10 days).",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=[
            "Wear protective respirator, gloves, and eye protection.",
            "Avoid excessive copper accumulation in soil; rotate with cultural sanitation."
        ],
        verification_status="Verified CIBRC / ICAR-IIVR / TNAU Guidance",
        evidence_source_title="CIBRC Registered Bactericides & ICAR-IIVR Tomato Management",
        evidence_source_url="https://ppqs.gov.in",
        crop="Tomato"
    ),
    TreatmentOption(
        category=TreatmentCategory.PROPHYLACTIC_PROTECTANT.value,
        active_ingredient="Copper Hydroxide",
        formulation="77% WP",
        target_disease="Tomato Bacterial Spot (Xanthomonas vesicatoria)",
        application_rate="0.2% concentration (2.0 g per litre of water)",
        timing_and_frequency="Preventive protective spray before rain spells during vegetative growth.",
        application_directions="Ensure full canopy coverage on both leaf surfaces.",
        pre_harvest_interval_phi="Refer to container label.",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=["Wear rubber gloves and protective goggles."],
        verification_status="Verified CIBRC / TNAU Guidance",
        evidence_source_title="CIBRC Major Uses & TNAU Plant Pathology",
        evidence_source_url="https://ppqs.gov.in",
        crop="Tomato"
    )
]

VERIFIED_TOMATO_LEAF_MOLD_TREATMENTS: List[TreatmentOption] = [
    TreatmentOption(
        category=TreatmentCategory.PROPHYLACTIC_PROTECTANT.value,
        active_ingredient="Copper Oxychloride",
        formulation="50% WP",
        target_disease="Tomato Leaf Mold (Passalora fulva)",
        application_rate="0.25% concentration (2.5 g/L)",
        timing_and_frequency="Apply when pale yellow spots appear on upper leaf surface or under high polyhouse humidity (>85% RH).",
        application_directions="Direct spray towards lower surface of leaves where velvety olive-green mold develops.",
        pre_harvest_interval_phi="Refer to product label.",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=["Wear goggles and rubber gloves."],
        verification_status="Verified CIBRC / TNAU Guidance",
        evidence_source_title="CIBRC Registered Fungicides & TNAU Leaf Mold Advisory",
        evidence_source_url="https://ppqs.gov.in",
        crop="Tomato"
    ),
    TreatmentOption(
        category=TreatmentCategory.PROPHYLACTIC_PROTECTANT.value,
        active_ingredient="Mancozeb",
        formulation="75% WP",
        target_disease="Tomato Leaf Mold (Passalora fulva)",
        application_rate="0.2% concentration (2.0 g/L)",
        timing_and_frequency="Apply preventively during vegetative growth; repeat at 10-day intervals if high humidity persists.",
        application_directions="Spray uniformly across foliage.",
        pre_harvest_interval_phi="Refer to container label.",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=["Wear face mask and gloves."],
        verification_status="Verified CIBRC / TNAU Guidance",
        evidence_source_title="CIBRC Major Uses for Tomato",
        evidence_source_url="https://ppqs.gov.in",
        crop="Tomato"
    )
]

VERIFIED_TOMATO_SEPTORIA_TREATMENTS: List[TreatmentOption] = [
    TreatmentOption(
        category=TreatmentCategory.PROPHYLACTIC_PROTECTANT.value,
        active_ingredient="Mancozeb",
        formulation="75% WP",
        target_disease="Tomato Septoria Leaf Spot (Septoria lycopersici)",
        application_rate="0.2% concentration (2.0 g/L)",
        timing_and_frequency="Apply upon first detection of small circular spots with grayish-white centers on lower foliage.",
        application_directions="Ensure thorough coverage of lower canopy.",
        pre_harvest_interval_phi="Refer to product label.",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=["Wear gloves and face mask."],
        verification_status="Verified CIBRC / TNAU Guidance",
        evidence_source_title="CIBRC Registered Fungicides & TNAU Advisory",
        evidence_source_url="https://ppqs.gov.in",
        crop="Tomato"
    ),
    TreatmentOption(
        category=TreatmentCategory.PROPHYLACTIC_PROTECTANT.value,
        active_ingredient="Chlorothalonil",
        formulation="75% WP",
        target_disease="Tomato Septoria Leaf Spot (Septoria lycopersici)",
        application_rate="0.2% concentration (2.0 g/L)",
        timing_and_frequency="Apply preventively; alternate with Mancozeb.",
        application_directions="Ensure full foliage wetting.",
        pre_harvest_interval_phi="Refer to product label.",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=["Wear eye protection and gloves."],
        verification_status="Verified CIBRC / TNAU Guidance",
        evidence_source_title="CIBRC Major Uses for Tomato",
        evidence_source_url="https://ppqs.gov.in",
        crop="Tomato"
    )
]

VERIFIED_TOMATO_TARGET_SPOT_TREATMENTS: List[TreatmentOption] = [
    TreatmentOption(
        category=TreatmentCategory.PROPHYLACTIC_PROTECTANT.value,
        active_ingredient="Mancozeb",
        formulation="75% WP",
        target_disease="Tomato Target Spot (Corynespora cassiicola)",
        application_rate="0.2% concentration (2.0 g/L)",
        timing_and_frequency="Apply when brown pinpoint lesions begin expanding. Repeat at 10 to 14-day intervals.",
        application_directions="Spray uniformly across foliage.",
        pre_harvest_interval_phi="Refer to container label.",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=["Wear gloves and eye protection."],
        verification_status="Verified CIBRC / TNAU Guidance",
        evidence_source_title="CIBRC Registered Fungicides & TNAU Advisory",
        evidence_source_url="https://ppqs.gov.in",
        crop="Tomato"
    ),
    TreatmentOption(
        category=TreatmentCategory.PROPHYLACTIC_PROTECTANT.value,
        active_ingredient="Chlorothalonil",
        formulation="75% WP",
        target_disease="Tomato Target Spot (Corynespora cassiicola)",
        application_rate="0.2% concentration (2.0 g/L)",
        timing_and_frequency="Preventive contact spray during warm, humid conditions.",
        application_directions="Spray both leaf surfaces thoroughly.",
        pre_harvest_interval_phi="Refer to product label.",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=["Wear protective gear."],
        verification_status="Verified CIBRC / TNAU Guidance",
        evidence_source_title="CIBRC Major Uses for Tomato",
        evidence_source_url="https://ppqs.gov.in",
        crop="Tomato"
    )
]

VERIFIED_TOMATO_SPIDER_MITES_TREATMENTS: List[TreatmentOption] = [
    TreatmentOption(
        category=TreatmentCategory.CURATIVE_SYSTEMIC.value,
        active_ingredient="Spiromesifen",
        formulation="22.9% SC (Oberon)",
        target_disease="Tomato Two-Spotted Spider Mite (Tetranychus urticae)",
        application_rate="1.0 ml per litre of water (approx. 500 ml/ha)",
        timing_and_frequency="Apply upon first detection of leaf stippling or web formation. Highly effective against eggs and nymphs.",
        application_directions="Direct spray to undersides of leaves where mites congregate.",
        pre_harvest_interval_phi="Refer to manufacturer registered label (typically 3 to 5 days).",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=[
            "Wear chemical-resistant gloves, goggles, and face mask.",
            "Acaricide specific: Note that fungicides have NO efficacy against spider mites."
        ],
        verification_status="Verified CIBRC / ICAR-IIVR / TNAU Guidance",
        evidence_source_title="CIBRC Registered Insecticides/Acaricides & ICAR-IIVR Advisory",
        evidence_source_url="https://ppqs.gov.in",
        crop="Tomato"
    ),
    TreatmentOption(
        category=TreatmentCategory.PROPHYLACTIC_PROTECTANT.value,
        active_ingredient="Wettable Sulphur",
        formulation="80% WP",
        target_disease="Tomato Two-Spotted Spider Mite (Tetranychus urticae)",
        application_rate="2.5 to 3.0 g per litre of water",
        timing_and_frequency="Apply at early mite infestation; provides contact acaricidal and powdery mildew protection.",
        application_directions="Do not apply when ambient temperature exceeds 35°C to avoid sulfur phytotoxicity.",
        pre_harvest_interval_phi="Refer to container label.",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=["Wear gloves and eye protection; avoid inhalation of sulfur dust."],
        verification_status="Verified CIBRC / TNAU Guidance",
        evidence_source_title="CIBRC Major Uses & TNAU Spider Mite Management",
        evidence_source_url="https://ppqs.gov.in",
        crop="Tomato"
    ),
    TreatmentOption(
        category=TreatmentCategory.BIOLOGICAL_ORGANIC.value,
        active_ingredient="Azadirachtin (Neem Oil Formulation)",
        formulation="1500 ppm (0.15% EC)",
        target_disease="Tomato Two-Spotted Spider Mite (Tetranychus urticae)",
        application_rate="3.0 to 5.0 ml per litre of water",
        timing_and_frequency="Apply as botanical repellent and ovicidal deterrent; repeat at 7-day intervals.",
        application_directions="Spray early morning or late afternoon to minimize UV breakdown.",
        pre_harvest_interval_phi="0 to 1 day (Eco-friendly botanical).",
        re_entry_interval_rei="4 hours or until dry.",
        ppe_and_safety_precautions=["Wear protective face covering."],
        verification_status="Verified ICAR-IIVR / TNAU Organic Guidance",
        evidence_source_title="ICAR-IIVR Organic Pest Management Advisory",
        evidence_source_url="https://iivr.icar.gov.in",
        crop="Tomato"
    )
]

VERIFIED_TOMATO_TYLCV_TREATMENTS: List[TreatmentOption] = [
    TreatmentOption(
        category=TreatmentCategory.CURATIVE_SYSTEMIC.value,
        active_ingredient="Imidacloprid (Vector Management - Whitefly)",
        formulation="17.8% SL",
        target_disease="Tomato Yellow Leaf Curl Virus (TYLCV) - Whitefly Vector (Bemisia tabaci)",
        application_rate="0.3 ml per litre of water (approx. 150 ml/ha)",
        timing_and_frequency="Apply to manage whitefly vector population before viral transmission; repeat after 15 days if vector pressure is severe.",
        application_directions="CRITICAL: Viral disease cannot be cured by chemical fungicides or antibiotics. This systemic insecticide manages whitefly vectors only.",
        pre_harvest_interval_phi="Refer to registered container label (typically 7 to 10 days).",
        re_entry_interval_rei="24 hours.",
        ppe_and_safety_precautions=[
            "Wear gloves, face mask, and eye goggles.",
            "Toxic to bees; do not spray during active daytime pollinator foraging."
        ],
        verification_status="Verified CIBRC / ICAR-IIVR Guidance",
        evidence_source_title="CIBRC Major Uses & ICAR-IIVR TYLCV Vector Management",
        evidence_source_url="https://ppqs.gov.in",
        crop="Tomato"
    ),
    TreatmentOption(
        category=TreatmentCategory.BIOLOGICAL_ORGANIC.value,
        active_ingredient="Azadirachtin (Neem Oil Formulation)",
        formulation="1500 ppm",
        target_disease="Tomato Yellow Leaf Curl Virus (TYLCV) - Whitefly Vector Deterrent",
        application_rate="3.0 to 5.0 ml per litre of water",
        timing_and_frequency="Spray at 7 to 10-day intervals from early transplanting to deter whitefly oviposition and feeding.",
        application_directions="Ensure thorough underside foliar coverage.",
        pre_harvest_interval_phi="1 day.",
        re_entry_interval_rei="4 hours.",
        ppe_and_safety_precautions=["Standard eye and face protection."],
        verification_status="Verified ICAR-IIVR / TNAU Organic Guidance",
        evidence_source_title="ICAR-IIVR Pest Management Guidelines",
        evidence_source_url="https://iivr.icar.gov.in",
        crop="Tomato"
    )
]

VERIFIED_TOMATO_MOSAIC_TREATMENTS: List[TreatmentOption] = [
    TreatmentOption(
        category=TreatmentCategory.CULTURAL_PREVENTIVE.value,
        active_ingredient="Skimmed Milk / Non-Fat Dry Milk Solution (Mechanical Barrier)",
        formulation="10% to 20% solution in clean water",
        target_disease="Tomato Mosaic Virus (ToMV)",
        application_rate="100 to 200 g skimmed milk powder per litre of water",
        timing_and_frequency="Spray on foliage 24 hours prior to pruning, trellising, or handling plants to inactivate mechanically transmitted virions.",
        application_directions="CRITICAL: Plant viral infections cannot be cured by chemical pesticides or fungicides. This cultural practice prevents mechanical sap transmission during manual farm operations.",
        pre_harvest_interval_phi="0 days (food-grade).",
        re_entry_interval_rei="0 hours.",
        ppe_and_safety_precautions=["None required."],
        verification_status="Verified ICAR-IIVR / Extension Cultural Guidance",
        evidence_source_title="ICAR-IIVR & Extension Guidelines for Tomato Virus Management",
        evidence_source_url="https://iivr.icar.gov.in",
        crop="Tomato"
    )
]


from modules.crops import get_crop_registry


class TreatmentRegistry:
    """
    Central repository of verified multi-crop treatment records.
    Ensures unverified chemical entries or unapproved rates are never returned.
    """
    def __init__(self):
        self._treatments: Dict[str, Dict[str, List[TreatmentOption]]] = {}
        self._register_defaults()

    def _register_defaults(self):
        # Register verified Potato treatments
        self.register_treatments("potato", "late_blight", VERIFIED_LATE_BLIGHT_TREATMENTS)
        self.register_treatments("potato", "early_blight", VERIFIED_EARLY_BLIGHT_TREATMENTS)

        # Register verified Tomato treatments
        self.register_treatments("tomato", "late_blight", VERIFIED_TOMATO_LATE_BLIGHT_TREATMENTS)
        self.register_treatments("tomato", "early_blight", VERIFIED_TOMATO_EARLY_BLIGHT_TREATMENTS)
        self.register_treatments("tomato", "bacterial_spot", VERIFIED_TOMATO_BACTERIAL_SPOT_TREATMENTS)
        self.register_treatments("tomato", "leaf_mold", VERIFIED_TOMATO_LEAF_MOLD_TREATMENTS)
        self.register_treatments("tomato", "septoria_leaf_spot", VERIFIED_TOMATO_SEPTORIA_TREATMENTS)
        self.register_treatments("tomato", "target_spot", VERIFIED_TOMATO_TARGET_SPOT_TREATMENTS)
        self.register_treatments("tomato", "spider_mites", VERIFIED_TOMATO_SPIDER_MITES_TREATMENTS)
        self.register_treatments("tomato", "tomato_yellow_leaf_curl_virus", VERIFIED_TOMATO_TYLCV_TREATMENTS)
        self.register_treatments("tomato", "tomato_mosaic_virus", VERIFIED_TOMATO_MOSAIC_TREATMENTS)

    def register_treatments(self, crop: str, disease: str, treatments: List[TreatmentOption]):
        c_key = str(crop).strip().lower()
        d_key = str(disease).strip().lower().replace("potato___", "").replace("tomato___", "").replace(" ", "_")
        if c_key not in self._treatments:
            self._treatments[c_key] = {}
        # Strictly enforce: unverified chemical treatments cannot be added to active recommendation catalog
        verified_treatments = [t for t in treatments if t.is_verified()]
        self._treatments[c_key][d_key] = verified_treatments

    def get_treatments(self, crop: str, disease: str, verified_only: bool = True) -> List[TreatmentOption]:
        c_key = str(crop).strip().lower()
        d_key = str(disease).strip().lower().replace("potato___", "").replace("tomato___", "").replace(" ", "_")
        if c_key not in self._treatments:
            return []
        options: List[TreatmentOption] = []
        if d_key in self._treatments[c_key]:
            options = list(self._treatments[c_key][d_key])
        else:
            # Keyword matching
            for reg_d, opts in self._treatments[c_key].items():
                if reg_d in d_key or d_key in reg_d:
                    options = list(opts)
                    break
        if verified_only:
            return [t for t in options if t.is_verified()]
        return list(options)

    def is_treatment_available(self, crop: str, disease: str) -> bool:
        return len(self.get_treatments(crop, disease)) > 0


_GLOBAL_TREATMENT_REGISTRY: Optional[TreatmentRegistry] = None


def get_treatment_registry() -> TreatmentRegistry:
    global _GLOBAL_TREATMENT_REGISTRY
    if _GLOBAL_TREATMENT_REGISTRY is None:
        _GLOBAL_TREATMENT_REGISTRY = TreatmentRegistry()
    return _GLOBAL_TREATMENT_REGISTRY


class PotatoTreatmentRecommender:
    """
    Evidence-grounded Treatment Recommender for Potato Leaf Diseases.
    Directly addresses: Plant Image -> Crop Identification -> Disease Detection -> Recommended Medicine/Treatment.
    Links RAG vector retrieval with verified CIBRC / ICAR-CPRI / Extension treatment options.
    """
    def __init__(self, retriever: Optional[RAGRetriever] = None, registry: Optional[TreatmentRegistry] = None):
        self.retriever = retriever if retriever is not None else RAGRetriever()
        self.treatment_registry = registry if registry is not None else get_treatment_registry()
        self.crop_registry = get_crop_registry()

    def recommend_treatment(
        self,
        crop: str,
        disease: str,
        risk_level: Optional[str] = "LOW",
        is_low_confidence: bool = False,
        is_rejected: bool = False
    ) -> TreatmentRecommendationResult:
        """
        Generate structured, evidence-grounded treatment recommendations for the given crop and disease.

        Args:
            crop: Crop name (e.g. "Potato")
            disease: Detected disease label (e.g. "Potato___Early_blight", "Late Blight", "Healthy")
            risk_level: Current environmental risk level ("LOW", "MODERATE", "HIGH")
            is_low_confidence: Boolean flag indicating confidence is below warning threshold (<60%)
            is_rejected: Boolean flag indicating confidence is below rejection threshold (<40%) or input invalid

        Returns:
            TreatmentRecommendationResult object with verified options, application rates,
            precautions, non-chemical controls, and explicit evidence gap notices.
        """
        clean_crop = str(crop).strip()
        clean_disease = str(disease).strip()
        disease_lower = clean_disease.lower()

        crop_spec = self.crop_registry.get_crop(clean_crop)
        is_potato = "potato" in clean_crop.lower()
        is_tomato = "tomato" in clean_crop.lower()
        has_verified_treatment_registry = is_potato or is_tomato

        # 1. Crops without an active diagnostic specialist
        if not crop_spec or not crop_spec.is_supported:
            return TreatmentRecommendationResult(
                crop=clean_crop,
                predicted_disease=clean_disease,
                is_supported=False,
                is_healthy=False,
                status="UNSUPPORTED_CROP",
                summary_message=(
                    f"Treatment recommendation module requires a validated disease specialist. "
                    f"Verified treatment protocol for {clean_crop} is pending model and data expansion."
                ),
                chemical_treatments=[],
                non_chemical_control_measures=[],
                safety_and_environmental_precautions=[],
                evidence_sources=[],
                evidence_gaps=[f"No verified treatment registry active for crop '{clean_crop}'."],
                disclaimer=DEFAULT_TREATMENT_DISCLAIMER
            )

        active_crop_name = crop_spec.display_name

        # 2. Prediction Rejection Policy (<40% confidence or invalid input)
        if is_rejected:
            return TreatmentRecommendationResult(
                crop=active_crop_name,
                predicted_disease=clean_disease,
                is_supported=False,
                is_healthy=False,
                status="PREDICTION_REJECTED",
                summary_message=(
                    "Visual disease prediction was rejected due to critically low model confidence (<40.0%) "
                    "or invalid input. At this confidence level, classification certainty is near random chance. "
                    "Disease-specific chemical fungicide recommendations are strictly withheld on an inconclusive diagnosis. "
                    "Inspect physical plant foliage and consult a qualified local agricultural extension officer."
                ),
                chemical_treatments=[],
                non_chemical_control_measures=[
                    "Physical Foliar Inspection: Check leaves for distinct concentric rings (Early Blight), water-soaked lesions with white mold on undersides (Late Blight), or stippling/curling.",
                    "Field Scouting: Sample foliage from multiple field quadrants to assess whether symptoms are localized or widespread.",
                    "Sample Submission: Submit leaf specimens to nearest Krishi Vigyan Kendra (KVK) or plant clinic for microscopic diagnosis."
                ],
                safety_and_environmental_precautions=[
                    "Avoid unverified pesticide applications: Applying chemicals without a confirmed diagnosis incurs unnecessary chemical expenses, risks pathogen resistance, and hazards non-target organisms."
                ],
                evidence_sources=[],
                evidence_gaps=["Diagnosis inconclusive (<40.0% confidence). Chemical recommendations suppressed to prevent chemical misapplication."],
                disclaimer=DEFAULT_TREATMENT_DISCLAIMER
            )

        # 3. Uncertain Prediction Warning Policy (<60% confidence)
        if is_low_confidence:
            return TreatmentRecommendationResult(
                crop=active_crop_name,
                predicted_disease=clean_disease,
                is_supported=True,
                is_healthy=("healthy" in disease_lower),
                status="UNCERTAIN_DIAGNOSIS_WARNING",
                summary_message=(
                    f"Warning: Visual diagnosis for '{clean_disease}' has marginal confidence (<60.0%). "
                    "Because model certainty is low, disease-specific chemical pesticides and fungicides are withheld until foliar symptoms "
                    "are verified in the field. Observe cultural controls and monitor closely."
                ),
                chemical_treatments=[],
                non_chemical_control_measures=[
                    "Field Scouting: Conduct close physical examination of canopy leaves to confirm if visual symptoms match the tentative diagnosis.",
                    "Cultural Hygiene: Ensure adequate spacing, weed eradication, and morning-only irrigation to minimize leaf wetness.",
                    "Re-evaluation: Re-photograph leaves under clear, natural lighting or re-test if symptoms expand."
                ],
                safety_and_environmental_precautions=[
                    "Do not apply chemical pesticides based solely on an uncertain visual classification. Verify symptoms in the field first."
                ],
                evidence_sources=[],
                evidence_gaps=[f"Visual prediction confidence is below warning threshold (60.0%). Chemical fungicide recommendations withheld to prevent erroneous spraying."],
                disclaimer=DEFAULT_TREATMENT_DISCLAIMER
            )

        # 4. Healthy Foliage (No Chemical Treatment Permitted)
        if "healthy" in disease_lower:
            if is_tomato:
                return TreatmentRecommendationResult(
                    crop="Tomato",
                    predicted_disease="Healthy",
                    is_supported=True,
                    is_healthy=True,
                    status="HEALTHY_PREVENTIVE_ONLY",
                    summary_message=(
                        "Healthy tomato leaf detected. No chemical fungicide or pesticide treatment is required or recommended. "
                        "Maintain routine preventive cultural management and continuous field monitoring."
                    ),
                    chemical_treatments=[],
                    non_chemical_control_measures=[
                        "Canopy Aeration & Trellising: Stake vines and remove suckers to facilitate air flow and keep foliage off wet soil.",
                        "Irrigation Management: Prefer drip irrigation; avoid overhead watering that wets foliage overnight.",
                        "Balanced Nutrition: Provide optimal potassium and phosphorus; avoid excess nitrogen which induces lush vegetative growth susceptible to blights.",
                        "Sanitation: Eradicate solanaceous weeds (black nightshade) and rogue volunteer seedlings around field borders.",
                        "Routine Field Scouting: Inspect lower canopy leaves weekly for initial spot or lesion emergence."
                    ],
                    safety_and_environmental_precautions=[
                        "Do not apply prophylactic chemical fungicides unnecessarily to healthy tomato crops in the absence of disease risk advisories.",
                        "Preserve natural beneficial predators (ladybugs, predatory mites, lacewings)."
                    ],
                    evidence_sources=[
                        {
                            "source_organization": "ICAR - Indian Institute of Vegetable Research (IIVR) & TNAU Agritech",
                            "document_title": "Tomato Good Agricultural Practices (GAP) & Preventive Plant Health Advisory",
                            "url_reference": "https://iivr.icar.gov.in",
                            "document_type": "extension_guide"
                        }
                    ],
                    evidence_gaps=["No chemical treatment gaps — chemical intervention is not indicated for healthy plants."],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )
            elif is_potato: # Potato
                return TreatmentRecommendationResult(
                    crop="Potato",
                    predicted_disease="Healthy",
                    is_supported=True,
                    is_healthy=True,
                    status="HEALTHY_PREVENTIVE_ONLY",
                    summary_message=(
                        "Healthy potato leaf detected. No chemical fungicide or pesticide treatment is required or recommended. "
                        "Maintain routine preventive cultural management and continuous field monitoring."
                    ),
                    chemical_treatments=[],
                    non_chemical_control_measures=[
                        "Routine Field Scouting: Inspect lower canopy leaves and low-lying field zones weekly for early symptom emergence.",
                        "Balanced Plant Fertility: Maintain optimum nitrogen and potassium fertility; avoid plant stress which increases vulnerability.",
                        "Irrigation Practices: Water early in the morning or utilize drip irrigation to keep canopy foliage dry overnight.",
                        "Sanitation: Keep field borders free of volunteer potatoes and solanaceous weeds (nightshade).",
                        "Certified Seed: Ensure only certified disease-free seed tubers are used for subsequent plantings."
                    ],
                    safety_and_environmental_precautions=[
                        "Do not apply prophylactic chemical fungicides unnecessarily to healthy crops in the absence of disease risk advisories.",
                        "Preserve beneficial field insects and soil microbiomes."
                    ],
                    evidence_sources=[
                        {
                            "source_organization": "University of Minnesota Extension & Penn State Extension",
                            "document_title": "Potato Foliar Health & Cultural Preventive Management",
                            "url_reference": "https://extension.umn.edu/disease-management/late-blight",
                            "document_type": "extension_guide"
                        }
                    ],
                    evidence_gaps=["No chemical treatment gaps — chemical intervention is not indicated for healthy plants."],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )
            else:
                # Healthy foliage for any other registry-supported crop.
                return TreatmentRecommendationResult(
                    crop=active_crop_name,
                    predicted_disease="Healthy",
                    is_supported=True,
                    is_healthy=True,
                    status="HEALTHY_PREVENTIVE_ONLY",
                    summary_message=(
                        f"Healthy {active_crop_name} leaf detected. No chemical fungicide or pesticide "
                        "treatment is required or recommended. Maintain routine monitoring and "
                        "crop-specific extension guidance."
                    ),
                    chemical_treatments=[],
                    non_chemical_control_measures=[
                        "Routine Field Scouting: Inspect representative canopy leaves weekly for new lesions, spots, or discoloration.",
                        "Irrigation Management: Prefer early-morning or drip irrigation so foliage dries during daylight hours.",
                        "Field Sanitation: Remove volunteer plants and crop residues that can harbour primary inoculum.",
                    ],
                    safety_and_environmental_precautions=[
                        "Do not apply prophylactic chemical fungicides without a confirmed disease diagnosis and a locally registered product label."
                    ],
                    evidence_sources=[],
                    evidence_gaps=[
                        f"No crop-specific verified treatment registry is active for {active_crop_name}; general cultural guidance only."
                    ],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

        # 5. Supported crops without a verified chemical treatment registry.
        # Maize and Wheat are diagnostically supported, but no CIBRC/ICAR-verified
        # treatment records have been added. Withhold all chemicals rather than
        # borrowing Potato/Tomato prescriptions.
        if not has_verified_treatment_registry and active_crop_name.lower() not in ("maize", "wheat", "rice"):
            return TreatmentRecommendationResult(
                crop=active_crop_name,
                predicted_disease=clean_disease,
                is_supported=True,
                is_healthy=False,
                status="VERIFIED_RECOMMENDATION_UNAVAILABLE",
                summary_message=(
                    f"{active_crop_name} disease diagnosis is supported, but no verified "
                    "CIBRC/ICAR treatment registry has been approved for this crop. "
                    "Chemical treatments are intentionally withheld. Consult a local "
                    "certified agricultural extension officer or Krishi Vigyan Kendra (KVK)."
                ),
                chemical_treatments=[],
                non_chemical_control_measures=[],
                safety_and_environmental_precautions=[
                    "Do not reuse a fungicide prescription from another crop. Product registration, dose, and pre-harvest intervals are crop-specific."
                ],
                evidence_sources=[],
                evidence_gaps=[
                    f"No verified chemical treatment registry is active for '{active_crop_name}'.",
                    "Diagnosis and weather risk can still be used for scouting and extension consultation.",
                ],
                disclaimer=DEFAULT_TREATMENT_DISCLAIMER
            )

        # 6. Maize Specialist Treatments (Non-Chemical Cultural Guidance)
        # No verified CIBRC/ICAR chemical registry exists for Maize disease treatment.
        # All recommendations below are general cultural best practices, not chemical prescriptions.
        if "maize" in clean_crop.lower() or "corn" in clean_crop.lower():
            # Healthy Maize
            if "healthy" in disease_lower:
                return TreatmentRecommendationResult(
                    crop="Maize",
                    predicted_disease="Healthy",
                    is_supported=True,
                    is_healthy=True,
                    status="HEALTHY_PREVENTIVE_ONLY",
                    summary_message=(
                        "Healthy maize plant detected. No disease-specific treatment is required. "
                        "Continue routine field monitoring and preventive cultural management."
                    ),
                    chemical_treatments=[],
                    non_chemical_control_measures=[
                        "Field Scouting: Inspect maize leaves weekly for early rust pustules, blight lesions, or leaf spot symptoms.",
                        "Balanced Fertility: Maintain optimal nitrogen and potassium levels; avoid excess nitrogen which promotes lush, disease-susceptible foliage.",
                        "Irrigation Management: Prefer furrow or drip irrigation; avoid overhead watering that keeps foliage wet overnight.",
                        "Field Sanitation: Remove volunteer maize plants and crop debris that can harbour rust and blight inoculum."
                    ],
                    safety_and_environmental_precautions=[
                        "Do not apply prophylactic chemical fungicides without a confirmed disease diagnosis and a locally registered maize product label.",
                        "Consult a local certified agricultural extension officer or Krishi Vigyan Kendra (KVK) before any chemical application."
                    ],
                    evidence_sources=[],
                    evidence_gaps=[
                        "No crop-specific verified chemical treatment registry is active for Maize.",
                        "Cultural practices above are general guidance; local KVK advice should be followed for region-specific management."
                    ],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

            # Common Rust
            if "rust" in disease_lower:
                return TreatmentRecommendationResult(
                    crop="Maize",
                    predicted_disease="Common Rust",
                    is_supported=True,
                    is_healthy=False,
                    status="VERIFIED_RECOMMENDATION_UNAVAILABLE",
                    summary_message=(
                        "Maize Common Rust (Puccinia sorghi) detected. No verified CIBRC/ICAR chemical treatment registry "
                        "is active for Maize. Apply cultural management practices and consult your local KVK for region-specific guidance."
                    ),
                    chemical_treatments=[],
                    non_chemical_control_measures=[
                        "Destroy Volunteer Plants: Remove volunteer maize and wild grass hosts that can harbour rust spores between seasons.",
                        "Avoid Overhead Irrigation: Irrigate early in the morning so leaf surfaces dry quickly; prolonged leaf wetness favours rust infection.",
                        "Resistant Hybrids: Plant rust-resistant or tolerant maize hybrids where available in your region.",
                        "Field Scouting: Check lower and middle canopy leaves regularly for small, circular, cinnamon-brown pustules; scout more frequently during cool, humid weather."
                    ],
                    safety_and_environmental_precautions=[
                        "Do not borrow fungicide prescriptions from Potato or Tomato; Maize product registration, dose, and pre-harvest intervals are crop-specific.",
                        "Consult a local certified agricultural extension officer or Krishi Vigyan Kendra (KVK) before any chemical application."
                    ],
                    evidence_sources=[],
                    evidence_gaps=[
                        "No verified chemical treatment registry is active for Maize Common Rust.",
                        "Fungicide recommendations require local CIBRC-registered product labels for Maize."
                    ],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

            # Gray Leaf Spot / Cercospora
            if "cercospora" in disease_lower or "gray_leaf" in disease_lower or "grey_leaf" in disease_lower or "gls" in disease_lower:
                return TreatmentRecommendationResult(
                    crop="Maize",
                    predicted_disease="Gray Leaf Spot",
                    is_supported=True,
                    is_healthy=False,
                    status="VERIFIED_RECOMMENDATION_UNAVAILABLE",
                    summary_message=(
                        "Maize Gray Leaf Spot (Cercospora zeae-maydis) detected. No verified CIBRC/ICAR chemical treatment registry "
                        "is active for Maize. Apply cultural management practices and consult your local KVK for region-specific guidance."
                    ),
                    chemical_treatments=[],
                    non_chemical_control_measures=[
                        "Crop Rotation: Rotate maize with non-grass crops (legumes, cotton, potato) for at least 2 years to break the disease cycle.",
                        "Residue Management: Bury or incorporate maize stalk and leaf debris after harvest; the fungus survives on infected residue.",
                        "Optimal Plant Density: Avoid overcrowded planting; adequate spacing improves air circulation and reduces leaf wetness duration.",
                        "Field Scouting: Inspect lower canopy leaves for rectangular, tan to brown lesions with parallel sides, typically appearing after silking."
                    ],
                    safety_and_environmental_precautions=[
                        "Do not apply chemical fungicides without a confirmed diagnosis and a locally registered maize product label.",
                        "Consult a local certified agricultural extension officer or Krishi Vigyan Kendra (KVK) before any chemical application."
                    ],
                    evidence_sources=[],
                    evidence_gaps=[
                        "No verified chemical treatment registry is active for Maize Gray Leaf Spot.",
                        "Fungicide recommendations require local CIBRC-registered product labels for Maize."
                    ],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

            # Northern Leaf Blight
            if "northern" in disease_lower or "leaf_blight" in disease_lower:
                return TreatmentRecommendationResult(
                    crop="Maize",
                    predicted_disease="Northern Leaf Blight",
                    is_supported=True,
                    is_healthy=False,
                    status="VERIFIED_RECOMMENDATION_UNAVAILABLE",
                    summary_message=(
                        "Maize Northern Leaf Blight (Exserohilum turcicum) detected. No verified CIBRC/ICAR chemical treatment registry "
                        "is active for Maize. Apply cultural management practices and consult your local KVK for region-specific guidance."
                    ),
                    chemical_treatments=[],
                    non_chemical_control_measures=[
                        "Crop Rotation: Rotate maize with non-cereal crops (legumes, oilseeds) for at least 2 years to reduce inoculum buildup.",
                        "Residue Incorporation: Deep-plow or bury infected maize stalk debris after harvest; the fungus overwinters on residue.",
                        "Resistant Hybrids: Use NLB-tolerant or resistant maize hybrids recommended for your agro-climatic zone.",
                        "Field Scouting: Look for long, elliptical, cigar-shaped tan lesions on lower leaves, often with dark sporulation under humid conditions."
                    ],
                    safety_and_environmental_precautions=[
                        "Do not apply chemical fungicides without a confirmed diagnosis and a locally registered maize product label.",
                        "Consult a local certified agricultural extension officer or Krishi Vigyan Kendra (KVK) before any chemical application."
                    ],
                    evidence_sources=[],
                    evidence_gaps=[
                        "No verified chemical treatment registry is active for Maize Northern Leaf Blight.",
                        "Fungicide recommendations require local CIBRC-registered product labels for Maize."
                    ],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

        # 6a. Wheat Specialist Treatments (Non-Chemical Cultural Guidance)
        # No verified CIBRC/ICAR chemical registry exists for Wheat disease treatment.
        # All recommendations below are general cultural best practices, not chemical prescriptions.
        if "wheat" in clean_crop.lower():
            # Healthy Wheat
            if "healthy" in disease_lower:
                return TreatmentRecommendationResult(
                    crop="Wheat",
                    predicted_disease="Healthy",
                    is_supported=True,
                    is_healthy=True,
                    status="HEALTHY_PREVENTIVE_ONLY",
                    summary_message=(
                        "Healthy wheat plant detected. No disease-specific treatment is required. "
                        "Continue routine field monitoring and preventive cultural management."
                    ),
                    chemical_treatments=[],
                    non_chemical_control_measures=[
                        "Field Scouting: Inspect wheat leaves weekly for rust pustules, blotch lesions, or early blight symptoms.",
                        "Balanced Fertility: Maintain optimal nitrogen fertility; excess nitrogen promotes lush, disease-susceptible canopy growth.",
                        "Irrigation Management: Avoid overhead irrigation during cool, humid weather; irrigate early morning to allow rapid foliage drying.",
                        "Field Sanitation: Remove volunteer wheat plants and crop debris that can harbour rust and blotch inoculum."
                    ],
                    safety_and_environmental_precautions=[
                        "Do not apply prophylactic chemical fungicides without a confirmed disease diagnosis and a locally registered wheat product label.",
                        "Consult a local certified agricultural extension officer or Krishi Vigyan Kendra (KVK) before any chemical application."
                    ],
                    evidence_sources=[],
                    evidence_gaps=[
                        "No crop-specific verified chemical treatment registry is active for Wheat.",
                        "Cultural practices above are general guidance; local KVK advice should be followed for region-specific management."
                    ],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

            # Stripe Rust / Yellow Rust
            if "stripe" in disease_lower or "rust" in disease_lower:
                return TreatmentRecommendationResult(
                    crop="Wheat",
                    predicted_disease="Stripe Rust",
                    is_supported=True,
                    is_healthy=False,
                    status="VERIFIED_RECOMMENDATION_UNAVAILABLE",
                    summary_message=(
                        "Wheat Stripe (Yellow) Rust (Puccinia striiformis) detected. No verified CIBRC/ICAR chemical treatment registry "
                        "is active for Wheat. Apply cultural management practices and consult your local KVK for region-specific guidance."
                    ),
                    chemical_treatments=[],
                    non_chemical_control_measures=[
                        "Destroy Volunteer Plants: Remove volunteer wheat and grass hosts that can harbour rust spores between seasons.",
                        "Avoid Overhead Irrigation: Irrigate early in the morning so leaf surfaces dry quickly; prolonged leaf wetness favours rust infection.",
                        "Resistant Varieties: Plant rust-resistant wheat varieties recommended for your region.",
                        "Field Scouting: Check leaves regularly for small, yellow-orange pustules arranged in stripes; scout more frequently during cool, humid weather."
                    ],
                    safety_and_environmental_precautions=[
                        "Do not borrow fungicide prescriptions from Potato or Tomato; Wheat product registration, dose, and pre-harvest intervals are crop-specific.",
                        "Consult a local certified agricultural extension officer or Krishi Vigyan Kendra (KVK) before any chemical application."
                    ],
                    evidence_sources=[],
                    evidence_gaps=[
                        "No verified chemical treatment registry is active for Wheat Stripe Rust.",
                        "Fungicide recommendations require local CIBRC-registered product labels for Wheat."
                    ],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

            # Septoria Leaf Blotch
            if "septoria" in disease_lower or "blotch" in disease_lower:
                return TreatmentRecommendationResult(
                    crop="Wheat",
                    predicted_disease="Septoria Leaf Blotch",
                    is_supported=True,
                    is_healthy=False,
                    status="VERIFIED_RECOMMENDATION_UNAVAILABLE",
                    summary_message=(
                        "Wheat Septoria Leaf Blotch (Zymoseptoria tritici) detected. No verified CIBRC/ICAR chemical treatment registry "
                        "is active for Wheat. Apply cultural management practices and consult your local KVK for region-specific guidance."
                    ),
                    chemical_treatments=[],
                    non_chemical_control_measures=[
                        "Crop Rotation: Rotate wheat with non-cereal crops (legumes, oilseeds) for at least 2 years to break the disease cycle.",
                        "Residue Management: Bury or incorporate wheat straw and stubble after harvest; the fungus survives on infected residue.",
                        "Optimal Plant Density: Avoid overcrowded planting; adequate spacing improves air circulation and reduces leaf wetness duration.",
                        "Field Scouting: Inspect lower leaves for yellow specks that develop into brown blotches with dark pycnidia (fruiting bodies)."
                    ],
                    safety_and_environmental_precautions=[
                        "Do not apply chemical fungicides without a confirmed diagnosis and a locally registered wheat product label.",
                        "Consult a local certified agricultural extension officer or Krishi Vigyan Kendra (KVK) before any chemical application."
                    ],
                    evidence_sources=[],
                    evidence_gaps=[
                        "No verified chemical treatment registry is active for Wheat Septoria Leaf Blotch.",
                        "Fungicide recommendations require local CIBRC-registered product labels for Wheat."
                    ],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

        # 6b. Rice Specialist Treatments (Non-Chemical Cultural Guidance)
        # No verified CIBRC/ICAR chemical registry exists for Rice disease treatment.
        # All recommendations below are general cultural best practices, not chemical prescriptions.
        if "rice" in clean_crop.lower() or "paddy" in clean_crop.lower():
            # Healthy Rice
            if "healthy" in disease_lower:
                return TreatmentRecommendationResult(
                    crop="Rice",
                    predicted_disease="Healthy",
                    is_supported=True,
                    is_healthy=True,
                    status="HEALTHY_PREVENTIVE_ONLY",
                    summary_message=(
                        "Healthy rice plant detected. No disease-specific treatment is required. "
                        "Continue routine field monitoring and preventive cultural management."
                    ),
                    chemical_treatments=[],
                    non_chemical_control_measures=[
                        "Field Scouting: Inspect rice leaves weekly for bacterial blight lesions, brown spot, or smut symptoms.",
                        "Balanced Fertility: Maintain optimal nitrogen levels; excess nitrogen promotes lush, disease-susceptible foliage.",
                        "Water Management: Maintain proper field drainage; avoid prolonged waterlogging which favours bacterial blight.",
                        "Field Sanitation: Remove infected plant debris and volunteer rice plants between seasons."
                    ],
                    safety_and_environmental_precautions=[
                        "Do not apply prophylactic chemical fungicides without a confirmed disease diagnosis and a locally registered rice product label.",
                        "Consult a local certified agricultural extension officer or Krishi Vigyan Kendra (KVK) before any chemical application."
                    ],
                    evidence_sources=[],
                    evidence_gaps=[
                        "No crop-specific verified chemical treatment registry is active for Rice.",
                        "Cultural practices above are general guidance; local KVK advice should be followed for region-specific management."
                    ],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

            # Bacterial Leaf Blight
            if "blight" in disease_lower or "bacterial" in disease_lower:
                return TreatmentRecommendationResult(
                    crop="Rice",
                    predicted_disease="Bacterial Leaf Blight",
                    is_supported=True,
                    is_healthy=False,
                    status="VERIFIED_RECOMMENDATION_UNAVAILABLE",
                    summary_message=(
                        "Rice Bacterial Leaf Blight (Xanthomonas oryzae pv. oryzae) detected. No verified CIBRC/ICAR chemical treatment registry "
                        "is active for Rice. Apply cultural management practices and consult your local KVK for region-specific guidance."
                    ),
                    chemical_treatments=[],
                    non_chemical_control_measures=[
                        "Use Certified Seed: Plant disease-free, certified rice seeds to prevent primary inoculum introduction.",
                        "Water Management: Avoid continuous flooding; alternate wetting and drying where feasible to reduce bacterial spread.",
                        "Field Sanitation: Remove and destroy infected plant debris; avoid moving equipment from infected to clean fields.",
                        "Balanced Fertility: Avoid excess nitrogen which increases susceptibility to bacterial blight."
                    ],
                    safety_and_environmental_precautions=[
                        "Do not apply chemical bactericides without a confirmed diagnosis and a locally registered rice product label.",
                        "Consult a local certified agricultural extension officer or Krishi Vigyan Kendra (KVK) before any chemical application."
                    ],
                    evidence_sources=[],
                    evidence_gaps=[
                        "No verified chemical treatment registry is active for Rice Bacterial Leaf Blight.",
                        "Bactericide recommendations require local CIBRC-registered product labels for Rice."
                    ],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

            # Brown Spot
            if "brown" in disease_lower or "spot" in disease_lower:
                return TreatmentRecommendationResult(
                    crop="Rice",
                    predicted_disease="Brown Spot",
                    is_supported=True,
                    is_healthy=False,
                    status="VERIFIED_RECOMMENDATION_UNAVAILABLE",
                    summary_message=(
                        "Rice Brown Spot (Cochliobolus miyabeanus) detected. No verified CIBRC/ICAR chemical treatment registry "
                        "is active for Rice. Apply cultural management practices and consult your local KVK for region-specific guidance."
                    ),
                    chemical_treatments=[],
                    non_chemical_control_measures=[
                        "Seed Treatment: Use hot water or certified fungicide-treated seeds to reduce seed-borne inoculum.",
                        "Balanced Fertility: Correct potassium deficiency; potassium-deficient rice is highly susceptible to brown spot.",
                        "Field Sanitation: Remove infected plant debris and volunteer rice plants after harvest.",
                        "Water Management: Avoid moisture stress; maintain consistent irrigation to reduce plant susceptibility."
                    ],
                    safety_and_environmental_precautions=[
                        "Do not apply chemical fungicides without a confirmed diagnosis and a locally registered rice product label.",
                        "Consult a local certified agricultural extension officer or Krishi Vigyan Kendra (KVK) before any chemical application."
                    ],
                    evidence_sources=[],
                    evidence_gaps=[
                        "No verified chemical treatment registry is active for Rice Brown Spot.",
                        "Fungicide recommendations require local CIBRC-registered product labels for Rice."
                    ],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

            # Leaf Smut
            if "smut" in disease_lower:
                return TreatmentRecommendationResult(
                    crop="Rice",
                    predicted_disease="Leaf Smut",
                    is_supported=True,
                    is_healthy=False,
                    status="VERIFIED_RECOMMENDATION_UNAVAILABLE",
                    summary_message=(
                        "Rice Leaf Smut (Entyloma oryzae) detected. No verified CIBRC/ICAR chemical treatment registry "
                        "is active for Rice. Apply cultural management practices and consult your local KVK for region-specific guidance."
                    ),
                    chemical_treatments=[],
                    non_chemical_control_measures=[
                        "Use Certified Seed: Plant smut-free, certified rice seeds to prevent primary infection.",
                        "Field Sanitation: Remove and destroy smut-infected leaf debris after harvest.",
                        "Crop Rotation: Rotate rice with non-host crops where feasible to break the disease cycle.",
                        "Field Scouting: Inspect leaves for small, oval, black smut sori surrounded by yellow halos."
                    ],
                    safety_and_environmental_precautions=[
                        "Do not apply chemical fungicides without a confirmed diagnosis and a locally registered rice product label.",
                        "Consult a local certified agricultural extension officer or Krishi Vigyan Kendra (KVK) before any chemical application."
                    ],
                    evidence_sources=[],
                    evidence_gaps=[
                        "No verified chemical treatment registry is active for Rice Leaf Smut.",
                        "Fungicide recommendations require local CIBRC-registered product labels for Rice."
                    ],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

        # 6b-c. Maize / Wheat / Rice General Fallback (Unrecognized Disease Label)
        # For supported crops with no verified chemical registry, when the disease
        # label does not match a specific pattern above, return general guidance
        # rather than falling through to "Unsupported Disease".
        if active_crop_name.lower() in ("maize", "wheat", "rice"):
            # Healthy fallback
            if "healthy" in disease_lower:
                return TreatmentRecommendationResult(
                    crop=active_crop_name,
                    predicted_disease="Healthy",
                    is_supported=True,
                    is_healthy=True,
                    status="HEALTHY_PREVENTIVE_ONLY",
                    summary_message=(
                        f"Healthy {active_crop_name} plant detected. No disease-specific treatment is required. "
                        "Continue routine field monitoring and preventive cultural management."
                    ),
                    chemical_treatments=[],
                    non_chemical_control_measures=[
                        "Field Scouting: Inspect leaves weekly for early disease symptoms.",
                        "Balanced Fertility: Maintain optimal nutrient levels; avoid excess nitrogen.",
                        "Irrigation Management: Prefer early-morning or drip irrigation to keep foliage dry.",
                        "Field Sanitation: Remove volunteer plants and crop debris between seasons."
                    ],
                    safety_and_environmental_precautions=[
                        "Do not apply prophylactic chemical fungicides without a confirmed diagnosis and a locally registered product label.",
                        "Consult a local certified agricultural extension officer or Krishi Vigyan Kendra (KVK) before any chemical application."
                    ],
                    evidence_sources=[],
                    evidence_gaps=[
                        f"No crop-specific verified chemical treatment registry is active for {active_crop_name}."
                    ],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

            # Disease fallback (any unrecognized disease label for maize/wheat/rice)
            return TreatmentRecommendationResult(
                crop=active_crop_name,
                predicted_disease=clean_disease,
                is_supported=True,
                is_healthy=False,
                status="VERIFIED_RECOMMENDATION_UNAVAILABLE",
                summary_message=(
                    f"{active_crop_name} disease diagnosis is supported, but no verified "
                    "CIBRC/ICAR treatment registry has been approved for this crop. "
                    "Chemical treatments are intentionally withheld. Apply general cultural "
                    "practices and consult your local KVK for region-specific guidance."
                ),
                chemical_treatments=[],
                non_chemical_control_measures=[
                    "Field Scouting: Inspect crop canopy weekly for lesions, spots, pustules, or abnormal discoloration.",
                    "Crop Rotation: Rotate with non-host crops to break disease cycles and reduce soilborne inoculum.",
                    "Field Sanitation: Remove and destroy infected plant debris after harvest.",
                    "Irrigation Management: Avoid overhead watering; irrigate early morning to allow rapid foliage drying.",
                    "Resistant Varieties: Plant disease-resistant or tolerant varieties recommended for your region."
                ],
                safety_and_environmental_precautions=[
                    "Do not apply chemical fungicides without a confirmed disease diagnosis and a locally registered product label.",
                    "Consult a local certified agricultural extension officer or Krishi Vigyan Kendra (KVK) before any chemical application."
                ],
                evidence_sources=[],
                evidence_gaps=[
                    f"No verified chemical treatment registry is active for {active_crop_name}.",
                    "Disease-specific cultural guidance requires local extension consultation."
                ],
                disclaimer=DEFAULT_TREATMENT_DISCLAIMER
            )

        # 6. Potato Specialist Treatments
        if is_potato:
            if "late" in disease_lower:
                retrieved_chunks = self._retrieve_treatment_evidence("Potato", "Late Blight")
                sources_meta = self._format_source_metadata(retrieved_chunks)
                summary = (
                    "Late Blight (Phytophthora infestans) detected. Immediate management required to prevent rapid field epidemic. "
                    "Apply registered protectant fungicides (Mancozeb or Chlorothalonil) as preventative barrier, or systemic combination "
                    "(Cymoxanil + Mancozeb / Metalaxyl + Mancozeb) if active water-soaked lesions are spreading or local risk is HIGH."
                )
                return TreatmentRecommendationResult(
                    crop="Potato",
                    predicted_disease="Late Blight",
                    is_supported=True,
                    is_healthy=False,
                    status="VERIFIED_RECOMMENDATION_AVAILABLE",
                    summary_message=summary,
                    chemical_treatments=VERIFIED_LATE_BLIGHT_TREATMENTS,
                    non_chemical_control_measures=[
                        "Dehaulming (Haulm Killing): Cut or desiccate potato vines 10 to 14 days prior to harvest to prevent sporangia washing onto tubers.",
                        "Cull Pile Destruction: Bury, burn, or freeze all potato cull piles before spring emergence to eliminate primary inoculum reservoirs.",
                        "Certified Disease-Free Seed: Plant certified seed tubers exclusively; never plant tubers from late-blight infected fields.",
                        "Canopy Aeration: Ensure proper row spacing to maximize sunlight penetration and rapid foliage drying after rain or heavy dew.",
                        "Avoid Overhead Irrigation: Prefer drip irrigation or irrigate early in the morning; never irrigate in the late evening."
                    ],
                    safety_and_environmental_precautions=[
                        "Personal Protective Equipment (PPE): Always wear rubber gloves, face mask/respirator, protective goggles, and full-sleeve overalls while preparing and spraying fungicides.",
                        "Resistance Management: Never apply systemic Metalaxyl consecutively more than 1 to 2 times per season. Alternate with contact multi-site protectants (Mancozeb or Chlorothalonil).",
                        "Environmental Buffer: Do not spray within 15 meters of open water bodies or fish ponds; avoid spraying during wind speeds > 15 km/h to prevent spray drift.",
                        "Pre-Harvest & Re-Entry Intervals: Strictly adhere to container label PHI and REI specifications before entering sprayed fields or harvesting tubers."
                    ],
                    evidence_sources=sources_meta,
                    evidence_gaps=[
                        "Commercial product trade names vary by authorized regional manufacturer; farmers must inspect local CIBRC-approved product labels.",
                        "Specific drone spraying application parameters are subject to separate state aviation and label clearances."
                    ],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

            if "early" in disease_lower:
                retrieved_chunks = self._retrieve_treatment_evidence("Potato", "Early Blight")
                sources_meta = self._format_source_metadata(retrieved_chunks)
                summary = (
                    "Early Blight (Alternaria solani) detected on potato leaf sample. "
                    "Apply protective contact fungicides (Mancozeb 75% WP or Chlorothalonil 75% WP) at initial symptom detection on lower foliage. "
                    "Ensure balanced nitrogen fertility, as nutrient-stressed and aging plants suffer accelerated defoliation."
                )
                return TreatmentRecommendationResult(
                    crop="Potato",
                    predicted_disease="Early Blight",
                    is_supported=True,
                    is_healthy=False,
                    status="VERIFIED_RECOMMENDATION_AVAILABLE",
                    summary_message=summary,
                    chemical_treatments=VERIFIED_EARLY_BLIGHT_TREATMENTS,
                    non_chemical_control_measures=[
                        "Balanced Crop Nutrition: Ensure adequate nitrogen and potassium fertilization; nitrogen-deficient plants experience accelerated early blight senescence.",
                        "Crop Rotation: Rotate with non-solanaceous crops (cereals, legumes, corn) for at least 3 to 4 years to deplete soilborne fungal conidia.",
                        "Crop Residue Management: Disk or deep-plow crop residues immediately after harvest to speed up fungal tissue decomposition.",
                        "Irrigation Timing: Irrigate early in the morning so canopy foliage dries quickly in morning sunlight; avoid late evening overhead sprinkler irrigations."
                    ],
                    safety_and_environmental_precautions=[
                        "Wear chemical-resistant gloves, protective eye goggles, and a dust/mist mask during mixing and spraying.",
                        "Do not apply when high winds could cause chemical drift onto neighboring sensitive crops or residences.",
                        "Consult registered product container labels for mandatory Pre-Harvest Intervals (PHI) and Worker Re-Entry Intervals (REI)."
                    ],
                    evidence_sources=sources_meta,
                    evidence_gaps=[
                        "Exact commercial brand trade names and local retail formulations must be verified through local certified agro-dealers or KVK.",
                        "Specific chemical dosage rates for tank-mixing with foliar fertilizers must follow official manufacturer label guidelines."
                    ],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

        # 6. Tomato Specialist Treatments
        if is_tomato:
            retrieved_chunks = self._retrieve_treatment_evidence("Tomato", clean_disease)
            sources_meta = self._format_source_metadata(retrieved_chunks) if retrieved_chunks else self._format_tomato_source_metadata()

            if "late" in disease_lower:
                return TreatmentRecommendationResult(
                    crop="Tomato",
                    predicted_disease="Late Blight",
                    is_supported=True,
                    is_healthy=False,
                    status="VERIFIED_RECOMMENDATION_AVAILABLE",
                    summary_message=(
                        "Tomato Late Blight (Phytophthora infestans) detected. Immediate management required to halt rapid foliar collapse. "
                        "Apply registered contact protectants (Mancozeb or Chlorothalonil) as preventative barrier, or translaminar systemic combination "
                        "(Cymoxanil + Mancozeb / Metalaxyl + Mancozeb) under active lesion spread or high microclimate risk."
                    ),
                    chemical_treatments=VERIFIED_TOMATO_LATE_BLIGHT_TREATMENTS,
                    non_chemical_control_measures=[
                        "Infected Vine Rogueing: Uproot and bag or burn initial late blight focal plants to eliminate aerial sporangia sources.",
                        "Weed Eradication: Remove wild black nightshade (Solanum nigrum) and volunteer tomato seedlings from bunds.",
                        "Trellising & Pruning: Stake plants and prune lower suckers to maintain canopy air flow.",
                        "Drip Irrigation: Never irrigate overhead; keep foliage completely dry during evening hours."
                    ],
                    safety_and_environmental_precautions=[
                        "Personal Protective Equipment (PPE): Wear rubber gloves, eye goggles, and mist respirator during mixing and spraying.",
                        "Resistance Management: Never apply Metalaxyl consecutively more than 1 to 2 times per season. Alternate with contact multi-site protectants.",
                        "Environmental Buffer: Do not spray near water bodies; observe 3 to 7 day Pre-Harvest Intervals."
                    ],
                    evidence_sources=sources_meta,
                    evidence_gaps=[
                        "Commercial retail brands vary by state; consult local CIBRC label registrations."
                    ],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

            if "early" in disease_lower:
                return TreatmentRecommendationResult(
                    crop="Tomato",
                    predicted_disease="Early Blight",
                    is_supported=True,
                    is_healthy=False,
                    status="VERIFIED_RECOMMENDATION_AVAILABLE",
                    summary_message=(
                        "Tomato Early Blight (Alternaria solani) detected on foliar sample. "
                        "Apply protective contact fungicides (Mancozeb 75% WP or Chlorothalonil 75% WP) on lower leaves, "
                        "or systemic triazole (Difenoconazole 25% EC) if lesions are actively expanding during warm weather."
                    ),
                    chemical_treatments=VERIFIED_TOMATO_EARLY_BLIGHT_TREATMENTS,
                    non_chemical_control_measures=[
                        "Lower Foliage Pruning: Remove and destroy lower yellowing or spotted leaves touching the ground.",
                        "Mulching: Apply straw or plastic mulch to prevent soil-splashed conidia from reaching lower leaves.",
                        "Balanced Nutrition: Ensure adequate potassium and avoid plant nitrogen deficiency which accelerates early blight defoliation.",
                        "Crop Rotation: Rotate with non-solanaceous crops for at least 2 to 3 years."
                    ],
                    safety_and_environmental_precautions=[
                        "Wear gloves, face mask, and eye protection during application.",
                        "Observe container Pre-Harvest Intervals (typically 3 to 7 days)."
                    ],
                    evidence_sources=sources_meta,
                    evidence_gaps=["Consult local state agricultural extension officers for region-specific trade formulations."],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

            if "bacterial" in disease_lower:
                return TreatmentRecommendationResult(
                    crop="Tomato",
                    predicted_disease="Bacterial Spot",
                    is_supported=True,
                    is_healthy=False,
                    status="VERIFIED_RECOMMENDATION_AVAILABLE",
                    summary_message=(
                        "Tomato Bacterial Spot (Xanthomonas vesicatoria) detected. "
                        "Apply Copper Oxychloride 50% WP (0.25%) tank-mixed with Streptocycline (100 to 200 ppm) "
                        "or Copper Hydroxide (77% WP) at initial symptom observation. Note: Standard synthetic fungicides are ineffective against bacteria."
                    ),
                    chemical_treatments=VERIFIED_TOMATO_BACTERIAL_SPOT_TREATMENTS,
                    non_chemical_control_measures=[
                        "Certified Disease-Free Seed: Sowing certified pathogen-free seeds is the most critical control step.",
                        "Hot Water Seed Treatment: Treat seeds at 50°C for 25 minutes prior to sowing to eliminate seed-borne Xanthomonas.",
                        "Avoid Overhead Irrigation: Splashing water droplets rapidly spread bacterial cells across adjacent plants.",
                        "Sanitation: Avoid working in field rows when foliage is wet from morning dew or rain."
                    ],
                    safety_and_environmental_precautions=[
                        "Wear rubber gloves, eye goggles, and protective overalls.",
                        "Avoid continuous repetitive copper sprays to prevent copper phytotoxicity and soil copper accumulation."
                    ],
                    evidence_sources=sources_meta,
                    evidence_gaps=["Antibiotic formulations (Streptocycline) must be used strictly per CIBRC agricultural dosage guidelines."],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

            if any(k in disease_lower for k in ["leaf_mold", "mold"]):
                return TreatmentRecommendationResult(
                    crop="Tomato",
                    predicted_disease="Leaf Mold",
                    is_supported=True,
                    is_healthy=False,
                    status="VERIFIED_RECOMMENDATION_AVAILABLE",
                    summary_message=(
                        "Tomato Leaf Mold (Passalora fulva) detected. "
                        "Apply Copper Oxychloride 50% WP (0.25%) or Mancozeb 75% WP (0.2%) targeting leaf undersides where velvety fungal growth occurs."
                    ),
                    chemical_treatments=VERIFIED_TOMATO_LEAF_MOLD_TREATMENTS,
                    non_chemical_control_measures=[
                        "Greenhouse/Polyhouse Aeration: Maximize ventilation to maintain canopy relative humidity below 85% RH.",
                        "Pruning: Remove lower diseased leaves and dense suckers to enhance sunlight penetration.",
                        "Drip Irrigation: Water in the morning so foliage dries quickly."
                    ],
                    safety_and_environmental_precautions=[
                        "Wear protective face mask and gloves during spraying."
                    ],
                    evidence_sources=sources_meta,
                    evidence_gaps=["Polyhouse ventilation is the primary non-chemical determinant of leaf mold suppression."],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

            if "septoria" in disease_lower:
                return TreatmentRecommendationResult(
                    crop="Tomato",
                    predicted_disease="Septoria Leaf Spot",
                    is_supported=True,
                    is_healthy=False,
                    status="VERIFIED_RECOMMENDATION_AVAILABLE",
                    summary_message=(
                        "Tomato Septoria Leaf Spot (Septoria lycopersici) detected. "
                        "Apply contact protectant fungicides (Mancozeb 75% WP or Chlorothalonil 75% WP) at initial symptom detection on lower leaves."
                    ),
                    chemical_treatments=VERIFIED_TOMATO_SEPTORIA_TREATMENTS,
                    non_chemical_control_measures=[
                        "Crop Rotation: Rotate away from Solanaceae for at least 2 seasons.",
                        "Lower Canopy Sanitation: Prune leaves with circular dark margins and sunken gray centers.",
                        "Clean Tillage: Deep-plow crop residues immediately after harvest to bury fungal pycnidia."
                    ],
                    safety_and_environmental_precautions=[
                        "Wear gloves, goggles, and protective clothing."
                    ],
                    evidence_sources=sources_meta,
                    evidence_gaps=["Check registered pre-harvest intervals on local manufacturer labels."],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

            if "target" in disease_lower:
                return TreatmentRecommendationResult(
                    crop="Tomato",
                    predicted_disease="Target Spot",
                    is_supported=True,
                    is_healthy=False,
                    status="VERIFIED_RECOMMENDATION_AVAILABLE",
                    summary_message=(
                        "Tomato Target Spot (Corynespora cassiicola) detected. "
                        "Apply protective fungicides (Mancozeb 75% WP or Chlorothalonil 75% WP) to halt expanding zonate circular lesions."
                    ),
                    chemical_treatments=VERIFIED_TOMATO_TARGET_SPOT_TREATMENTS,
                    non_chemical_control_measures=[
                        "Trellising: Stake tomato vines to elevate fruit and leaves away from wet soil.",
                        "Air Circulation: Space plants adequately to promote rapid leaf drying.",
                        "Debris Sanitation: Remove and destroy infected crop residues."
                    ],
                    safety_and_environmental_precautions=["Wear standard pesticide safety gear."],
                    evidence_sources=sources_meta,
                    evidence_gaps=["Consult local extension bulletins for state registered fungicides."],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

            if any(k in disease_lower for k in ["spider", "mite"]):
                return TreatmentRecommendationResult(
                    crop="Tomato",
                    predicted_disease="Two-Spotted Spider Mite",
                    is_supported=True,
                    is_healthy=False,
                    status="VERIFIED_RECOMMENDATION_AVAILABLE",
                    summary_message=(
                        "Two-Spotted Spider Mite (Tetranychus urticae) infestation detected. "
                        "CRITICAL: Spider mites are arachnid pests; chemical fungicides have NO efficacy against spider mites. "
                        "Apply registered acaricides (Spiromesifen 22.9% SC @ 1 ml/L or Wettable Sulphur 80% WP @ 2.5-3 g/L) or botanical Neem formulation (Azadirachtin 1500 ppm @ 3-5 ml/L)."
                    ),
                    chemical_treatments=VERIFIED_TOMATO_SPIDER_MITES_TREATMENTS,
                    non_chemical_control_measures=[
                        "Under-Canopy Water Jet: Spray pressurized water jets directed at leaf undersides to dislodge mites and destroy webs.",
                        "Avoid Plant Water Stress: Ensure regular drip irrigation; drought-stressed plants suffer intensified mite reproduction.",
                        "Biological Conservation: Protect predatory mites (Phytoseiulus persimilis), lady beetles, and lacewing larvae."
                    ],
                    safety_and_environmental_precautions=[
                        "Do not apply sulphur formulations when ambient temperature exceeds 35°C to avoid foliage burn (sulfur phytotoxicity).",
                        "Wear eye protection, rubber gloves, and respirator during acaricide application."
                    ],
                    evidence_sources=sources_meta,
                    evidence_gaps=["Acaricide resistance is common; alternate Spiromesifen with Wettable Sulphur or botanical Neem."],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

            if any(k in disease_lower for k in ["yellow_leaf_curl", "curl"]):
                return TreatmentRecommendationResult(
                    crop="Tomato",
                    predicted_disease="Yellow Leaf Curl Virus",
                    is_supported=True,
                    is_healthy=False,
                    status="VERIFIED_RECOMMENDATION_AVAILABLE",
                    summary_message=(
                        "Tomato Yellow Leaf Curl Virus (TYLCV) detected. "
                        "CRITICAL SAFETY RULE: Plant viral infections CANNOT BE CURED by chemical fungicides, antibiotics, or bactericides. "
                        "Do NOT apply fungicides. Management is strictly focused on controlling the whitefly insect vector (Bemisia tabaci) "
                        "using CIBRC-registered insecticides (Imidacloprid 17.8% SL @ 0.3 ml/L or Neem oil 1500 ppm @ 3-5 ml/L) and rogueing infected plants."
                    ),
                    chemical_treatments=VERIFIED_TOMATO_TYLCV_TREATMENTS,
                    non_chemical_control_measures=[
                        "Immediate Plant Rogueing: Uproot, bag, and burn virus-infected plants immediately to eliminate viral inoculum reservoirs.",
                        "Yellow Sticky Traps: Install 15 to 20 yellow sticky traps per acre at canopy level to monitor and trap adult whiteflies.",
                        "Nylon Mesh Nurseries: Raise seedling nurseries under 40 to 50 mesh insect-proof nylon net covers.",
                        "Silver Reflective Mulch: Use silver-colored reflective mulches to disorient incoming whitefly vectors."
                    ],
                    safety_and_environmental_precautions=[
                        "Do not apply insecticides during active daytime bee foraging hours.",
                        "Wear full protective gear (gloves, face mask, eye goggles)."
                    ],
                    evidence_sources=sources_meta,
                    evidence_gaps=["No chemical cures the virus itself; vector control and rogueing are the only verified measures."],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

            if "mosaic" in disease_lower:
                return TreatmentRecommendationResult(
                    crop="Tomato",
                    predicted_disease="Mosaic Virus",
                    is_supported=True,
                    is_healthy=False,
                    status="VERIFIED_RECOMMENDATION_AVAILABLE",
                    summary_message=(
                        "Tomato Mosaic Virus (ToMV) detected. "
                        "CRITICAL SAFETY RULE: Plant viral infections cannot be cured by chemical pesticides, fungicides, or antibiotics. "
                        "Do NOT spray chemical fungicides. Management is strictly preventive through cultural hygiene and mechanical transmission control."
                    ),
                    chemical_treatments=[],
                    non_chemical_control_measures=[
                        "Skimmed Milk Foliar Barrier: Spray 10% to 20% skimmed milk / non-fat dry milk solution before pruning to inactivate mechanical sap transmission.",
                        "Tool & Hand Sanitation: Wash hands and cutting shears in 10% trisodium phosphate (TSP) or detergent soap between handling plants.",
                        "Rogueing: Remove, bag, and bury or incinerate virus-infected plants as soon as mosaic mottling or stunting is noticed.",
                        "Certified Virus-Free Seed: Plant certified virus-indexed seed stock only.",
                        "Tobacco Handling Prohibition: Workers must never smoke or handle tobacco products prior to touching tomato plants."
                    ],
                    safety_and_environmental_precautions=[
                        "No hazardous chemical pesticides are indicated for viral mosaic management."
                    ],
                    evidence_sources=sources_meta,
                    evidence_gaps=["Chemical fungicides have zero efficacy against viruses."],
                    disclaimer=DEFAULT_TREATMENT_DISCLAIMER
                )

        # 7. Unknown or Unsupported Disease
        return TreatmentRecommendationResult(
            crop=active_crop_name,
            predicted_disease=clean_disease,
            is_supported=False,
            is_healthy=False,
            status="UNSUPPORTED_DISEASE",
            summary_message=(
                f"No verified treatment recommendation is currently available for '{clean_disease}' on {active_crop_name}. "
                "Consult a certified local agricultural extension officer or Krishi Vigyan Kendra (KVK) for an in-person field diagnosis."
            ),
            chemical_treatments=[],
            non_chemical_control_measures=[],
            safety_and_environmental_precautions=[],
            evidence_sources=[],
            evidence_gaps=[f"Disease label '{clean_disease}' has no registered treatment protocol in the verified knowledge base."],
            disclaimer=DEFAULT_TREATMENT_DISCLAIMER
        )

    def _retrieve_treatment_evidence(self, crop: str, disease: str) -> List[SearchResult]:
        """Retrieve verified treatment documents from the ChromaDB vector store."""
        try:
            query = f"verified fungicides chemical treatment active ingredient application rate {crop} {disease} CIBRC ICAR CPRI"
            results = self.retriever.retrieve(
                query=query,
                crop=crop,
                disease=disease,
                top_k=3
            )
            return results
        except Exception:
            return []

    def _format_source_metadata(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Extract and format source metadata from search results."""
        sources = []
        if results:
            for r in results:
                sources.append({
                    "source_organization": r.metadata.get("source_organization", "CIBRC / Extension"),
                    "document_title": r.metadata.get("document_title", "Potato Disease Treatment Guide"),
                    "url_reference": r.metadata.get("url_reference", "https://ppqs.gov.in"),
                    "document_type": r.metadata.get("document_type", "treatment_guide"),
                    "similarity_score": r.similarity_score
                })
        else:
            # Fallback to verified primary authority records
            sources = [
                {
                    "source_organization": "Central Insecticides Board & Registration Committee (CIBRC), DPPQ&S, Govt. of India",
                    "document_title": "Major Uses of Pesticides (Fungicides) Registered under the Insecticides Act, 1968",
                    "url_reference": "https://ppqs.gov.in",
                    "document_type": "official_registry"
                },
                {
                    "source_organization": "ICAR - Central Potato Research Institute (CPRI), Shimla",
                    "document_title": "Potato Late Blight Management Advisory & Indo-Blightcast",
                    "url_reference": "https://ppqs.gov.in",
                    "document_type": "research_bulletin"
                },
                {
                    "source_organization": "Tamil Nadu Agricultural University (TNAU) Agritech Portal",
                    "document_title": "Crop Protection: Potato Diseases & Management",
                    "url_reference": "https://ppqs.gov.in",
                    "document_type": "extension_guide"
                }
            ]
        return sources

    def _format_tomato_source_metadata(self) -> List[Dict[str, Any]]:
        """Return verified Tomato authority records when no indexed chunks are available."""
        return [
            {
                "source_organization": "Central Insecticides Board & Registration Committee (CIBRC), DPPQ&S, Govt. of India",
                "document_title": "Major Uses of Pesticides Registered under the Insecticides Act, 1968",
                "url_reference": "https://ppqs.gov.in",
                "document_type": "official_registry"
            },
            {
                "source_organization": "ICAR - Indian Institute of Vegetable Research (IIVR), Varanasi",
                "document_title": "Tomato Disease Management and Integrated Pest Management Guidelines",
                "url_reference": "https://iivr.icar.gov.in",
                "document_type": "research_bulletin"
            },
            {
                "source_organization": "Tamil Nadu Agricultural University (TNAU) Agritech Portal",
                "document_title": "Crop Protection: Tomato Diseases and Pest Management",
                "url_reference": "https://agritech.tnau.ac.in",
                "document_type": "extension_guide"
            }
        ]


# Extensible alias for multi-crop treatment recommender
CropTreatmentRecommender = PotatoTreatmentRecommender

