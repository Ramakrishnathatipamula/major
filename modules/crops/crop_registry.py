import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@dataclass
class CropSpec:
    """
    Metadata specification for a registered crop in AI Crop Doctor.
    Represents species identity, supported disease/healthy classes,
    specialist model availability, risk model compatibility, and trusted evidence sources.
    """
    crop_id: str
    display_name: str
    is_supported: bool = False
    support_status_message: str = ""
    supported_diseases: List[str] = field(default_factory=list)
    disease_friendly_names: Dict[str, str] = field(default_factory=dict)
    healthy_class: Optional[str] = None
    model_path: Optional[str] = None
    compatible_risk_models: List[str] = field(default_factory=list)
    trusted_knowledge_sources: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def diagnostic_supported(self) -> bool:
        """Diagnostic support status (True if active specialist model is integrated)."""
        return self.is_supported

    @property
    def disease_classes(self) -> List[str]:
        """List of supported disease classes."""
        return self.supported_diseases

    @property
    def model_file_path(self) -> Optional[Path]:
        """Return absolute Path to the specialist model checkpoint file if configured."""
        if not self.model_path:
            return None
        return ROOT_DIR / self.model_path

    def is_model_available(self, base_dir: Optional[Path] = None) -> bool:
        """Check if the specialist model checkpoint file actually exists on disk."""
        if not self.model_path:
            return False
        root = base_dir if base_dir is not None else ROOT_DIR
        return (root / self.model_path).exists()

    def has_disease(self, disease_name: str) -> bool:
        """Check if disease is supported for this crop (case-insensitive slug or friendly name)."""
        if not disease_name:
            return False
        d_lower = str(disease_name).strip().lower()
        if self.healthy_class and (self.healthy_class.lower() == d_lower or "healthy" in d_lower):
            return True
        for d in self.supported_diseases:
            supported_lower = d.lower()
            if supported_lower == d_lower or supported_lower.split("___", 1)[-1] == d_lower:
                return True
        for k, v in self.disease_friendly_names.items():
            if k.lower() == d_lower or v.lower() == d_lower:
                return True
            if d_lower in v.lower():
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["diagnostic_supported"] = self.is_supported
        data["disease_classes"] = self.supported_diseases
        data["is_model_available"] = self.is_model_available()
        return data


class CropRegistry:
    """
    Central registry for crops, supported disease models, risk compatibility, and evidence sources.
    Enables multi-crop architecture without hardcoding crop names as global constants.
    """

    def __init__(self):
        self._crops: Dict[str, CropSpec] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register initial set of target crops with full foundation metadata."""
        # Potato: Currently active development and validation crop
        self.register_crop(
            CropSpec(
                crop_id="potato",
                display_name="Potato",
                is_supported=True,
                support_status_message="Full ML diagnostic model and verified CIBRC/ICAR knowledge base active.",
                supported_diseases=[
                    "Potato___Early_blight",
                    "Potato___Late_blight",
                    "Potato___healthy"
                ],
                disease_friendly_names={
                    "Potato___Early_blight": "Early Blight",
                    "Potato___Late_blight": "Late Blight",
                    "Potato___healthy": "Healthy"
                },
                healthy_class="Potato___healthy",
                model_path="models/disease/potato_disease_model.pth",
                compatible_risk_models=["wallin_dsv", "p_day", "baseline_weather"],
                trusted_knowledge_sources=[
                    {
                        "organization": "CIBRC",
                        "title": "Central Insecticides Board & Registration Committee Registered Fungicides",
                        "url": "https://ppqs.gov.in"
                    },
                    {
                        "organization": "ICAR-CPRI",
                        "title": "ICAR-Central Potato Research Institute Disease Protocols",
                        "url": "https://cpri.icar.gov.in"
                    },
                    {
                        "organization": "TNAU Agritech",
                        "title": "Tamil Nadu Agricultural University Potato Crop Protection",
                        "url": "https://agritech.tnau.ac.in"
                    },
                    {
                        "organization": "UMN Extension",
                        "title": "University of Minnesota Extension Potato Pathology",
                        "url": "https://extension.umn.edu"
                    }
                ]
            )
        )

        # Tomato: Priority Solanaceae candidate (shares blight pathogens with Potato)
        tomato_model_path = "models/disease/tomato_disease_model.pth"
        tomato_model_available = (ROOT_DIR / tomato_model_path).exists()
        tomato_support_msg = (
            "Full ML diagnostic specialist model (10 classes) and verified CIBRC/ICAR-IIVR knowledge base active."
            if tomato_model_available
            else "Tomato disease detection model checkpoint (models/disease/tomato_disease_model.pth) is missing. Re-train or restore model checkpoint."
        )

        self.register_crop(
            CropSpec(
                crop_id="tomato",
                display_name="Tomato",
                is_supported=tomato_model_available,
                support_status_message=tomato_support_msg,
                supported_diseases=[
                    "Tomato___Bacterial_spot",
                    "Tomato___Early_blight",
                    "Tomato___Late_blight",
                    "Tomato___Leaf_Mold",
                    "Tomato___Septoria_leaf_spot",
                    "Tomato___Spider_mites Two-spotted_spider_mite",
                    "Tomato___Target_Spot",
                    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
                    "Tomato___Tomato_mosaic_virus",
                    "Tomato___healthy"
                ],
                disease_friendly_names={
                    "Tomato___Bacterial_spot": "Bacterial Spot",
                    "Tomato___Early_blight": "Early Blight",
                    "Tomato___Late_blight": "Late Blight",
                    "Tomato___Leaf_Mold": "Leaf Mold",
                    "Tomato___Septoria_leaf_spot": "Septoria Leaf Spot",
                    "Tomato___Spider_mites Two-spotted_spider_mite": "Spider Mites",
                    "Tomato___Target_Spot": "Target Spot",
                    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Yellow Leaf Curl Virus",
                    "Tomato___Tomato_mosaic_virus": "Mosaic Virus",
                    "Tomato___healthy": "Healthy"
                },
                healthy_class="Tomato___healthy",
                model_path=tomato_model_path,
                compatible_risk_models=["wallin_dsv", "p_day", "baseline_weather"],
                trusted_knowledge_sources=[
                    {
                        "organization": "ICAR-IIVR",
                        "title": "ICAR - Indian Institute of Vegetable Research Tomato Pathology",
                        "url": "https://iivr.icar.gov.in"
                    },
                    {
                        "organization": "TNAU Agritech",
                        "title": "TNAU Agritech Portal Tomato Pest & Disease Management",
                        "url": "https://agritech.tnau.ac.in"
                    },
                    {
                        "organization": "CIBRC",
                        "title": "Central Insecticides Board & Registration Committee Registered Tomato Pesticides",
                        "url": "https://ppqs.gov.in"
                    }
                ]
            )
        )

        # Maize / Corn: specialist trained from cached PlantVillage Corn classes
        maize_model_path = "models/disease/maize_disease_model.pth"
        maize_model_available = (ROOT_DIR / maize_model_path).exists()
        maize_support_msg = (
            "Full ML diagnostic specialist model (4 classes) active. No verified chemical treatment registry is active for Maize, so chemical advice is withheld."
            if maize_model_available
            else "Maize disease detection model checkpoint (models/disease/maize_disease_model.pth) is missing. Re-train or restore the checkpoint."
        )
        self.register_crop(
            CropSpec(
                crop_id="maize",
                display_name="Maize",
                is_supported=maize_model_available,
                support_status_message=maize_support_msg,
                supported_diseases=[
                    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
                    "Corn_(maize)___Common_rust_",
                    "Corn_(maize)___Northern_Leaf_Blight",
                    "Corn_(maize)___healthy"
                ],
                disease_friendly_names={
                    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Gray Leaf Spot",
                    "Corn_(maize)___Common_rust_": "Common Rust",
                    "Corn_(maize)___Northern_Leaf_Blight": "Northern Leaf Blight",
                    "Corn_(maize)___healthy": "Healthy"
                },
                healthy_class="Corn_(maize)___healthy",
                model_path=maize_model_path,
                compatible_risk_models=["cereal_baseline"],
                trusted_knowledge_sources=[
                    {
                        "organization": "ICAR-IIMR",
                        "title": "ICAR - Indian Institute of Maize Research Pathology",
                        "url": "https://iimr.icar.gov.in"
                    },
                    {
                        "organization": "CIMMYT",
                        "title": "International Maize and Wheat Improvement Center — Maize Disease Resources",
                        "url": "https://www.cimmyt.org"
                    }
                ]
            )
        )

        # Rice: Critical Staple Cereal candidate
        rice_model_path = "models/disease/rice_disease_model.pth"
        rice_model_available = (ROOT_DIR / rice_model_path).exists()
        rice_support_msg = (
            "Full ML diagnostic specialist model (4 classes) active. No verified chemical treatment registry is active for Rice, so chemical advice is withheld."
            if rice_model_available
            else "Rice disease detection model checkpoint (models/disease/rice_disease_model.pth) is missing. Re-train or restore the checkpoint."
        )
        self.register_crop(
            CropSpec(
                crop_id="rice",
                display_name="Rice",
                is_supported=rice_model_available,
                support_status_message=rice_support_msg,
                supported_diseases=[
                    "Rice___Bacterial_leaf_blotch",
                    "Rice___Brown_spot",
                    "Rice___Leaf_smut",
                    "Rice___healthy"
                ],
                disease_friendly_names={
                    "Rice___Bacterial_leaf_blotch": "Bacterial Leaf Blight",
                    "Rice___Brown_spot": "Brown Spot",
                    "Rice___Leaf_smut": "Leaf Smut",
                    "Rice___healthy": "Healthy"
                },
                healthy_class="Rice___healthy",
                model_path=rice_model_path,
                compatible_risk_models=["cereal_baseline"],
                trusted_knowledge_sources=[
                    {
                        "organization": "ICAR-NRRI",
                        "title": "ICAR - National Rice Research Institute Crop Protection",
                        "url": "https://icar-nrri.in"
                    }
                ]
            )
        )

        # Wheat: specialist trained from a public field dataset; no PlantVillage class exists
        wheat_model_path = "models/disease/wheat_disease_model.pth"
        wheat_model_available = (ROOT_DIR / wheat_model_path).exists()
        wheat_support_msg = (
            "Full ML diagnostic specialist model (3 classes) active. No verified chemical treatment registry is active for Wheat, so chemical advice is withheld."
            if wheat_model_available
            else "Wheat disease detection model checkpoint (models/disease/wheat_disease_model.pth) is missing. Re-train or restore the checkpoint."
        )
        self.register_crop(
            CropSpec(
                crop_id="wheat",
                display_name="Wheat",
                is_supported=wheat_model_available,
                support_status_message=wheat_support_msg,
                supported_diseases=[
                    "Wheat___Septoria_leaf_blotch",
                    "Wheat___Stripe_rust",
                    "Wheat___healthy"
                ],
                disease_friendly_names={
                    "Wheat___Septoria_leaf_blotch": "Septoria Leaf Blotch",
                    "Wheat___Stripe_rust": "Stripe Rust (Yellow Rust)",
                    "Wheat___healthy": "Healthy"
                },
                healthy_class="Wheat___healthy",
                model_path=wheat_model_path,
                compatible_risk_models=["cereal_baseline"],
                trusted_knowledge_sources=[
                    {
                        "organization": "ICAR-IIWBR",
                        "title": "ICAR - Indian Institute of Wheat and Barley Research",
                        "url": "https://iiwbr.icar.gov.in"
                    },
                    {
                        "organization": "CIMMYT",
                        "title": "International Maize and Wheat Improvement Center — Wheat Disease Resources",
                        "url": "https://www.cimmyt.org"
                    }
                ]
            )
        )

        # Candidate crops without PlantVillage coverage (ICAR research institutions)
        candidate_sources = {
            "cotton": [
                {
                    "organization": "ICAR-CICR",
                    "title": "ICAR - Central Institute for Cotton Research",
                    "url": "https://cicr.icar.gov.in"
                }
            ],
            "mango": [
                {
                    "organization": "ICAR-CISH",
                    "title": "ICAR - Central Institute for Subtropical Horticulture (Mango Research)",
                    "url": "https://cish.icar.gov.in"
                }
            ],
            "banana": [
                {
                    "organization": "ICAR-NRCB",
                    "title": "ICAR - National Research Centre for Banana",
                    "url": "https://nrcb.icar.gov.in"
                }
            ]
        }

        for crop_id, display_name in [("cotton", "Cotton"), ("mango", "Mango"), ("banana", "Banana")]:
            self.register_crop(
                CropSpec(
                    crop_id=crop_id,
                    display_name=display_name,
                    is_supported=False,
                    support_status_message=f"Disease detection model and verified extension knowledge for {display_name} are pending field dataset sourcing (outside PlantVillage). Only crops with active specialist checkpoints are supported.",
                    compatible_risk_models=["baseline_weather"],
                    trusted_knowledge_sources=candidate_sources.get(crop_id, [])
                )
            )

        # PlantVillage candidate species recognized by Stage 1 Plant Identifier
        pv_candidate_crops = [
            ("apple", "Apple"),
            ("blueberry", "Blueberry"),
            ("cherry", "Cherry"),
            ("grape", "Grape"),
            ("orange", "Orange"),
            ("peach", "Peach"),
            ("bell_pepper", "Pepper, Bell"),
            ("raspberry", "Raspberry"),
            ("soybean", "Soybean"),
            ("squash", "Squash"),
            ("strawberry", "Strawberry"),
        ]
        for crop_id, display_name in pv_candidate_crops:
            self.register_crop(
                CropSpec(
                    crop_id=crop_id,
                    display_name=display_name,
                    is_supported=False,
                    support_status_message=f"Disease detection specialist for {display_name} is pending training/calibration. Only crops with active specialist checkpoints are supported.",
                    compatible_risk_models=["baseline_weather"]
                )
            )

    def register_crop(self, crop_spec: CropSpec) -> None:
        """Register or update a crop specification."""
        self._crops[crop_spec.crop_id.lower()] = crop_spec

    def get_crop(self, crop_id_or_name: str) -> Optional[CropSpec]:
        """Look up a crop by ID or display name (case-insensitive)."""
        if not crop_id_or_name:
            return None

        key = str(crop_id_or_name).strip().lower()
        if key in self._crops:
            return self._crops[key]

        for spec in self._crops.values():
            if spec.display_name.lower() == key:
                return spec

        return None

    def list_crops(self) -> List[CropSpec]:
        """Return list of all registered crop specifications."""
        return list(self._crops.values())

    def list_supported_crops(self) -> List[CropSpec]:
        """Return list of all crops currently supported for live ML diagnosis."""
        return [c for c in self._crops.values() if c.is_supported]

    def list_unsupported_crops(self) -> List[CropSpec]:
        """Return list of registered candidate crops not yet active for live ML diagnosis."""
        return [c for c in self._crops.values() if not c.is_supported]

    def is_crop_supported(self, crop_id_or_name: str) -> bool:
        """Check if a crop is currently supported for live ML disease diagnosis."""
        spec = self.get_crop(crop_id_or_name)
        return spec.is_supported if spec else False

    def is_model_available(self, crop_id_or_name: str) -> bool:
        """Check if the specialist model checkpoint for a crop exists on disk."""
        spec = self.get_crop(crop_id_or_name)
        return spec.is_model_available() if spec else False

    def get_compatible_risk_models(self, crop_id_or_name: str) -> List[str]:
        """Return the list of epidemiological risk models valid for this crop."""
        spec = self.get_crop(crop_id_or_name)
        return list(spec.compatible_risk_models) if spec else []

    def get_trusted_knowledge_sources(self, crop_id_or_name: str) -> List[Dict[str, Any]]:
        """Return trusted agricultural extension sources for this crop."""
        spec = self.get_crop(crop_id_or_name)
        return list(spec.trusted_knowledge_sources) if spec else []


# Singleton instance
_GLOBAL_CROP_REGISTRY: Optional[CropRegistry] = None


def get_crop_registry() -> CropRegistry:
    """Return the global CropRegistry instance."""
    global _GLOBAL_CROP_REGISTRY
    if _GLOBAL_CROP_REGISTRY is None:
        _GLOBAL_CROP_REGISTRY = CropRegistry()
    return _GLOBAL_CROP_REGISTRY
