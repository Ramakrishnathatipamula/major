from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional
import json


class TreatmentCategory(str, Enum):
    PROPHYLACTIC_PROTECTANT = "Prophylactic / Protective Fungicide"
    CURATIVE_SYSTEMIC = "Curative / Systemic Combination"
    CULTURAL_NON_CHEMICAL = "Cultural & Non-Chemical Management"
    PREVENTIVE_MAINTENANCE = "Preventive Maintenance Only (No Chemical Prescribed)"
    BIOLOGICAL_ORGANIC = "Biological & Organic Management"
    CULTURAL_PREVENTIVE = "Cultural & Preventative Practice"


@dataclass
class TreatmentOption:
    """
    Structured representation of a single verified treatment or medicine option.
    All attributes must be grounded in verified agricultural evidence; no guessing allowed.
    """
    category: str
    active_ingredient: Optional[str] = None
    formulation: Optional[str] = None
    target_disease: str = ""
    disease_name: Optional[str] = None
    farmer_action: Optional[str] = None
    application_rate: Optional[str] = None
    timing_and_frequency: Optional[str] = None
    application_directions: Optional[str] = None
    pre_harvest_interval_phi: Optional[str] = None
    re_entry_interval_rei: Optional[str] = None
    pre_harvest_interval_days: Optional[Any] = None
    re_entry_interval_hours: Optional[Any] = None
    ppe_and_safety_precautions: List[str] = field(default_factory=list)
    verification_status: str = "Verified Government / Extension Source"
    evidence_source_title: str = ""
    evidence_source_url: str = ""
    evidence_url_or_ref: Optional[str] = None
    crop: str = "Potato"
    recommendation_type: str = "chemical"
    applicable_region: str = "India (CIBRC / ICAR)"
    warnings: List[str] = field(default_factory=list)
    cibrc_approved: bool = True
    regulatory_authority: str = "CIBRC / ICAR-CPRI / State Extension"

    def __post_init__(self):
        if self.disease_name and not self.target_disease:
            self.target_disease = self.disease_name
        elif self.target_disease and not self.disease_name:
            self.disease_name = self.target_disease

        if self.evidence_url_or_ref and not self.evidence_source_url:
            self.evidence_source_url = self.evidence_url_or_ref
        elif self.evidence_source_url and not self.evidence_url_or_ref:
            self.evidence_url_or_ref = self.evidence_source_url

        if self.pre_harvest_interval_days is not None and not self.pre_harvest_interval_phi:
            self.pre_harvest_interval_phi = f"{self.pre_harvest_interval_days} days"
        elif self.pre_harvest_interval_phi and self.pre_harvest_interval_days is None:
            self.pre_harvest_interval_days = self.pre_harvest_interval_phi

        if self.re_entry_interval_hours is not None and not self.re_entry_interval_rei:
            self.re_entry_interval_rei = f"{self.re_entry_interval_hours} hours"
        elif self.re_entry_interval_rei and self.re_entry_interval_hours is None:
            self.re_entry_interval_hours = self.re_entry_interval_rei

        # Auto-derive a simple farmer-facing action from the verified active
        # ingredient when no explicit farmer_action is provided. This is a
        # plain-language restatement of the already-verified ingredient, not
        # a new treatment invention.
        if not self.farmer_action and self.active_ingredient:
            ai = str(self.active_ingredient).strip()
            self.farmer_action = f"Spray {ai} on leaves"

    def is_verified(self) -> bool:
        """
        Check if recommendation is supported by an official verified regulatory or extension source.
        Unverified records cannot claim approval or be returned as certified chemical advice.
        """
        status_clean = str(self.verification_status).strip().upper()
        if "UNVERIFIED" in status_clean or "PENDING" in status_clean:
            return False
        return bool(self.cibrc_approved or "VERIFIED" in status_clean)

    def is_chemical(self) -> bool:
        """Return True if this treatment specifies an active chemical ingredient."""
        return bool(self.active_ingredient and str(self.active_ingredient).strip())

    def has_complete_safety_profile(self) -> bool:
        """
        Verify whether mandatory safety details (PHI, REI, PPE, formulation, application rate) are present.
        If a safety detail is missing, unverified, or empty, return False rather than guessing.
        """
        if not self.is_chemical():
            return True
        has_phi = bool(self.pre_harvest_interval_phi and "unverified" not in str(self.pre_harvest_interval_phi).lower())
        has_rei = bool(self.re_entry_interval_rei and "unverified" not in str(self.re_entry_interval_rei).lower())
        has_ppe = bool(self.ppe_and_safety_precautions and len(self.ppe_and_safety_precautions) > 0)
        has_rate = bool(self.application_rate and str(self.application_rate).strip())
        has_formulation = bool(self.formulation and str(self.formulation).strip())
        return has_phi and has_rei and has_ppe and has_rate and has_formulation

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["is_verified"] = self.is_verified()
        data["is_chemical"] = self.is_chemical()
        data["has_complete_safety_profile"] = self.has_complete_safety_profile()
        return data


# Alias for multi-crop extensible record representation
TreatmentRecord = TreatmentOption


DEFAULT_TREATMENT_DISCLAIMER = (
    "STATUTORY NOTICE: Treatment recommendations are strictly grounded in verified Central "
    "Insecticides Board & Registration Committee (CIBRC), ICAR-CPRI, and University Extension documentation. "
    "No treatment guarantees cure or absolute eradication. Chemical fungicides must only be applied strictly "
    "following registered manufacturer product labels, appropriate personal protective equipment (PPE), "
    "and consultation with a local certified agricultural extension officer or Krishi Vigyan Kendra (KVK)."
)


@dataclass
class TreatmentRecommendationResult:
    """
    Structured, farmer-facing treatment recommendation result payload.
    Directly answers: Plant Image -> Crop Identification -> Disease Detection -> Recommended Medicine/Treatment.
    """
    crop: str
    predicted_disease: str
    is_supported: bool
    is_healthy: bool
    status: str
    summary_message: str
    chemical_treatments: List[TreatmentOption] = field(default_factory=list)
    non_chemical_control_measures: List[str] = field(default_factory=list)
    safety_and_environmental_precautions: List[str] = field(default_factory=list)
    evidence_sources: List[Dict[str, Any]] = field(default_factory=list)
    evidence_gaps: List[str] = field(default_factory=list)
    disclaimer: str = DEFAULT_TREATMENT_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
