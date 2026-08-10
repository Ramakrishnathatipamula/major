import sys
from pathlib import Path
from typing import Tuple, Optional

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.intelligence.context_builder import DiagnosticContext


class RAGQueryGenerator:
    """
    Generates targeted RAG search queries and metadata filters from a DiagnosticContext.
    Does not invent facts; uses strictly facts present in the diagnostic state.
    """
    @staticmethod
    def generate_query(context: DiagnosticContext) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Generate (query_string, crop_filter, disease_filter).

        Args:
            context: Clean DiagnosticContext object

        Returns:
            Tuple of (query_string, crop_filter, disease_filter)
        """
        crop = context.crop
        disease = context.disease
        risk_level = context.risk_level
        scenario = context.what_if_scenario

        # Normalize disease string for filters
        if "healthy" in disease.lower():
            disease_filter = None
            query_str = (
                f"What cultural management practices and crop sanitation recommendations "
                f"protect healthy {crop.lower()} foliage from disease infection?"
            )
            return query_str, crop, disease_filter

        # Disease detection (Late Blight / Early Blight)
        disease_filter = "Late Blight" if "late" in disease.lower() else "Early Blight" if "early" in disease.lower() else disease

        # Incorporate What-If scenario context if present
        if scenario and scenario.get("is_hypothetical"):
            params = scenario.get("scenario_parameters", {})
            scen_risk = scenario.get("scenario_risk_level", risk_level)

            param_desc = []
            if "override_relative_humidity_pct" in params:
                param_desc.append(f"relative humidity increased to {params['override_relative_humidity_pct']}%")
            elif "delta_temperature_c" in params:
                param_desc.append(f"temperature shifted by {params['delta_temperature_c']}°C")

            param_str = " and ".join(param_desc) if param_desc else "modified weather conditions"

            query_str = (
                f"What agricultural extension guidance applies to {crop.lower()} {disease_filter} "
                f"under a hypothetical scenario with {param_str} creating {scen_risk} risk?"
            )
            return query_str, crop, disease_filter

        # Standard baseline query with key weather factors
        primary_factor = context.risk_factors[0] if context.risk_factors else "favorable weather conditions"
        query_str = (
            f"What agricultural extension guidance applies to {crop.lower()} {disease_filter} "
            f"under {risk_level} environmental risk conditions driven by {primary_factor}?"
        )

        return query_str, crop, disease_filter
