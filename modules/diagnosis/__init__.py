"""
End-to-End Diagnostic Pipeline for AI Crop Doctor.
Orchestrates DiseasePredictor, GradCAM, WeatherClient, and RiskEngine.
"""

from .pipeline import CropDoctor, DiagnosticResult

__all__ = [
    "CropDoctor",
    "DiagnosticResult",
]
