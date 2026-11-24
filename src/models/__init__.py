"""Model wrappers package."""

from src.models.classifier import VehicleClassifier
from src.models.detector import VehicleDetector

__all__ = [
    "VehicleDetector",
    "VehicleClassifier",
]
