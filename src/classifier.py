"""Backward-compatibility bridge for legacy classifier imports.

New code should import from `src.models.classifier` or `src.models`.
"""

from src.models.classifier import VehicleClassifier

__all__ = ["VehicleClassifier"]
