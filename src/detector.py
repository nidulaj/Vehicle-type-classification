"""Backward-compatibility bridge for legacy detector imports.

New code should import from `src.models.detector` or `src.models`.
"""

from src.models.detector import VehicleDetector

__all__ = ["VehicleDetector"]
