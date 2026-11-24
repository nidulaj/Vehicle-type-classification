"""Two-stage vehicle prediction pipeline package."""

from src.pipeline.schema import DetectionResult
from src.pipeline.vehicle_pipeline import VehiclePipeline

__all__ = [
    "DetectionResult",
    "VehiclePipeline",
]
