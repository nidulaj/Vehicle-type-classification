"""Model training package for vehicle classification."""

from src.training.trainer import (
    VehicleTrainer,
    compute_balanced_class_weights,
)

__all__ = [
    "VehicleTrainer",
    "compute_balanced_class_weights",
]
