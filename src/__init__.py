"""
MyONN: Integrated Photonic Neural Network based on Silicon Metalines

Implementation of the optical neural network (ONN) from:
Marzban, M.-R., Zarei, S., & Khavasi, A. (2020).
"Integrated photonic neural network based on silicon metalines."
Optics Express, 28(24), 36668.
https://doi.org/10.1364/OE.404386
"""

__version__ = "1.0.0"
__author__ = "Mahmood-Reza Marzban"

from .physics import gaussian, propagation_kernel
from .metalines import mask_length, mask_Phase, mask_length_diff
from .data_utils import load_mnist_data, build_gaussian_targets
from .onn_model import ONN_forward, ONN_Test
from .training import train_onn
from .visualization import (
    plot_confusion_matrix,
    visualize_phase_profiles,
    plot_training_curves,
)
from .io_utils import (
    Phasetarget_saver,
    Slot_length_mask_saver,
    Slot_length_mask_load,
)

__all__ = [
    "gaussian",
    "propagation_kernel",
    "mask_length",
    "mask_Phase",
    "mask_length_diff",
    "load_mnist_data",
    "build_gaussian_targets",
    "ONN_forward",
    "ONN_Test",
    "train_onn",
    "plot_confusion_matrix",
    "visualize_phase_profiles",
    "plot_training_curves",
    "Phasetarget_saver",
    "Slot_length_mask_saver",
    "Slot_length_mask_load",
]
