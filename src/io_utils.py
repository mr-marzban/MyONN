"""
I/O utilities: save and load trained phase profiles and slot-length masks.

Reference:
Marzban, M.-R., Zarei, S., & Khavasi, A. (2020).
Optics Express, 28(24), 36668. https://doi.org/10.1364/OE.404386
"""

from pathlib import Path
import numpy as np


def Phasetarget_saver(Phasetarget_1, Phasetarget_2, Phasetarget_3, save_dir="Phase_Target"):
    """Save the three phase target arrays to text files."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(save_dir / "Phasetarget_1.txt", Phasetarget_1, delimiter=",")
    np.savetxt(save_dir / "Phasetarget_2.txt", Phasetarget_2, delimiter=",")
    np.savetxt(save_dir / "Phasetarget_3.txt", Phasetarget_3, delimiter=",")


def Phasetarget_load(save_dir="Phase_Target"):
    """Load three phase target arrays from text files."""
    save_dir = Path(save_dir)
    Ph1 = np.loadtxt(save_dir / "Phasetarget_1.txt")
    Ph2 = np.loadtxt(save_dir / "Phasetarget_2.txt")
    Ph3 = np.loadtxt(save_dir / "Phasetarget_3.txt")
    return Ph1, Ph2, Ph3


def Slot_length_mask_saver(
    Slot_1, Slot_2, Slot_3, save_dir="Slot_length_mask", version=1
):
    """Save the three slot-length mask arrays to text files."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(save_dir / f"Slot_1_length_mask_version{version}.txt", Slot_1, delimiter=",")
    np.savetxt(save_dir / f"Slot_2_length_mask_version{version}.txt", Slot_2, delimiter=",")
    np.savetxt(save_dir / f"Slot_3_length_mask_version{version}.txt", Slot_3, delimiter=",")


def Slot_length_mask_load(save_dir="Slot_length_mask", version=1):
    """Load three slot-length mask arrays from text files."""
    save_dir = Path(save_dir)
    S1 = np.loadtxt(save_dir / f"Slot_1_length_mask_version{version}.txt")
    S2 = np.loadtxt(save_dir / f"Slot_2_length_mask_version{version}.txt")
    S3 = np.loadtxt(save_dir / f"Slot_3_length_mask_version{version}.txt")
    return S1, S2, S3
