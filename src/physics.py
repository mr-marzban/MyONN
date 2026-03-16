"""
Physics module: propagation kernels and fundamental optical functions.

Based on the free-space propagation model described in:
Marzban, M.-R., Zarei, S., & Khavasi, A. (2020).
Optics Express, 28(24), 36668. https://doi.org/10.1364/OE.404386
"""

import numpy as np


# Physical constants for the silicon metaline ONN
WAVELENGTH = 1.55e-6        # Operating wavelength (m)
N_EFF = 2.96607             # Effective refractive index of SOI slab waveguide
META_ATOM_PITCH = 0.5e-6    # Lattice constant / meta-atom pitch (m)


def gaussian(x, mu, sig):
    """
    Gaussian function used to define desired output intensity distributions.

    Each output detector region has a Gaussian target centered at the detector
    position, as described in Section 4 of the paper.

    Args:
        x:   Input values (array or scalar)
        mu:  Center/mean of the Gaussian
        sig: Standard deviation

    Returns:
        Gaussian-evaluated values with same shape as x
    """
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def propagation_kernel(N, pitch, distance, wavelength=WAVELENGTH, n_eff=N_EFF):
    """
    Compute the free-space angular spectrum propagation kernel P^m.

    This implements the plane-wave propagation matrix from Eq. (1) of the paper.
    Each plane-wave component accumulates a phase factor as it travels between
    two successive metaline layers separated by `distance`.

    Args:
        N:          Number of meta-atoms (pixels) per layer
        pitch:      Meta-atom pitch / lattice constant (m)
        distance:   Propagation distance between layers (m)
        wavelength: Free-space wavelength (m), default 1.55 µm
        n_eff:      Effective refractive index of the waveguide slab

    Returns:
        p_shifted: 1-D complex array of length N, FFT-shifted propagation kernel
    """
    k0 = 2 * np.pi / wavelength
    k = k0 * n_eff

    ky1 = np.linspace(-N / 2, N / 2 - 1, num=N)
    ky = 2 * np.pi * (1.0 / (N * pitch)) * ky1

    # Plane-wave phase accumulated over propagation distance
    p = np.exp(1j * k * distance * np.sqrt(1 - (ky / k) ** 2))
    p_shifted = np.fft.fftshift(p)
    return p_shifted
