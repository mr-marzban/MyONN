"""Tests for the physics module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from physics import gaussian, propagation_kernel


def test_gaussian_peak():
    """Gaussian should equal 1 at its mean."""
    assert gaussian(0.0, 0.0, 1.0) == 1.0


def test_gaussian_symmetry():
    x = np.linspace(-5, 5, 100)
    g = gaussian(x, 0.0, 1.0)
    # Symmetric around mu=0
    np.testing.assert_allclose(g, g[::-1], atol=1e-10)


def test_propagation_kernel_shape():
    N = 196
    p = propagation_kernel(N, pitch=0.5e-6, distance=40e-6)
    assert p.shape == (N,)


def test_propagation_kernel_unit_amplitude():
    """Free-space propagation preserves amplitude (|exp(j*phi)| == 1)."""
    p = propagation_kernel(196, 0.5e-6, 40e-6)
    np.testing.assert_allclose(np.abs(p), 1.0, atol=1e-10)
