"""Tests for the metalines phase-length mapping."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import math
from metalines import mask_length, mask_Phase, mask_length_diff


def test_mask_length_shape():
    phase = -2 * math.pi * np.ones(196)
    S1, S2, S3 = mask_length(phase, phase, phase)
    assert S1.shape == (196,)
    assert S2.shape == (196,)
    assert S3.shape == (196,)


def test_mask_phase_roundtrip():
    """Phase -> slot length -> phase should be approximately invertible."""
    phase_in = np.linspace(-2 * math.pi, 0, 196)
    S1, S2, S3 = mask_length(phase_in, phase_in, phase_in)
    Ph1, Ph2, Ph3 = mask_Phase(S1, S2, S3)
    np.testing.assert_allclose(Ph1, phase_in, atol=0.5)


def test_mask_length_diff_shape():
    phase = -2 * math.pi * np.ones(196)
    S1, S2, S3 = mask_length(phase, phase, phase)
    d1, d2, d3 = mask_length_diff(S1, S2, S3)
    assert d1.shape == (196,)
