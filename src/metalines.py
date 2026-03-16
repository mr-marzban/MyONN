"""
Metalines module: phase-to-slot-length mapping and inverse mapping.

Silicon metalines are 1-D arrays of etched rectangular slots in an SOI substrate.
By choosing the slot length, each meta-atom imposes an arbitrary phase shift
on the guided TE wave (0 to 2π, transmission > 0.96).

FDTD-characterised data and linear-regression mappings are from:
Marzban, M.-R., Zarei, S., & Khavasi, A. (2020).
Optics Express, 28(24), 36668. https://doi.org/10.1364/OE.404386
"""

import numpy as np
from sklearn.linear_model import LinearRegression


# --------------------------------------------------------------------------
# Characterisation data (slot-length sweep, Lumerical FDTD, λ = 1.55 µm,
# slot width fixed at 140 nm)
# --------------------------------------------------------------------------
SLOT_LENGTH_DATA = np.array([
    5e-8, 1e-7, 1.5e-7, 2e-7, 2.5e-7, 3e-7, 3.5e-7, 4e-7, 4.5e-7, 5e-7,
    5.5e-7, 6e-7, 6.5e-7, 7e-7, 7.5e-7, 8e-7, 8.5e-7, 9e-7, 9.5e-7,
    1e-6, 1.05e-6, 1.1e-6, 1.15e-6, 1.2e-6, 1.25e-6, 1.3e-6, 1.35e-6,
    1.4e-6, 1.45e-6, 1.5e-6, 1.55e-6, 1.6e-6, 1.65e-6, 1.7e-6, 1.75e-6,
    1.8e-6, 1.85e-6, 1.9e-6, 1.95e-6, 2e-6, 2.05e-6, 2.1e-6, 2.15e-6,
    2.2e-6, 2.25e-6, 2.3e-6, 2.35e-6, 2.4e-6, 2.45e-6, 2.5e-6
])

NORM_PHASE_DATA = np.array([
    1.47569, 1.33102, 1.16544, 0.991634, 0.857614, 0.672238, 0.506028,
    0.360678, 0.223044, 0.0973003, -0.0748034, -0.245628, -0.393768,
    -0.497378, -0.643672, -0.797562, -0.95729, -1.11658, -1.24395,
    -1.41558, -1.56924, -1.70545, -1.84089, -1.97413, -2.15015, -2.31859,
    -2.46667, -2.57435, -2.72253, -2.87741, -3.0382, -3.20099, -3.32656,
    -3.49174, -3.63976, -3.78114, -3.91895, -4.05189, -4.22939, -4.39839,
    -4.54392, -4.64823, -4.7969, -4.95722, -5.12032, -5.27798, -5.40463,
    -5.56892, -5.71522, -5.85474
])

# d(phase)/d(slot_length) gradient data (from FDTD sweep)
GRAD_X_DATA = np.array([
    5e-8, 1e-7, 1.5e-7, 2e-7, 2.5e-7, 3e-7, 3.5e-7, 4e-7, 4.5e-7, 5e-7,
    5.5e-7, 6e-7, 6.5e-7, 7e-7, 7.5e-7, 8e-7, 8.5e-7, 9e-7, 9.5e-7, 1e-6,
    1.05e-6, 1.1e-6, 1.15e-6, 1.2e-6, 1.25e-6, 1.3e-6, 1.35e-6, 1.4e-6,
    1.45e-6, 1.5e-6, 1.55e-6, 1.6e-6, 1.65e-6, 1.7e-6, 1.75e-6, 1.8e-6,
    1.85e-6, 1.9e-6, 1.95e-6, 2e-6, 2.05e-6, 2.1e-6, 2.15e-6, 2.2e-6,
    2.25e-6, 2.3e-6, 2.35e-6, 2.4e-6, 2.45e-6
])

GRAD_Y_DATA = np.array([
    -2.8934e6, -3.3116e6, -3.4762e6, -3.4762e6, -2.6804e6, -3.7075e6,
    -3.3242e6, -2.907e6, -2.7527e6, -2.5149e6, -3.4421e6, -3.4165e6,
    -2.9628e6, -2.0722e6, -2.9259e6, -3.0778e6, -3.1946e6, -3.1858e6,
    -2.5474e6, -3.4325e6, -3.0732e6, -2.7243e6, -2.7088e6, -2.6648e6,
    -3.5203e6, -3.3689e6, -2.9616e6, -2.1535e6, -2.9636e6, -3.0977e6,
    -3.2158e6, -3.2558e6, -3.3036e6, -2.9604e6, -2.8276e6, -2.7562e6,
    -2.6587e6, -3.55e6, -3.3801e6, -2.9107e6, -2.086e6, -2.9736e6,
    -3.2062e6, -3.262e6, -3.1533e6, -2.533e6, -3.2857e6, -2.9262e6,
    -2.7902e6
])


def mask_length(Phasetarget_1, Phasetarget_2, Phasetarget_3):
    """
    Map target phase profiles to physical slot lengths via linear regression.

    The linear regression model is fitted on the FDTD-characterised
    phase vs. slot-length curve (Fig. 2c of the paper).

    Args:
        Phasetarget_1: Phase profile for metaline layer 1 (rad), shape (N,)
        Phasetarget_2: Phase profile for metaline layer 2 (rad), shape (N,)
        Phasetarget_3: Phase profile for metaline layer 3 (rad), shape (N,)

    Returns:
        Tuple of three slot-length arrays (m), one per layer
    """
    lr = LinearRegression()
    lr.fit(NORM_PHASE_DATA[:, None], SLOT_LENGTH_DATA)
    Slot_1 = lr.predict(Phasetarget_1[:, None])
    Slot_2 = lr.predict(Phasetarget_2[:, None])
    Slot_3 = lr.predict(Phasetarget_3[:, None])
    return Slot_1, Slot_2, Slot_3


def mask_Phase(Slot_1_length_mask, Slot_2_length_mask, Slot_3_length_mask):
    """
    Inverse mapping: recover phase profiles from slot lengths.

    Args:
        Slot_1_length_mask: Slot lengths for layer 1 (m), shape (N,)
        Slot_2_length_mask: Slot lengths for layer 2 (m), shape (N,)
        Slot_3_length_mask: Slot lengths for layer 3 (m), shape (N,)

    Returns:
        Tuple of three phase arrays (rad), one per layer
    """
    lr = LinearRegression()
    lr.fit(SLOT_LENGTH_DATA[:, None], NORM_PHASE_DATA)
    Ph1 = lr.predict(Slot_1_length_mask[:, None])
    Ph2 = lr.predict(Slot_2_length_mask[:, None])
    Ph3 = lr.predict(Slot_3_length_mask[:, None])
    return Ph1, Ph2, Ph3


def mask_length_diff(Slot_1_length_mask, Slot_2_length_mask, Slot_3_length_mask):
    """
    Compute d(slot_length)/d(phase) gradient used in the adjoint update.

    This derivative is needed in Eq. (3) of the paper:
        dC/dw^m = (d phi^wm / dw^m) ⊗ (dC / d phi^wm)

    Args:
        Slot_1_length_mask: Slot lengths for layer 1 (m)
        Slot_2_length_mask: Slot lengths for layer 2 (m)
        Slot_3_length_mask: Slot lengths for layer 3 (m)

    Returns:
        Tuple of three gradient arrays, one per layer
    """
    lr = LinearRegression()
    lr.fit(GRAD_X_DATA[:, None], GRAD_Y_DATA)
    d1 = lr.predict(Slot_1_length_mask[:, None])
    d2 = lr.predict(Slot_2_length_mask[:, None])
    d3 = lr.predict(Slot_3_length_mask[:, None])
    return d1, d2, d3
