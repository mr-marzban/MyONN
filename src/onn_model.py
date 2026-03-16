"""
ONN model: forward propagation and inference.

Implements Eq. (1) of the paper:
    E_out = (∏ F⁺ P^m F Φ^{w^m}) E_in

where F/F⁺ are the discrete / inverse Fourier transforms, P^m is the
plane-wave propagation kernel, and Φ^{w^m} = exp(j φ^{w^m}) is the
diagonal phase-modulation matrix of the m-th metaline.

Reference:
Marzban, M.-R., Zarei, S., & Khavasi, A. (2020).
Optics Express, 28(24), 36668. https://doi.org/10.1364/OE.404386
"""

import numpy as np
from .data_utils import DIM, GAP_LEN, DESIRED_LEN


def ONN_forward(Ein, Phi1, Phi2, Phi3, p_shifted, p_shifted_output):
    """
    Single forward pass through the 3-layer ONN.

    Each layer applies phase modulation then free-space diffraction:
        E_{m} = IFFT( P^m · FFT( Φ^m · E_{m-1} ) )

    Args:
        Ein:               Input electric field vector, shape (N,)
        Phi1/2/3:          Complex phase masks for each metaline, shape (N,)
        p_shifted:         Propagation kernel between layers (FFT-shifted)
        p_shifted_output:  Propagation kernel from last layer to output plane

    Returns:
        Eout: Output electric field vector, shape (N,)
        E1, E2: Intermediate fields (needed for adjoint back-propagation)
    """
    h1 = Phi1 * Ein
    E1 = np.fft.ifft(p_shifted * np.fft.fft(h1))

    h2 = Phi2 * E1
    E2 = np.fft.ifft(p_shifted * np.fft.fft(h2))

    h3 = Phi3 * E2
    Eout = np.fft.ifft(p_shifted_output * np.fft.fft(h3))

    return Eout, E1, E2


def _detector_sums(intensity, dim=DIM, gap_len=GAP_LEN, desired_len=DESIRED_LEN):
    """Integrate intensity over each of the 10 detector regions."""
    q1 = 2 + dim * 2
    q_mat = [q1 + i * (desired_len + gap_len) for i in range(10)]
    return [float(np.sum(intensity[q : q + desired_len])) for q in q_mat]


def ONN_Test(
    Phasetarget_1,
    Phasetarget_2,
    Phasetarget_3,
    test_image_reduced,
    test_labels,
    p_shifted,
    p_shifted_output,
    dim=DIM,
    gap_len=GAP_LEN,
    desired_len=DESIRED_LEN,
):
    """
    Evaluate ONN classification accuracy on the full test set.

    Args:
        Phasetarget_1/2/3:  Trained phase profiles (rad), shape (N,)
        test_image_reduced: Flattened test images, shape (n_test, N)
        test_labels:        Ground-truth integer labels, shape (n_test,)
        p_shifted:          Inter-layer propagation kernel
        p_shifted_output:   Output-layer propagation kernel
        dim, gap_len, desired_len: Detector layout parameters

    Returns:
        accuracy: Classification accuracy (%)
    """
    Phi1 = np.exp(1j * Phasetarget_1)
    Phi2 = np.exp(1j * Phasetarget_2)
    Phi3 = np.exp(1j * Phasetarget_3)

    correct = 0
    n_test = test_image_reduced.shape[0]

    for idx in range(n_test):
        Ein = test_image_reduced[idx]
        Eout, _, _ = ONN_forward(Ein, Phi1, Phi2, Phi3, p_shifted, p_shifted_output)
        intensity = np.real(Eout * np.conj(Eout))
        answer = _detector_sums(intensity, dim, gap_len, desired_len)
        if np.argmax(answer) == test_labels[idx]:
            correct += 1

    return (correct / n_test) * 100
