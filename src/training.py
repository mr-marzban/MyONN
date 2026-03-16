"""
Training module: adjoint-method gradient descent with ADAM optimiser.

Implements the back-propagation approach from Section 3 of the paper.
The gradient dC/d(phi^{w^m}) is obtained via one forward simulation and
one adjoint backward simulation (Eq. 5), then converted to dC/dw^m via
the chain rule (Eq. 3).

Optimiser: ADAM (Kingma & Ba, 2014) — reference [25] in the paper.

Reference:
Marzban, M.-R., Zarei, S., & Khavasi, A. (2020).
Optics Express, 28(24), 36668. https://doi.org/10.1364/OE.404386
"""

import math
import numpy as np

from .onn_model import ONN_forward, ONN_Test, _detector_sums
from .metalines import mask_length, mask_length_diff
from .io_utils import Phasetarget_saver, Slot_length_mask_saver
from .data_utils import DIM, GAP_LEN, DESIRED_LEN


def compute_cost(AL, Y):
    """Mean squared error cost (Eq. 2 of the paper)."""
    m = AL.shape[0]
    return (1 / m) * np.sum((AL - Y) ** 2)


def train_onn(
    train_images_reduced,
    Y_train_extended,
    Y_train,
    train_labels,
    test_image_reduced,
    test_labels,
    p_shifted,
    p_shifted_output,
    epoch_max=50,
    batch_size=64,
    alpha=0.065,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8,
    dim=DIM,
    gap_len=GAP_LEN,
    desired_len=DESIRED_LEN,
    phase_min=-2 * math.pi,
    phase_max=0.0,
    verbose=True,
    save_dir=None,
):
    """
    Train the 3-layer ONN using adjoint method + ADAM.

    The adjoint field a^k = E_out,k ⊗ (I^k − I_des,k) is back-propagated
    through the optical system to compute dC/d(phi^{w^m}) efficiently
    (one forward pass + one adjoint backward pass per iteration).

    Args:
        train_images_reduced: Flattened training images (n_train, N)
        Y_train_extended:     Gaussian target vectors  (n_train, N)
        Y_train:              One-hot labels            (n_train, 10)
        train_labels:         Integer labels            (n_train,)
        test_image_reduced:   Flattened test images     (n_test, N)
        test_labels:          Integer test labels       (n_test,)
        p_shifted:            Inter-layer propagation kernel
        p_shifted_output:     Output-layer propagation kernel
        epoch_max:            Number of training epochs
        batch_size:           Mini-batch size (default 64, as in the paper)
        alpha:                ADAM learning rate
        adam_beta1/2/eps:     ADAM hyper-parameters
        dim/gap_len/desired_len: Detector layout
        phase_min/max:        Hard clip range for phase values (rad)
        verbose:              Print progress every 2 epochs
        save_dir:             If provided, save phase/slot files here after training

    Returns:
        dict with keys:
            Phasetarget_1/2/3  - final trained phase profiles
            cost_history       - list of per-epoch cost values
            test_acc_history   - list of per-epoch test accuracy (%)
            train_acc_history  - list of per-epoch train accuracy (%)
    """
    N = dim * dim
    Phasetarget_0 = np.zeros(N)
    phase_init = -2 * math.pi * np.ones(N)

    Phasetarget_1 = phase_init.copy()
    Phasetarget_2 = phase_init.copy()
    Phasetarget_3 = phase_init.copy()

    Slot_1, Slot_2, Slot_3 = mask_length(Phasetarget_1, Phasetarget_2, Phasetarget_3)

    # ADAM state
    V1 = V2 = V3 = 0.0
    S1 = S2 = S3 = 0.0

    cost_history = []
    test_acc_history = []
    train_acc_history = []

    # Accuracy before any training
    acc0 = ONN_Test(
        Phasetarget_1, Phasetarget_2, Phasetarget_3,
        test_image_reduced, test_labels, p_shifted, p_shifted_output,
        dim, gap_len, desired_len,
    )
    test_acc_history.append(acc0)
    if verbose:
        print(f"Initial test accuracy: {acc0:.1f}%")

    for epoch in range(1, epoch_max):
        start = (epoch - 1) * batch_size
        end = epoch * batch_size

        batch_imgs = train_images_reduced[start:end]
        batch_Y_ext = Y_train_extended[start:end]
        batch_Y = Y_train[start:end]
        batch_labels = train_labels[start:end]

        Phi0 = np.exp(1j * Phasetarget_0)
        Phi1 = np.exp(1j * Phasetarget_1)
        Phi2 = np.exp(1j * Phasetarget_2)
        Phi3 = np.exp(1j * Phasetarget_3)

        diff_S1, diff_S2, diff_S3 = mask_length_diff(Slot_1, Slot_2, Slot_3)

        predict_train = []
        correct = 0

        for counter in range(batch_size):
            Ein = batch_imgs[counter]
            Is_des = batch_Y_ext[counter]

            # ---- Forward pass ----
            Eout, E1, E2 = ONN_forward(Ein, Phi1, Phi2, Phi3, p_shifted, p_shifted_output)
            Is = np.real(Eout * np.conj(Eout))

            # ---- Adjoint back-propagation (Eq. 5) ----
            a_adj = Eout * (Is - Is_des)

            b1 = np.fft.fft(a_adj)
            d1 = np.fft.ifft(np.conj(p_shifted_output) * b1)
            e1 = np.conj(Phi3) * d1

            b2 = np.fft.fft(e1)
            d2 = np.fft.ifft(np.conj(p_shifted) * b2)
            e2 = np.conj(Phi2) * d2

            b3 = np.fft.fft(e2)
            d3 = np.fft.ifft(np.conj(p_shifted) * b3)
            e3 = np.conj(Phi1) * d3

            dC1 = -4 * np.real(1j * np.conj(Ein) * e3)
            dC2 = -4 * np.real(1j * np.conj(E1) * e2)
            dC3 = -4 * np.real(1j * np.conj(E2) * e1)

            # ---- ADAM update ----
            V1 = adam_beta1 * V1 + (1 - adam_beta1) * dC1
            V2 = adam_beta1 * V2 + (1 - adam_beta1) * dC2
            V3 = adam_beta1 * V3 + (1 - adam_beta1) * dC3

            S1 = adam_beta2 * S1 + (1 - adam_beta2) * dC1 ** 2
            S2 = adam_beta2 * S2 + (1 - adam_beta2) * dC2 ** 2
            S3 = adam_beta2 * S3 + (1 - adam_beta2) * dC3 ** 2

            V1c = V1 / (1 - adam_beta1 ** epoch)
            V2c = V2 / (1 - adam_beta1 ** epoch)
            V3c = V3 / (1 - adam_beta1 ** epoch)

            S1c = S1 / (1 - adam_beta2 ** epoch)
            S2c = S2 / (1 - adam_beta2 ** epoch)
            S3c = S3 / (1 - adam_beta2 ** epoch)

            Phasetarget_1 = Phasetarget_1 - alpha * V1c / (np.sqrt(S1c) + adam_eps)
            Phasetarget_2 = Phasetarget_2 - alpha * V2c / (np.sqrt(S2c) + adam_eps)
            Phasetarget_3 = Phasetarget_3 - alpha * V3c / (np.sqrt(S3c) + adam_eps)

            # Hard-clip to [phase_min, phase_max]
            Phasetarget_1 = np.clip(Phasetarget_1, phase_min, phase_max)
            Phasetarget_2 = np.clip(Phasetarget_2, phase_min, phase_max)
            Phasetarget_3 = np.clip(Phasetarget_3, phase_min, phase_max)

            Phi1 = np.exp(1j * Phasetarget_1)
            Phi2 = np.exp(1j * Phasetarget_2)
            Phi3 = np.exp(1j * Phasetarget_3)

            Slot_1, Slot_2, Slot_3 = mask_length(Phasetarget_1, Phasetarget_2, Phasetarget_3)

            answer = _detector_sums(Is, dim, gap_len, desired_len)
            predict_train.append(answer)
            if np.argmax(answer) == batch_labels[counter]:
                correct += 1

        # ---- Per-epoch metrics ----
        predict_train = np.array(predict_train)
        predict_norm = np.array([p / np.max(p) for p in predict_train])
        cost = compute_cost(predict_norm, batch_Y)
        train_acc = (correct / batch_size) * 100
        test_acc = ONN_Test(
            Phasetarget_1, Phasetarget_2, Phasetarget_3,
            test_image_reduced, test_labels, p_shifted, p_shifted_output,
            dim, gap_len, desired_len,
        )

        cost_history.append(cost)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        if verbose and (epoch - 1) % 2 == 0:
            print(
                f"Epoch {epoch:3d} | cost {cost:.4f} | "
                f"train acc {train_acc:.1f}% | test acc {test_acc:.1f}%"
            )

    if save_dir:
        Phasetarget_saver(Phasetarget_1, Phasetarget_2, Phasetarget_3, save_dir)
        Slot_length_mask_saver(Slot_1, Slot_2, Slot_3, save_dir)

    return {
        "Phasetarget_1": Phasetarget_1,
        "Phasetarget_2": Phasetarget_2,
        "Phasetarget_3": Phasetarget_3,
        "cost_history": cost_history,
        "test_acc_history": test_acc_history,
        "train_acc_history": train_acc_history,
    }
