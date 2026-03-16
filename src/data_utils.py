"""
Data utilities: MNIST loading, preprocessing, and Gaussian target generation.

Input images are down-sampled to 14×14 pixels and encoded as the amplitude of
the input electric field. Each detector region has a Gaussian desired-output
profile, as described in Section 4 of the paper.

Reference:
Marzban, M.-R., Zarei, S., & Khavasi, A. (2020).
Optics Express, 28(24), 36668. https://doi.org/10.1364/OE.404386
"""

import numpy as np
import cv2
from keras.datasets import mnist
from keras.utils import np_utils

from .physics import gaussian


# --------------------------------------------------------------------------
# Default hyper-parameters (match the reduced ONN structure in the paper)
# --------------------------------------------------------------------------
DIM = 14            # Spatial dimension after downsampling (14×14 = 196 neurons)
GAP_LEN = 4         # Gap between neighbouring detector regions (pixels)
DESIRED_LEN = 10    # Width of each detector region (pixels)
SIG = 4             # Gaussian sigma for desired output (pixels)
DOWN = 1            # Crop start (to remove border artifacts)
UP = 27             # Crop end


def load_mnist_data(
    train_sample=10000,
    test_sample=2000,
    dim=DIM,
    down=DOWN,
    up=UP,
    pad_amt=0,
    seed=None,
):
    """
    Load, crop, resize, and normalise MNIST images.

    Args:
        train_sample: Number of training images to use
        test_sample:  Number of test images to use
        dim:          Target spatial size (images resized to dim×dim)
        down:         Pixel crop start (row/col)
        up:           Pixel crop end (row/col)
        pad_amt:      Zero-padding amount before resize
        seed:         Optional random seed for reproducibility

    Returns:
        dict with keys:
            train_images_org    - original 28×28 train images
            train_images_reduced - flattened dim²-vectors, float32
            train_labels        - integer labels
            test_images_org     - original 28×28 test images
            test_image_reduced  - flattened dim²-vectors, float32
            test_labels         - integer labels
    """
    rng = np.random.RandomState(seed)

    (train_imgs_raw, train_lbl_raw), (test_imgs_raw, test_lbl_raw) = mnist.load_data()

    # --- Training set ---
    idx_train = rng.randint(0, len(train_imgs_raw), size=train_sample)
    train_images_org = train_imgs_raw[idx_train]
    train_labels = train_lbl_raw[idx_train]

    # Crop + resize
    cropped = train_images_org[:, down:up, down:up]
    reduced = []
    for img in cropped:
        img = img / 255.0
        img = np.pad(img, ((pad_amt, pad_amt),), mode="constant")
        img = cv2.resize(img, dsize=(dim, dim), interpolation=cv2.INTER_AREA)
        reduced.append(img)
    train_images_reduced = np.array(reduced, dtype=np.float32).reshape(train_sample, dim * dim)

    # --- Test set ---
    idx_test = rng.randint(0, len(test_imgs_raw), size=test_sample)
    test_images_org = test_imgs_raw[idx_test]
    test_labels = test_lbl_raw[idx_test]

    cropped_t = test_images_org[:, down:up, down:up]
    reduced_t = []
    for img in cropped_t:
        img = img / 255.0
        img = np.pad(img, ((pad_amt, pad_amt),), mode="constant")
        img = cv2.resize(img, dsize=(dim, dim), interpolation=cv2.INTER_AREA)
        reduced_t.append(img)
    test_image_reduced = np.array(reduced_t, dtype=np.float32).reshape(test_sample, dim * dim)

    return {
        "train_images_org": train_images_org,
        "train_images_reduced": train_images_reduced,
        "train_labels": train_labels,
        "test_images_org": test_images_org,
        "test_image_reduced": test_image_reduced,
        "test_labels": test_labels,
    }


def build_gaussian_targets(labels, dim=DIM, gap_len=GAP_LEN, desired_len=DESIRED_LEN, sig=SIG):
    """
    Build extended Gaussian target vectors for each sample.

    For a sample labelled class c, the output vector has a Gaussian peak
    centred over detector c (and zeros elsewhere), following the target
    definition in Section 4 of the paper (σ² = 4 µm²).

    Args:
        labels:      Integer class labels, shape (N_samples,)
        dim:         Spatial dimension (images are dim×dim)
        gap_len:     Gap between detector regions (pixels)
        desired_len: Width of each detector region (pixels)
        sig:         Gaussian sigma (pixels)

    Returns:
        Y_extended: Float array of shape (N_samples, dim²) with Gaussian targets
    """
    N = len(labels)
    total = dim * dim
    n_classes = 10

    q1 = 2 + dim * 2
    q_mat = [q1 + i * (desired_len + gap_len) for i in range(n_classes)]

    x_values = np.linspace(q_mat[0], q_mat[0] + desired_len, desired_len)
    gaussian_output = gaussian(x_values, q_mat[0] + desired_len / 2, sig)

    Y_one_hot = np_utils.to_categorical(labels, num_classes=n_classes)
    Y_extended = np.zeros((N, total), dtype=np.float32)

    for sample_num in range(N):
        for cls in range(n_classes):
            start = q_mat[cls]
            Y_extended[sample_num, start : start + desired_len] = (
                gaussian_output * Y_one_hot[sample_num, cls]
            )

    return Y_extended
