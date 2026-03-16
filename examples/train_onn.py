"""
Example: Train the 3-layer silicon-metaline ONN on MNIST.

This script reproduces the training experiment described in Section 4 of:

  Marzban, M.-R., Zarei, S., & Khavasi, A. (2020).
  "Integrated photonic neural network based on silicon metalines."
  Optics Express, 28(24), 36668.
  https://doi.org/10.1364/OE.404386

Usage
-----
    python examples/train_onn.py

The script will:
  1. Load and preprocess MNIST (14×14 down-sampled images)
  2. Set up the free-space propagation kernels
  3. Train 3 metaline phase profiles via adjoint method + ADAM
  4. Report test accuracy and save training-curve / confusion-matrix plots
"""

import sys
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from keras.utils import np_utils

from physics import propagation_kernel, WAVELENGTH, N_EFF, META_ATOM_PITCH
from data_utils import load_mnist_data, build_gaussian_targets, DIM
from training import train_onn
from visualization import plot_training_curves, plot_confusion_matrix, visualize_phase_profiles
from onn_model import ONN_forward, _detector_sums

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    # ------------------------------------------------------------------ #
    # 1. Load data                                                         #
    # ------------------------------------------------------------------ #
    print("Loading MNIST …")
    data = load_mnist_data(train_sample=10000, test_sample=2000, seed=42)

    train_images_reduced = data["train_images_reduced"]
    train_labels         = data["train_labels"]
    test_image_reduced   = data["test_image_reduced"]
    test_labels          = data["test_labels"]

    Y_train = np_utils.to_categorical(train_labels, num_classes=10)
    Y_train_extended = build_gaussian_targets(train_labels)
    print(f"  Train: {train_images_reduced.shape}  Test: {test_image_reduced.shape}")

    # ------------------------------------------------------------------ #
    # 2. Propagation kernels (Eq. 1 of the paper)                         #
    # ------------------------------------------------------------------ #
    N          = DIM * DIM           # 196 neurons
    layer_dist = 40e-6               # 40 µm between layers
    output_dist = 50e-6              # 50 µm to output plane

    p_shifted        = propagation_kernel(N, META_ATOM_PITCH, layer_dist)
    p_shifted_output = propagation_kernel(N, META_ATOM_PITCH, output_dist)

    # ------------------------------------------------------------------ #
    # 3. Train                                                             #
    # ------------------------------------------------------------------ #
    print("Training ONN …")
    results = train_onn(
        train_images_reduced = train_images_reduced,
        Y_train_extended     = Y_train_extended,
        Y_train              = Y_train,
        train_labels         = train_labels,
        test_image_reduced   = test_image_reduced,
        test_labels          = test_labels,
        p_shifted            = p_shifted,
        p_shifted_output     = p_shifted_output,
        epoch_max            = 50,
        batch_size           = 64,
        alpha                = 0.065,
        verbose              = True,
        save_dir             = OUTPUT_DIR / "saved_phases",
    )

    Ph1 = results["Phasetarget_1"]
    Ph2 = results["Phasetarget_2"]
    Ph3 = results["Phasetarget_3"]
    print(f"\nFinal test accuracy: {results['test_acc_history'][-1]:.1f}%")

    # ------------------------------------------------------------------ #
    # 4. Plots                                                             #
    # ------------------------------------------------------------------ #
    plot_training_curves(
        results["cost_history"],
        results["test_acc_history"],
        save_path=OUTPUT_DIR / "training_curves.png",
    )
    print(f"Saved training curve → {OUTPUT_DIR / 'training_curves.png'}")

    visualize_phase_profiles(Ph1, Ph2, Ph3, save_dir=OUTPUT_DIR)
    print(f"Saved phase profiles → {OUTPUT_DIR}")

    # Confusion matrix on full test set
    import numpy as np
    Phi1 = np.exp(1j * Ph1)
    Phi2 = np.exp(1j * Ph2)
    Phi3 = np.exp(1j * Ph3)

    pred_labels = []
    for img in test_image_reduced:
        Eout, _, _ = ONN_forward(img, Phi1, Phi2, Phi3, p_shifted, p_shifted_output)
        intensity = np.real(Eout * np.conj(Eout))
        pred_labels.append(np.argmax(_detector_sums(intensity)))

    class_names = [str(i) for i in range(10)]
    plot_confusion_matrix(
        test_labels,
        pred_labels,
        classes=class_names,
        normalize=True,
        save_path=OUTPUT_DIR / "confusion_matrix.png",
    )
    print(f"Saved confusion matrix → {OUTPUT_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
