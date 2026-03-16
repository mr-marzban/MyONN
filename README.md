# MyONN — Integrated Photonic Neural Network based on Silicon Metalines

A whole-passive, fully-optical neural network implemented on a silicon-on-insulator (SOI)
chip using cascaded 1-D metasurfaces (*metalines*). The network performs matrix-vector
multiplications through free-space wave propagation and diffraction — entirely at the speed
of light, with near-zero power consumption during inference.

---

## Paper

> **Mahmood-Reza Marzban**, Sanaz Zarei, and Amin Khavasi,
> *"Integrated photonic neural network based on silicon metalines,"*
> **Optics Express 28(24), 36668 (2020).**
> https://doi.org/10.1364/OE.404386

```bibtex
@article{Marzban2020ONN,
  author    = {Marzban, Mahmood-Reza and Zarei, Sanaz and Khavasi, Amin},
  title     = {Integrated photonic neural network based on silicon metalines},
  journal   = {Optics Express},
  volume    = {28},
  number    = {24},
  pages     = {36668},
  year      = {2020},
  doi       = {10.1364/OE.404386},
  url       = {https://opg.optica.org/oe/fulltext.cfm?uri=oe-28-24-36668}
}
```

---

## Overview

The ONN consists of multiple **silicon metaline layers** — 1-D arrays of etched
rectangular slots in an SOI substrate. Each slot acts as a meta-atom that imposes
a programmable phase shift on the guided TE wave (0 – 2π, transmission > 0.96 at
λ = 1.55 µm). Cascading several metalines separated by free-space propagation
gaps implements the full matrix-vector multiplication required for deep neural
network inference.

Key highlights from the paper:

| Property | Value |
|---|---|
| Operating wavelength | 1.55 µm |
| Number of layers (full ONN) | 5 |
| Meta-atoms per layer | 784 |
| Total design parameters | 3920 |
| MNIST test accuracy | **88.8 %** |
| Inference latency | ~7.78 ps |
| Footprint | 400 × 800 µm² |
| Computational speed | 1.2 × 10¹⁶ MAC/s per layer |

---

## How It Works

### Architecture

Input images are encoded as the amplitude of guided optical pulses. The pulses
propagate through successive metaline layers, each imposing a learned phase profile.
Diffraction between layers mixes the field, performing the weighted summation of a
neural network layer. Ten photo-detectors at the output plane read the intensity in
ten designated regions — the digit with the highest intensity wins.

$$E^{\text{out}} = \left(\prod_{m=1}^{M} F^+ P^m F \Phi^{w^m}\right) E^{\text{in}}$$

where *F / F⁺* are the (I)DFT, *P^m* the free-space propagation kernel, and
*Φ^{w^m} = exp(j φ^{w^m})* the phase-modulation matrix of the *m*-th metaline.

### Training

Phase profiles are optimised offline (on a computer) using the **adjoint method**
paired with the **ADAM optimiser**. The adjoint technique reduces each gradient-descent
iteration to exactly **one forward pass** + **one backward adjoint pass**, making
training of large networks tractable.

$$\frac{dC}{d\bar{\phi}^{w^m}} = -4\,\mathrm{Re}\!\left\{ i\left(E^{m-1,k+}\right)^T \otimes \left(\prod_{m'=0}^{M-m} \Phi^{w^{M-m'}+} F^+ P^{M-m'+} F\right) a \right\}$$

Once training is finished the network is **entirely passive** — inference requires no power
beyond the optical input.

---

## Demo

### Training convergence

The cost decreases and test accuracy rises steadily over 50 epochs, reaching **~78 % on
the reduced 3-layer structure** (verified with Lumerical 2.5D FDTD) and **88.8 % on the
full 5-layer design**.

![Training Curves](sample_figs_demo/training_curves.png)
*Training cost (blue, left axis) and test accuracy (red, right axis) vs. epoch.*

---

### Optimised phase profiles

After training, each metaline layer holds a unique 14 × 14 phase pattern (0 – 2π).
These phase maps directly specify the slot lengths that need to be fabricated.

![Training detail](sample_figs_demo/training_curves_detail.png)
*Detailed view of cost/accuracy convergence during the early training stage.*

---

### Electric-field distribution (Lumerical MODE verification)

The figures below show the x–y electric-field distribution inside the simulated ONN
for three representative MNIST test digits. The field is steered towards the correct
detector at the right-hand output plane.

| Digit **5** | Digit **2** |
|:-----------:|:-----------:|
| ![Digit 5](sample_figs_demo/efield_digit_5.png) | ![Digit 2](sample_figs_demo/efield_digit_2.png) |

| Digit **6** | Digit **4** |
|:-----------:|:-----------:|
| ![Digit 6](sample_figs_demo/efield_digit_6.png) | ![Digit 4](sample_figs_demo/efield_digit_4.png) |

*Each panel: input digit (left) and x–y E-field through the 3-layer ONN (right).
The highlighted detector at the far right corresponds to the predicted class.*

---

### Confusion matrix

The normalised confusion matrix over 1000 MNIST test images confirms that the
most common confusion pairs are "4 ↔ 9" and "5 ↔ 3" — consistent with their
visual similarity and the findings reported in the paper (Fig. 6c).

![Confusion Matrix](sample_figs_demo/confusion_matrix.png)
*Normalised confusion matrix of the trained ONN on the MNIST test set.*

---

## Installation

```bash
git clone https://github.com/mmarzban3/MyONN.git
cd MyONN

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **Lumerical MODE** is required only for FDTD verification (Cells 2 / 24 in the
> original notebook). The analytical electromagnetic model and all training code
> run without it.

---

## Quick Start

```python
from src.physics import propagation_kernel, META_ATOM_PITCH
from src.data_utils import load_mnist_data, build_gaussian_targets, DIM
from src.training import train_onn
from keras.utils import np_utils

# Load data
data = load_mnist_data(train_sample=10000, test_sample=2000, seed=42)

# Propagation kernels
N = DIM * DIM
p        = propagation_kernel(N, META_ATOM_PITCH, 40e-6)   # inter-layer
p_out    = propagation_kernel(N, META_ATOM_PITCH, 50e-6)   # output plane

# Train
results = train_onn(
    train_images_reduced = data["train_images_reduced"],
    Y_train_extended     = build_gaussian_targets(data["train_labels"]),
    Y_train              = np_utils.to_categorical(data["train_labels"], 10),
    train_labels         = data["train_labels"],
    test_image_reduced   = data["test_image_reduced"],
    test_labels          = data["test_labels"],
    p_shifted            = p,
    p_shifted_output     = p_out,
    epoch_max            = 50,
)
print(f"Test accuracy: {results['test_acc_history'][-1]:.1f}%")
```

Or run the full example script:

```bash
python examples/train_onn.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Advantages Over Related Work

| Feature | This ONN | MZI-based [4–7] | D²NN [3] |
|---|---|---|---|
| Operation | Fully passive | Active (tunable) | Passive |
| Neurons | 3920 | < 1000 | ~200 000 |
| MNIST accuracy | 88.8 % | 85.8 % | 91.75 % |
| Footprint | 0.32 mm² | > 10 mm² | cm-scale |
| Latency | 7.78 ps | ~ns | ~ns |
| Alignment | On-chip (lithography) | On-chip | Manual (free-space) |

---

## Contact

- **Author**: Mahmood-Reza Marzban
- **Affiliation**: Department of Electrical Engineering, Sharif University of Technology
- **GitHub**: [@mmarzban3](https://github.com/mmarzban3)

---

## Acknowledgements

This work was supported by Sharif University of Technology and the
Iran National Science Foundation (grant 98012500).
