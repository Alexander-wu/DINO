<div align="center">

# DINO: Differential-Integral Neural Operator

**Physics-decomposed neural operator for stable long-term turbulence forecasting**

Hao Wu, Yuan Gao, Fan Xu, Fan Zhang, Qingsong Wen, Xiaomeng Huang, Xian Wu

<p>
  <a href="https://arxiv.org/"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?style=flat-square" alt="arXiv"></a>
  <a href="https://github.com/Alexander-wu/DINO"><img src="https://img.shields.io/badge/GitHub-Code-181717.svg?style=flat-square&logo=github" alt="GitHub"></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat-square&logo=pytorch" alt="PyTorch"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License"></a>
</p>

<img src="image/DINO_intro.png" width="95%"/>

</div>

---

## News

- **[2026.05]** Our paper has been accepted by **KDD 2026**.

---

## Overview

**DINO** is a neural operator designed for long-term forecasting of turbulent physical systems. Instead of using a monolithic backbone, DINO follows a **Physics-Decomposition** principle: it explicitly separates the evolution operator into a **global integral branch** and a **local differential branch**.

This design helps suppress long-horizon error accumulation and preserve physical structures such as fine-scale vortices and energy spectra.

**Key features**

- **Differential-Integral decomposition**: combines local differential modeling with global integral modeling.
- **Stable long-term rollout**: significantly reduces error accumulation in autoregressive forecasting.
- **Physical fidelity**: preserves vortex structures and matches the theoretical $k^{-3}$ enstrophy spectrum.
- **Strong generalization**: works across Kolmogorov flow, isotropic turbulence, Prometheus-T, and sparse ocean forecasting.

---

## Model Architecture

<div align="center">
  <img src="image/DINO.png" width="95%"/>
</div>

DINO consists of four main stages:

1. **Lifting operator**: maps the input field into a latent representation.
2. **Global corrector**: uses Transformer attention to capture non-local interactions.
3. **Local refiner**: uses constrained convolutions to model local differential dynamics.
4. **Residual prediction**: outputs the next-step forecast with a residual update.

---

## Benchmarks

<div align="center">
  <img src="image/dataset.png" width="95%"/>
</div>

We evaluate DINO on multiple challenging physical forecasting benchmarks:

| Dataset | Task | Evaluation focus |
| :--- | :--- | :--- |
| **2D Kolmogorov Flow** | Forced turbulence forecasting | Long-term stability and spectral fidelity |
| **2D Isotropic Turbulence** | Decaying turbulence forecasting | Physical dissipation and rollout accuracy |
| **Prometheus-T** | Fire dynamics forecasting | Multi-physics and OOD generalization |
| **Sparse Ocean Forecasting** | Forecasting from sparse observations | Robustness to incomplete real-world data |

---

## Main Results

### Quantitative performance

DINO achieves the best performance across all benchmarks, especially in long-term rollout settings.

| Model | Kolmogorov 1-step ↓ | Kolmogorov 99-step ↓ | Isotropic 19-step ↓ | Prometheus-T OOD ↓ |
| :--- | :---: | :---: | :---: | :---: |
| FNO | 0.0267 | 3.1284 | 1.9832 | 0.0506 |
| CNO | 0.0407 | 11.3015 | 1.5676 | 0.0749 |
| LSM | 0.0046 | 5.1127 | 2.0382 | 0.0456 |
| NMO | 0.0018 | 2.1923 | 0.1873 | 0.0441 |
| PDE-Refiner | 0.0021 | 1.9954 | 0.2103 | 0.0428 |
| U-Net | 0.0182 | 4.6647 | 0.6583 | 0.1067 |
| SimVP | 0.0019 | 5.0405 | 0.2231 | 0.0608 |
| **DINO** | **0.0002** | **0.5876** | **0.1110** | **0.0359** |
| **Improvement** | **88.89%** | **70.55%** | **40.74%** | **16.12%** |

### Qualitative results

<div align="center">
  <img src="image/figure_result_1.png" width="98%"/>
</div>

DINO maintains sharper and more physically consistent predictions in long-term Kolmogorov rollouts and out-of-distribution Prometheus-T forecasting.

### Comprehensive comparison

<div align="center">
  <img src="image/comprehensive_performance_enhanced.png" width="98%"/>
</div>

DINO consistently achieves better correlation, lower forecasting error, and stronger OOD robustness than existing neural operators and spatiotemporal models.

### Spectral fidelity

<div align="center">
  <img src="image/esd_new.png" width="98%"/>
</div>

DINO accurately reproduces the theoretical $k^{-3}$ scaling law, while baseline models suffer from over-smoothing or non-physical energy artifacts.

### Sparse observation robustness

<div align="center">
  <img src="image/sparse.png" width="98%"/>
</div>

Geo-DINO remains robust under sparse observational data and accurately reconstructs key ocean-current structures such as the Kuroshio Current.

---

## Ablation Study

| Variant | Kolmogorov 99-step ↓ | Isotropic 19-step ↓ | Prometheus-T OOD ↓ |
| :--- | :---: | :---: | :---: |
| **DINO** | **0.587** | **0.111** | **0.0359** |
| w/o Integral branch | 1.852 | 0.983 | 0.0891 |
| w/o Differential branch | 1.134 | 0.456 | 0.0624 |
| Standard CNN branch | 0.975 | 0.289 | 0.0517 |
| FNO integral branch | 0.821 | 0.215 | 0.0488 |

Both branches are necessary: the integral branch captures global interactions, while the differential branch preserves local high-frequency dynamics.

---

## Installation

> Code will be released soon. The following commands show the planned usage.

```bash
git clone https://github.com/Alexander-wu/DINO.git
cd DINO
conda create -n dino python=3.10 -y
conda activate dino
pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python train.py --config configs/kolmogorov.yaml
```

### Evaluation

```bash
python eval.py \
  --config configs/kolmogorov.yaml \
  --checkpoint checkpoints/dino_kolmogorov.pt \
  --rollout 99
```

### Visualization

```bash
python tools/plot_rollout.py
python tools/plot_spectrum.py
```

---

## Repository Structure

```text
DINO/
├── configs/              # Training and evaluation configs
├── data/                 # Dataset folder
├── dino/                 # Model and training code
├── tools/                # Visualization scripts
├── train.py
├── eval.py
└── requirements.txt
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{wu2026dino,
  title     = {DINO: Differential-Integral Neural Operator for Long-Term Turbulence Forecasting},
  author    = {Wu, Hao and Gao, Yuan and Xu, Fan and Zhang, Fan and Wen, Qingsong and Huang, Xiaomeng and Wu, Xian},
  booktitle = {Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2026}
}
```

---

## Contact

For questions or discussions, please open an issue in this repository.
