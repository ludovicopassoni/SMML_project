# Kernels and Random Feature Maps

This project explores the connection between kernel methods and explicit feature maps, implemented from scratch as part of the *Statistical Methods for Machine Learning* course (A.Y. 2025/26, Assignment 7). Three classifiers are compared on the same datasets: a linear SVM trained via subgradient descent, an exact RBF kernel SVM trained via Sequential Minimal Optimization (SMO), and a linear model trained on top of Random Fourier Features (Rahimi & Recht, 2007), an explicit finite-dimensional approximation of the RBF kernel. The goal is to empirically show where a purely linear model fails, how the kernel trick recovers non-linear decision boundaries, and how random feature maps approximate the exact kernel at a fraction of the computational cost.

## Repository Structure

```
experiments/
├──common.py                        #shared functions (dataset loading, plotting, gamma_list)
├──feature_map_comparison.py        #RFF vs polynomial feature map
├──gamma_effect.py                  #effect of RBF bandwidth
├──run_experiment.py                #comparison between linear vs kernel vs RFF 
└──runtime_accuracy.py              #runtime vs accuracy trade-off

notebook/

plots/                              #figures
├──real_gamma_effect.png
├──real_rff_comparison.png
├──real_runtime_accuracy.png
├──synthetic_gamma_effect.png
├──synthetic_rff_comparison.png
└──synthetic_runtime_accuracy.png

report/                             #LaTeX report and final PDF

src/
├──data.py                          #synthetic (XOR) and real-world (digits parity) datasets
├──kernels.py                       #RBF kernel + median heuristic for gamma
├──linear_model.py                  #linear svm (subgradiant descent)
├──polynomial_features.py           #explicit polynomial feature map
├──random_features.py               #Random Fourier Features
└──svm_smo.py                       #kernel SVM via SMO

README.md

requirements.txt
```

## Environment Setup

```bash
conda create -n nome_ambiente python=3.11.13
conda activate nome_ambiente
pip install -r requirements.txt
```

## Running the Experiments

```bash
python -m experiments.run_experiment                # linear SVM vs kernel SVM vs RFF, produces rff_comparison plots
python -m experiments.runtime_accuracy              # runtime vs accuracy trade-off across models
python -m experiments.gamma_effect                  # effect of RBF bandwidth (gamma) on performance
python -m experiments.feature_map_comparison        # RFF vs explicit polynomial feature map
```

## Key Findings

- **Linear vs. Kernel vs. Random Features**: on the synthetic (XOR) dataset, the linear SVM baseline reaches only 59.2% test accuracy, while the exact RBF kernel SVM reaches 97.5% and RFF (best D) reaches 99.2% — confirming that a linear model cannot solve a non-linearly separable problem, while both kernel methods can. On the real-world dataset (digits parity), the gap is smaller but still clear: 89.3% (linear) vs. 95.6% (kernel) vs. 94.8% (RFF).
- **Effect of the RBF bandwidth (gamma)**: too small a gamma makes the kernel degenerate (50.0% accuracy, no better than chance, on the synthetic dataset); too large a gamma leads to overfitting, most visibly on the real dataset, where test accuracy peaks at 98.3% before dropping as gamma increases further while training accuracy keeps rising to 100%.
- **RFF vs. explicit polynomial features**: the polynomial feature map is exact but its dimensionality is dictated by combinatorics, not choice — 6 features (degree 2) suffice on the 2D synthetic dataset, but the 64-dimensional real dataset requires 2,145 (degree 2) to 47,905 (degree 3) features, compared to a few hundred freely-chosen random features for RFF.
- **Runtime vs. accuracy trade-off**: on the real dataset, the exact kernel SVM takes 15.6s to train, while RFF with D=1000 achieves comparable accuracy (94.1% vs. 95.6%) in just 1.3s — roughly a 12x speed-up, illustrating the practical advantage of random feature approximations at scale.