# Mitigating Algorithmic Bias with DB-VAE (Reproduction Study)

This repository implements and analyzes two models for facial detection:
1. **A baseline CNN classifier**
2. **A DB-VAE (Debiasing Variational Autoencoder)** based on the AAAI/ACM AIES paper  
   *“Uncovering and Mitigating Algorithmic Bias through Learned Latent Structure”*  
   (https://dl.acm.org/doi/pdf/10.1145/3306618.3314243)

The goal of the project is to compare the standard classifier with the DB-VAE to evaluate:
- Fairness across demographic groups
- Bias metrics (dbval)
- Validation performance
- Stability and behavior under different hyperparameter settings

All experiments are tracked using **Comet ML**, enabling systematic comparison across trials.

---

## 📘 Background

The AAAI/ACM AIES paper proposes a debiasing method that:
- Learns latent variables using a variational autoencoder (VAE)
- Detects under-represented regions of the latent space
- Adaptively **resamples rarer examples more often**
- Improves fairness without manual annotations

This repository reproduces the key components:
- Encoder + decoder + supervised head (DB-VAE)
- Latent-space histogram estimation
- Adaptive reweighting using resampling probabilities
- Fairness evaluation using dbval (max difference between group means)

---

## 🧠 Models

### **1. Baseline CNN Classifier**
A standard convolutional neural network:
- 4 convolution layers with ReLU + batch norm  
- 2 fully connected layers  
- Sigmoid output for binary classification  
- Trained on CelebA (faces) + ImageNet (non-faces)

Tracked metrics:
- `loss`, `loss_smooth`, `loss_ema`
- `val`
- `dbval`
- `best_epoch`, `best_val`

---

### **2. DB-VAE (Debiasing Variational Autoencoder)**

Implements:
- Encoder: CNN + latent variable outputs (μ, σ)
- Decoder: reverse CNN for reconstruction
- Supervised head for the classification task
- Reparameterization trick
- Three-part loss:
  - Reconstruction  
  - KL divergence  
  - Supervised classification  

Key feature: **Adaptive resampling**  
The histogram of latent variables is used to compute sampling probability:

\[
W(z(x)) \propto \prod_i \frac{1}{\hat{Q_i}(z_i(x)) + \alpha}
\]

This increases exposure to rare features and reduces bias.

---

## 📊 Experiment Tracking with Comet ML

All models are logged using Comet ML:
- Live training curves  
- Hyperparameter tracking  
- Validation & fairness metrics  
- Comparison between experiments  
- Automatic selection of best trials per configuration  

The `comet.ipynb` notebook aggregates:
- Best experiment per hyperparameter
- Best validation score
- Best fairness score (dbval)
- Early stopping epoch  
- Distribution of results across trials

---

## 🧪 Results Summary

- The **DB-VAE reduces bias** (lower dbval) while maintaining competitive accuracy.
- Latent-space–based resampling increases diversity of positive (face) examples.
- The baseline CNN often overfits to over-represented facial features.
- Comet ML analysis shows consistent improvements across multiple trials.


