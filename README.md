# Debiasing Face Detection with DB-VAE

This repository implements and empirically evaluates DB-VAE (Debiasing Variational Autoencoder) for mitigating algorithmic bias in face detection, based on:

Amini et al., “Uncovering and Mitigating Algorithmic Bias through Learned Latent Structure” (AAAI 2019)

We compare a standard CNN baseline against DB-VAE under large-scale controlled experiments to study accuracy–bias tradeoffs, group-wise performance, and training dynamics.

## 🔍 Motivation

Deep learning models trained on imbalanced datasets often exhibit unequal performance across demographic groups.
DB-VAE proposes a latent-space–driven resampling mechanism that increases the sampling probability of under-represented examples without explicitly using demographic labels.

This project investigates:

Does DB-VAE reduce demographic performance disparities?

How does DB-VAE affect overall validation accuracy?

How are accuracy improvements distributed across demographic groups?

## 🧠 Method Overview
Baseline: Standard CNN

Binary face / non-face classifier

Trained with cross-entropy loss

Serves as a reference model for accuracy and bias

## DB-VAE

DB-VAE extends a standard CNN with a variational autoencoder and an adaptive resampling strategy:

Learns a latent representation of training data

Estimates latent-space density via histograms

Upsamples rare latent regions during training

Controlled by a smoothing (debiasing) parameter α

As α increases:

α → 0: uniform latent sampling (strong debiasing)

α → ∞: standard random sampling (no debiasing)

## 📊 Evaluation Metrics

We evaluate models using both performance and fairness metrics:

Overall Validation Accuracy (val)

Mean validation accuracy across all demographic groups.

Demographic Bias Metric (dbval)

Measures performance disparity across groups using group-wise validation losses:

Higher dbval ⇒ larger demographic disparity

Lower dbval ⇒ more uniform performance

This metric does not assume a predefined majority or minority group, making it robust to dataset composition assumptions.

## 🧪 Experimental Design
### Experiment 1 — Global Performance

100+ hyperparameter configurations per model

DB-VAE evaluated at α ∈ {0.6, 1.0, 1.4}

3 seeds per configuration

Early stopping with patience = 3

Goal:
Assess whether DB-VAE improves accuracy and/or reduces bias at scale.

result:
<img width="545" height="417" alt="image" src="https://github.com/user-attachments/assets/8b107d6f-99f5-44d4-a944-65e50b4018f6" />
<img width="553" height="416" alt="image" src="https://github.com/user-attachments/assets/e90cbe90-3479-46da-8959-c2acac65a6d7" />
Bias metric (dbval) - From the(figure 1) dbval-at-best-epoch boxplot:
•	The DB-VAE variants (smoothing = 0.6, 1.0, 1.4) do not show a lower median dbval than the standard CNN.
•	The interquartile ranges (IQRs) of dbval for DB-VAE and the baseline overlap substantially. DB-VAE also exhibits similar or larger variances in dbval, with comparable high-end outliers.
Overall performance (Val) - From the(figure 2) Val-at-best-epoch boxplot:
•	All DB-VAE variants consistently show higher median validation performance than the standard CNN.
•	The entire distribution of DB-VAE validation scores is shifted upward. Higher smoothing rates tend to correlate with slightly higher median validation performance.


### Experiment 2 — Group-wise Analysis

Top-10 configurations from Experiment 1 per model

5 independent seeds per configuration

Group-wise validation accuracy recorded for:

White Male (WM)

White Female (WF)

Black Male (BM)

Black Female (BF)

## Goal:
Determine whether DB-VAE disproportionately benefits under-represented groups.

## 📈 Key Findings

DB-VAE consistently improves overall validation accuracy

No significant reduction in demographic disparity (dbval)

Performance improvements are uniform across demographic groups

DB-VAE weakens the coupling between accuracy gains and bias amplification observed in standard CNNs

## Interpretation:
DB-VAE acts primarily as a representation-learning regularizer, rather than an explicit demographic bias mitigation method under the chosen metric.

📂 Repository Structure
.
├── data/                  # Dataset loaders and preprocessing
├── models/
│   ├── cnn.py              # Baseline CNN
│   └── dbvae.py            # DB-VAE implementation
├── training/
│   ├── train_cnn.py
│   └── train_dbvae.py
├── evaluation/
│   ├── metrics.py          # val, dbval, group-wise metrics
│   └── analysis.py
├── experiments/
│   ├── experiment1.py
│   └── experiment2.py
├── notebooks/
│   └── analysis.ipynb
├── README.md

## 🚀 Running Experiments
Train baseline CNN
python training/train_cnn.py --config configs/cnn.yaml

Train DB-VAE
python training/train_dbvae.py --config configs/dbvae.yaml

Run evaluation
python evaluation/analysis.py

## 🛠️ Tools & Stack

Python, TensorFlow

NumPy, Pandas

Comet ML for experiment tracking

Controlled multi-seed experimentation

Custom fairness metrics

## 📌 Notes on Fairness Evaluation

This work highlights an important lesson in responsible ML:

Fairness mechanisms must align with fairness metrics.

Latent-space debiasing does not necessarily translate into improved demographic parity unless demographic structure is explicitly reflected in the learned representation.

## 📚 References

Amini et al., Uncovering and Mitigating Algorithmic Bias through Learned Latent Structure, AAAI 2019

CelebA Dataset

ImageNet (negative samples)

👤 Author

Zhiyuan Jin
M.S. Data Science, CUNY Graduate Center
GitHub: https://github.com/zhiyuan-95
