# Debiasing Face Detection with DB-VAE
This repository implements and empirically evaluates DB-VAE (Debiasing Variational Autoencoder) for mitigating algorithmic bias in face detection, based on:
Amini et al., “Uncovering and Mitigating Algorithmic Bias through Learned Latent Structure” (AAAI 2019)
We compare a standard CNN baseline against DB-VAE under large-scale controlled experiments to study accuracy–bias tradeoffs, group-wise performance, and training dynamics.

## 🔍 Motivation
Deep learning models trained on imbalanced datasets often exhibit unequal performance across demographic groups.
DB-VAE proposes a latent-space–driven resampling mechanism that increases the sampling probability of under-represented examples without explicitly using demographic labels.
#### This project investigates:

Does DB-VAE reduce demographic performance disparities?

How does DB-VAE affect overall validation accuracy?

How are accuracy improvements distributed across demographic groups?

## 🧠 Method Overview
### Baseline model:
**Standard CNN**(Binary face / non-face classifier) which is trained with cross-entropy loss and serves as a reference model for accuracy and bias

### DB-VAE model:
**DB-VAE model** extends a standard CNN with a variational autoencoder and an adaptive resampling strategy:

1. Learns a latent representation of training data

2. Estimates latent-space density via histograms

3. Upsamples rare latent regions during training

4. degree of debiasing is controlled by a smoothing (debiasing) parameter α

As α increases:

α → 0: uniform latent sampling (strong debiasing)

α → ∞: standard random sampling (no debiasing)

## 📊 Evaluation Metrics

We evaluate models using both performance and fairness metrics:

**val**(Overall Validation Accuracy): Mean validation accuracy across all demographic groups.

**dbval**(Demographic Bias Metric):  Measures performance disparity across groups using group-wise validation losses:

  Higher dbval ⇒ larger demographic disparity

  Lower dbval ⇒ more uniform performance

This metric does not assume a predefined majority or minority group, making it robust to dataset composition assumptions.

## 🧪 Experimental Design
### Experiment 1 — Global Performance

100+ hyperparameter configurations per model

DB-VAE evaluated at α ∈ {0.6, 1.0, 1.4}, latent space = 100, bins = 10

3 seeds per configuration

Early stopping with patience = 3

### Goal:
Assess whether DB-VAE improves accuracy and/or reduces bias at scale.

### Result:
**Bias metric (dbval) - From the dbval-at-best-epoch boxplot:**

•	The DB-VAE variants (smoothing = 0.6, 1.0, 1.4) do not show a lower median dbval than the standard CNN.

•	The interquartile ranges (IQRs) of dbval for DB-VAE and the baseline overlap substantially. DB-VAE also exhibits similar or larger variances in dbval, with comparable high-end outliers.

<img width="545" height="417" alt="image" src="https://github.com/user-attachments/assets/8b107d6f-99f5-44d4-a944-65e50b4018f6" />

**Overall performance (Val) - From the(figure 2) Val-at-best-epoch boxplot:**

•	All DB-VAE variants consistently show higher median validation performance than the standard CNN.

•	The entire distribution of DB-VAE validation scores is shifted upward. Higher smoothing rates tend to correlate with slightly higher median validation performance.

<img width="553" height="416" alt="image" src="https://github.com/user-attachments/assets/e90cbe90-3479-46da-8959-c2acac65a6d7" />




### Experiment 2 — Group-wise Analysis

Top-10 configurations from Experiment 1 per model

5 independent seeds per configuration

Group-wise validation accuracy recorded for:

White Male (WM),White Female (WF),Black Male (BM),Black Female (BF)

### Goal:
Determine whether DB-VAE disproportionately benefits under-represented groups.

### Result:
**Group-wise Validation Performance:**

DB-VAE improves median validation performance across all demographic groups. For the WM group, median performance increases from 0.616 with standard CNN to average of 0.775, while WF, BM, and BF groups experience median increases of Δ_WF(0.06), Δ_BM(0.03), and Δ_BF(0.10), respectively. Importantly, the magnitude of improvement is comparable across groups, with no demographic group exhibiting a consistently larger gain.
<img width="1155" height="861" alt="image" src="https://github.com/user-attachments/assets/75e45289-3300-44f9-b60d-b0a0691f4be0" />


**Bias Metric (dbval)**

Despite improvements in overall and group-wise performance, DB-VAE does not reduce demographic disparity as measured by dbval. Median dbval values remain comparable across the baseline and DB-VAE models, with overlapping interquartile ranges across all smoothing configurations.

<img width="500" height="381" alt="image" src="https://github.com/user-attachments/assets/83073aac-c7fd-4d87-90b0-ccf2f2d447b2" />


**Accuracy–Bias Tradeoff Analysis**

Scatter plot below illustrates the relationship between overall validation performance and demographic disparity. For the standard CNN baseline, we observe a strong positive correlation between val and dbval, indicating that improvements in overall performance are accompanied by increased demographic disparity. This suggests that accuracy gains are primarily driven by dominant subpopulations. In contrast, DB-VAE models exhibit a substantially weaker relationship between val and dbval, implying that performance improvements are more evenly distributed across latent subpopulations. Although DB-VAE does not consistently reduce demographic disparity, it effectively decouples accuracy improvement from disparity amplification, highlighting a qualitative difference in training dynamics.

<img width="578" height="482" alt="image" src="https://github.com/user-attachments/assets/e64c6b34-b093-4390-8b79-9835301b0692" />

## 📈 Key Findings

DB-VAE consistently improves overall validation accuracy

No clear reduction in demographic disparity (dbval) under the evaluated settings

Performance gains are comparable across demographic groups

DB-VAE weakens the coupling between accuracy improvement and bias amplification observed in standard CNNs

### Interpretation:
Under the tested configurations, DB-VAE primarily acts as a representation-learning regularizer rather than an explicit demographic bias mitigation method.

## ⚠️ Experimental Limitations and Discussion
### Limited Hyperparameter Coverage

Due to limited computational resources, several DB-VAE hyperparameters were fixed during experimentation:

Latent dimensionality: 100

Number of histogram bins per latent dimension: 10

Smoothing (debiasing) parameter: α ∈ {0.4, 0.6, 1.0}

While these settings are sufficient to reproduce the qualitative behavior of DB-VAE and to enable controlled comparisons with a standard CNN baseline, they do not exhaustively explore the DB-VAE design space. Alternative choices—such as larger latent spaces, finer histogram resolutions, or different smoothing schedules—may yield different or improved bias–accuracy tradeoffs.

### Latent-Space Bias vs. Demographic Bias

DB-VAE is explicitly designed to mitigate imbalance in the learned latent space, rather than directly optimizing for fairness across predefined demographic groups. Consequently, reductions in latent rarity do not necessarily translate into immediate reductions in demographic performance disparity under fixed evaluation metrics.

Importantly, as overall representation quality and classification accuracy increase, demographic performance gaps may shrink indirectly, even in the absence of explicit demographic supervision. Under higher-capacity latent representations or at different operating points, accuracy gains may propagate unevenly across subpopulations, potentially leading to reduced demographic disparity.

The present experiments do not rule out this possibility; rather, they demonstrate that under the tested configurations, DB-VAE primarily improves global performance while leaving demographic disparity largely unchanged.

### Scope of Conclusions

Accordingly, the conclusions of this work should be interpreted with the following scope:

1. DB-VAE reliably improves overall validation accuracy under constrained computational settings

2. Under the chosen hyperparameters and fairness metric, no clear demographic bias reduction is observed

3. These findings do not constitute a negative result for DB-VAE in general, but instead highlight the importance of: broader hyperparameter exploration, sufficient model capacity, calignment between debiasing mechanisms and evaluation metrics

This work is intended as an empirical study under constrained resources, rather than a definitive evaluation of DB-VAE’s fairness potential.

## 🛠️ Tools & Stack

Python, TensorFlow

NumPy, Pandas

Comet ML for experiment tracking

Controlled multi-seed experimentation

Custom fairness metrics


## 📚 References

Amini et al., Uncovering and Mitigating Algorithmic Bias through Learned Latent Structure, AAAI 2019

CelebA Dataset

ImageNet (negative samples)

👤 Author

Zhiyuan Jin
M.S. Data Science, CUNY Graduate Center
GitHub: https://github.com/zhiyuan-95
