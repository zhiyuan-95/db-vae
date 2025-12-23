# Mitigating Algorithmic Bias with DB-VAE (Reproduction Study)

This repository implements and analyzes two models for facial detection:
1. **A baseline CNN classifier**
2. **A DB-VAE (Debiasing Variational Autoencoder)** based on the AAAI/ACM AIES paper  
   *“Uncovering and Mitigating Algorithmic Bias through Learned Latent Structure”*  
   (https://dl.acm.org/doi/pdf/10.1145/3306618.3314243)

Results Overview 

This project presents a comprehensive empirical evaluation of the Debiasing Variational Autoencoder (DB-VAE) for mitigating demographic performance disparities in face detection under imbalanced training data. Using a controlled experimental framework, we compare DB-VAE against a standard convolutional neural network (CNN) baseline across extensive hyperparameter sweeps and multi-seed trials. Model performance is evaluated jointly in terms of overall validation accuracy and demographic disparity across race–gender subgroups.

Across more than 150 controlled training runs, DB-VAE consistently achieves higher overall validation performance than the standard CNN baseline. However, these accuracy gains do not correspond to a statistically significant reduction in demographic performance disparity when bias is measured as the gap between the best- and worst-performing demographic groups. Instead, group-wise analysis reveals that DB-VAE improves performance uniformly across all demographic subgroups, resulting in largely unchanged disparity metrics despite substantial absolute gains in accuracy.

Further analysis of the accuracy–bias relationship reveals a qualitative difference in training dynamics between the two models. While the standard CNN exhibits a strong positive correlation between overall accuracy and demographic disparity—indicating that performance improvements primarily benefit dominant subpopulations—DB-VAE substantially weakens this coupling. This decoupling suggests that DB-VAE modifies representation learning in a way that distributes performance gains more evenly across latent subpopulations, even though it does not explicitly target demographic parity.

Taken together, these results indicate that DB-VAE functions primarily as a representation-learning regularizer rather than a direct demographic bias mitigation mechanism. The findings highlight an important limitation of latent-space debiasing approaches: improvements in global performance and fairness-oriented training dynamics do not necessarily translate into reductions in group-defined disparity metrics unless the debiasing objective is explicitly aligned with the fairness evaluation criterion.

[report for dbvae.docx](https://github.com/user-attachments/files/24318470/report.for.dbvae.docx)

