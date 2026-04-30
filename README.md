# ECC Mix Design Optimization and Uncertainty Quantification

Note: This project is currently under a strict pre-publication embargo by the lab, so I am limited to discussing the conceptual framework. More details may be available upon request. Thank you for understanding.


This repository outlines a machine learning framework for the inverse design of Engineered Cementitious Composites (ECC). By combining ensemble surrogate models with evolutionary algorithms, this project identifies cost-optimal ECC mix designs that meet specific tensile stress and strain requirements. 

A core focus of this project is quantifying the uncertainty inherent in macroscopic concrete datasets, separating the model's knowledge gaps from physical lab variance.

## 1. Dataset and Feature Engineering
Standard mixture proportions are often insufficient for predicting the complex ductility of ECC. We engineered 37 features to better represent the physical interactions:
* Basic ingredients and standard ratios (W/C, W/B).
* Data-driven volumetric metrics (paste volume, SCM ratios).
* Micromechanical proxies based on the theoretical frameworks of Li (2003), such as fiber surface area and calculated crack-bridging indices.

## 2. Surrogate Modeling
We trained tree-based ensemble models (Extra Trees, Random Forest, Gradient Boosting) to predict ultimate stress and tensile strain. We evaluated the models on two distinct dataset formulations:

* **Expanded Dataset (560 specimens):** Includes laboratory replicates to capture localized variance. Yields an R-squared of 0.84 for Stress and 0.64 for Strain.
* **Aggregated Dataset (282 unique mixes):** Averages replicates to completely prevent data leakage and test true generalization. Yields a highly conservative baseline (Strain R-squared around 0.59), which we use as the foundation for our uncertainty quantification.

## 3. Inverse Design Methodologies
We approach the inverse design problem through two parallel methods:

* **Ranged Approach (Probabilistic):** Uses random sampling and local refinement to identify global "safe zones." This categorizes feasibility into different budget and premium cost tiers.
* **Discrete Approach (NSGA-II):** Uses a genetic algorithm to search the 17-dimensional feature space. It mathematically defines the Pareto frontier to generate exact, cost-minimized mix recipes.

## 4. Uncertainty Quantification (UQ)
Because our dataset lacks microscopic processing variables (like fiber dispersion or mixing speed), predicting strain carries significant uncertainty. We address this in two steps:

* **Epistemic Uncertainty (Conformal Quantile Regression):** Variance decomposition shows that roughly 68% of our prediction variance is epistemic (due to data sparsity). We use CQR to generate mathematically guaranteed 80%, 90%, and 95% prediction intervals, providing honest bounds instead of single point predictions.
* **Aleatoric Robustness (Monte Carlo Simulation):** To test if the NSGA-II recipes are physically viable, we run them through a Monte Carlo simulator. By injecting a 5% Gaussian noise to simulate human batching errors in the lab, we measure how many virtual batches fail the strain targets, outputting a real-world lab reliability score.

