# ECC Pipeline Comparison

This consolidated README compares the notebooks in this repository across forward prediction and inverse design. It also states which forward pipeline each inverse pipeline uses.

## Forward Pipeline Comparison

| Notebook | Architecture | Main method | Stress MAE | Stress R2 | Stress Cov80 | Strain MAE | Strain R2 | Strain Cov80 | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `ECC.ipynb` | Baseline tree pipeline | ExtraTrees/GBR plus CQR | 0.702 | 0.6268 | 0.837 at 80 percent | 0.01190 | 0.3350 | 0.776 at 80 percent | Honest GroupKFold baseline; shows leakage gap versus Random KFold |
| `backend_comparison.ipynb` | Backend benchmark | ET+GBR, CatBoost, NGBoost, LightGBM | not captured | not captured | not captured | 0.00733 best | 0.55553 best | 0.85507 CatBoost | CatBoost is best captured strain backend |
| `ECC_catboost_mondrian_cqr.ipynb` | Quantile CatBoost | CatBoost q10/q50/q90 plus Mondrian CQR | 0.5048 | 0.7400 | 0.895 | 0.0073 | 0.5555 | 0.855 | Strong simple production candidate |
| `ECC_Pipeline_with_PCA.ipynb` | CatBoost plus reduction study | Full 37 vs reduced features | 0.5090 | 0.7355 | 0.851 | 0.0075 | 0.5358 | 0.873 | Reduced features do not clearly beat all 37 |
| `ECC_Pipeline_with_Inverse.ipynb` | CatBoost plus inverse | CatBoost q10/q50/q90 plus Mondrian CQR | 0.5090 | 0.7355 | 0.851 | 0.0075 | 0.5358 | 0.873 | Early inverse integration |
| `ECC_Inverse_Enhanced_Pipeline.ipynb` | CatBoost plus enhanced inverse | CatBoost q10/q50/q90 plus fixes | 0.5048 | 0.7400 | 0.895 | 0.0073 | 0.5555 | 0.855 | Adds binder cap, fiber cost, uncertainty constraints |
| `ECC_Inverse_Pipeline_Fixed.ipynb` | Fixed CatBoost inverse | CatBoost q10/q50/q90 plus fixed inverse | 0.5048 | 0.7400 | 0.895 | 0.0073 | 0.5555 | 0.855 | Best standalone inverse notebook |
| `ECC_Pipeline_Clean.ipynb` | Clean CatBoost end-to-end | CatBoost plus V1-V5 inverse grid | 0.5013 | 0.7405 | 0.880 | 0.0076 | 0.5599 | 0.862 | Cleanest CatBoost workflow |
| `ECC_Pipeline_Combined.ipynb` | Combined inverse experiment | Clean CatBoost plus V4 Plus OOD | 0.5013 | 0.7405 | 0.880 | 0.0076 | 0.5599 | 0.862 | Most complete inverse experiment |
| `ECC_FINAL_pipeline.ipynb` | Final MoE forward pipeline | Global ET plus sparse residual MoE plus Mondrian CQR | 0.4866 tuned | 0.7529 | not in summary | 0.00746 tuned | 0.5311 | not in summary | Best captured stress MAE |
| `mondrian_cqr_moe.ipynb` | MoE variants | Global model plus residual experts | 0.4881 best | not captured | 0.7971 best-config | 0.00746 | not captured | 0.8841-0.8877 | Strong stress MAE but coverage tradeoff |
| `tabpfn_moe_router.ipynb` | Router experiment | TabPFN class router plus experts | n/a | n/a | n/a | 0.0117 | n/a | n/a | Weaker strain pipeline; useful as routing experiment |

## Inverse Pipeline Comparison

| Notebook | Uses which forward pipeline? | Inverse architecture | Main result |
|---|---|---|---|
| `ECC.ipynb` | Its own ExtraTrees/GBR baseline forward model | Random search, refinement, Budget/Standard/Premium tiering, Monte Carlo validation | 8,367 candidates tiered; Budget about 168 USD/m3, Standard about 214 USD/m3, Premium about 289 USD/m3 |
| `ECC_catboost_mondrian_cqr.ipynb` | Its own CatBoost + Mondrian CQR forward model | Random search with cost-tier ranking | 24,069 feasible candidates from 100,000; tiers around 160-287 USD/m3 |
| `ECC_Pipeline_with_PCA.ipynb` | Its own CatBoost + Mondrian CQR forward model | Same tiered inverse search after PCA/reduction diagnostics | 24,766 feasible candidates from 100,000; tiers around 160-287 USD/m3 |
| `ECC_Pipeline_with_Inverse.ipynb` | Its own CatBoost + Mondrian CQR forward model | Physics-constrained Pareto search plus tiered inverse search | Pareto front of 200 solutions; constrained cost about 97.5-227.6 USD/m3 |
| `ECC_Inverse_Enhanced_Pipeline.ipynb` | Its own CatBoost + Mondrian CQR forward model | Binder cap, fiber cost, uncertainty-aware NSGA-II, V1-V5 grid | Early constrained Pareto front of 200 solutions; V1-V5 run across Easy/Medium/Hard |
| `ECC_Inverse_Fixes.ipynb` | Existing main inverse notebook state | Patch: data bounds, deduplicated OOD, warm-start NSGA-II | Re-runs V1-V5 after fixes; intended as a drop-in correction notebook |
| `ECC_Inverse_Pipeline_Fixed.ipynb` | Its own CatBoost + Mondrian CQR forward model | Fixed V1-V5 inverse grid plus distribution recommendation | Medium V4 produced 199 feasible solutions, cost 227.842-275.953 USD/m3, 100 percent in-distribution |
| `ECC_Patch_ECC_Identity_and_OOD_1.ipynb` | Existing clean/combined inverse state | Patch: ECC PSH recalibration and OOD as 4th objective | Same patch in combined notebook gives 200 Easy and 200 Medium V4 Plus OOD solutions, no Hard solutions |
| `ECC_Pipeline_Clean.ipynb` | Its own clean CatBoost + Mondrian CQR forward model | V1-V5 grid, distribution recommendation, diagnostics | Easy V4 has 200 solutions with 96.5 percent in-distribution; recommendation found 44 certifying mixes |
| `ECC_Pipeline_Combined.ipynb` | Its own clean CatBoost + Mondrian CQR forward model | V1-V5 plus V4 Plus OOD, anchors, recommendation | V4 Plus OOD tightens Easy cost range to 235.79-315.09 USD/m3 and stress width to 1.35 |

## Architecture Takeaways

| Decision | Evidence | Recommendation |
|---|---|---|
| Use grouped evaluation | `ECC.ipynb` shows Random KFold greatly overstates performance | Keep GroupKFold or group-level aggregation for all claims |
| Prefer CatBoost for simple forward modeling | CatBoost has stress MAE around 0.50 and strain MAE around 0.0073 with calibrated intervals | Use `ECC_catboost_mondrian_cqr.ipynb` or clean derivatives for forward prediction |
| Use MoE when stress MAE matters most | `ECC_FINAL_pipeline.ipynb` reaches tuned stress MAE about 0.4866 | Use final/MoE pipeline for stress-focused reporting |
| Keep all 37 features by default | PCA/reduced variants do not clearly dominate | Use all 37 unless model simplicity is the main objective |
| Use fixed inverse pipeline for recommendations | Data-derived bounds, OOD reference, warm-starting, and fiber cost make designs more realistic | Use `ECC_Inverse_Pipeline_Fixed.ipynb` for standalone inverse design |
| Treat Hard inverse targets carefully | Combined V4 Plus OOD found no feasible Hard solutions | Report Hard target as outside current feasible/reliable design region |

## Best Notebook by Use Case

| Use case | Best notebook |
|---|---|
| Clean forward CatBoost baseline | `ECC_catboost_mondrian_cqr.ipynb` |
| Best stress-forward performance | `ECC_FINAL_pipeline.ipynb` |
| Architecture/ablation research | `other_implementations/mondrian_cqr_moe.ipynb` |
| Backend comparison | `other_implementations/backend_comparison.ipynb` |
| Standalone inverse recommendation | `ECC_Inverse_Pipeline_Fixed.ipynb` |
| Most complete inverse experiment | `ECC_Pipeline_Combined.ipynb` |

