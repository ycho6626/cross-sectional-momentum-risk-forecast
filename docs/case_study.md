# Momenta Research: forecasting momentum risk

**Personal research project by [ycho6626](https://github.com/ycho6626) | Mathematics · Statistical validation · Research engineering**

**Question and contribution.** Can cross-sectional structure improve momentum-risk forecasts beyond established predictors? My contribution is an end-to-end investigation: mathematical redundancy checks, leakage-controlled forecasting and portfolio accounting, and a placebo diagnosis that changed the initial comparison design before holdout evaluation.

**Primary conclusion.** Adding the proposed features lowered holdout forecast loss by 4.62% for raw momentum and 4.09% for residual momentum, but did not pass the pre-specified rule: Holm p = 0.050728, with both bootstrap intervals including zero. Improved net trading performance was not established.

## Research judgment: a failed comparison design

Under a shared regularization penalty, adding four or ten noise predictors lowered MSE in 78% or 82% of 200 RAW replications per block size. Added noise changed shrinkage of the baseline, making model expansion appear informative. Before unblinding, the design was amended to sequential block penalties, exact exclusion and training-standardized clipping. Across 800 amended placebo replications, all four series/block-size mean incremental R² values were negative. Superseded results remain available.

## Mathematics and implementation

A factor–residual variance decomposition motivates separating common exposure from residual dependence. Identities rule out redundant information: sign entropy is determined by breadth; average correlation follows from standardized cross-sectional means; and a rank transport gap reduces to twice one minus Spearman correlation under stated assumptions. Property tests enforce their domains.

The Python/DuckDB/Polars pipeline constructs monthly 1,000-stock US equity universes and raw/residual long–short books with permanent identifiers, terminal returns and a next-close position ledger. Annual walk-forward fits use completed targets and purged inner validation, refitting learned transformations within training. Secondary models use sliced-Wasserstein embeddings, k-means, PCA and forward-filtered Gaussian HMMs.

## Three stages, distinct conclusions

The locked historical holdout contains 127 month-end origins, December 2015–June 2026. In the registered secondary comparison, established predictors (L1) reduce holdout MSE relative to own-volatility HAR by 18.51% for RAW and 17.63% for RES, with both improvement intervals above zero and Holm p below 0.0001. In design, RAW MSE rose 12.19%; RES's 1.78% improvement had an interval including zero.

The proposed additions (L2) are tested against L1, producing the unsupported primary result above. In post-unblinding descriptive checks, their gains concentrate in 2020 for RAW, especially October, and 2026 for RES; excluding those respective years leaves L2 slightly worse. These checks do not replace the full-sample test.

At 10 bp per traded notional, L1 + L2's net Lo Sharpe is 0.1240 versus 0.3013 for historical-volatility scaling in RAW, and −0.1524 versus −0.0384 in RES. Better forecast point estimates do not establish profitable alpha.

## Reproducibility and limits

The full study passed 490 tests and reproduced 87 artifacts exactly in 60 minutes 36 seconds, excluding downloads and robustness. Public methods, tests, a synthetic demonstration and aggregate evidence require no licensed observations. The study was internally frozen, not publicly preregistered; final-vintage inputs, full-sample data-quality review, capitalization backfills and few crashes limit interpretation.

**Repository:** [github.com/ycho6626/cross-sectional-momentum-risk-forecast](https://github.com/ycho6626/cross-sectional-momentum-risk-forecast)
