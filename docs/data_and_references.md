# Data and references

The research used pinned provider snapshots for security histories, factor returns, comparison portfolios and macroeconomic inputs. This repository publishes source code, tests using synthetic or hand-calculated inputs, derived statistics and figures. It contains no provider downloads, security-level panels, vendor return series or account-access information. The data commands fetch these inputs at run time with the user's own Sharadar subscription and FRED key, read from an ignored `.env` file; the French and JKP files are public downloads.

| Source | Role in the study |
|---|---|
| [Sharadar](https://sharadar.com/) | Daily equity prices, market capitalizations, security identifiers and corporate actions for universe construction, signals, portfolios and terminal-return handling. |
| [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | Daily and monthly Fama–French three-factor returns and risk-free rates; momentum references and decile portfolios for reconciliation, historical-volatility replication and baseline calibration. |
| [Jensen, Kelly and Pedersen Global Factor Data](https://jkpfactors.com/data) | US capped-value-weighted raw and residual momentum portfolios, used for reconciliation and long-history comparisons. |
| [FRED, Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org/) | VIXCLS, BAA10Y and T10Y3M for the macroeconomic robustness block; USREC for descriptive recession shading only. |

The provider snapshots were aligned by observation date. Publication lags and subsequent data revisions were not reconstructed. Some early capitalizations use later share counts as a disclosed approximation; terminal returns are partly imputed. These features make the study a final-vintage historical comparison. [Research design](research_design.md) states the resulting limits.

The following references supply the portfolio definitions, predictors, comparison strategies and interpretation used here. Their published results are not performance claims for this study.

- Jegadeesh, N., and S. Titman (1993). [Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x). *Journal of Finance*, 48(1), 65–91. Cross-sectional momentum.
- Blitz, D., J. Huij, and M. Martens (2011). [Residual momentum](https://repub.eur.nl/pub/22252/ResidualMomentum-2011.pdf). *Journal of Empirical Finance*, 18(3), 506–521. Residual-return ranking.
- Grundy, B. D., and J. S. Martin (2001). [Understanding the Nature of the Risks and the Source of the Rewards to Momentum Investing](https://doi.org/10.1093/rfs/14.1.29). *Review of Financial Studies*, 14(1), 29–78. Momentum's changing factor exposure.
- Cooper, M. J., R. C. Gutierrez Jr., and A. Hameed (2004). [Market States and Momentum](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2004.00665.x). *Journal of Finance*, 59, 1345–1365. Market-trend predictors.
- Barroso, P., and P. Santa-Clara (2015). [Momentum has its moments](https://novaresearch.unl.pt/en/publications/momentum-has-its-moments/). *Journal of Financial Economics*, 116(1), 111–120. Historical-volatility scaling.
- Daniel, K., and T. J. Moskowitz (2016). [Momentum crashes](https://www.nber.org/papers/w20439). *Journal of Financial Economics*, 122(2), 221–247. Market-state risk and the transferred dynamic baseline; the link provides the working-paper version.
- Stivers, C. T., and L. Sun (2010). [Cross-Sectional Return Dispersion and Time Variation in Value and Momentum Premiums](https://doi.org/10.1017/s0022109010000384). *Journal of Financial and Quantitative Analysis*, 45(4), 987–1014. Cross-sectional dispersion.
- Huang, S. (2022). [The Momentum Gap and Return Predictability](https://doi.org/10.1093/rfs/hhab093). *Review of Financial Studies*, 35(7), 3303–3336. Formation-return dispersion.
- Lou, D., and C. Polk (2022). [Comomentum: Inferring Arbitrage Activity from Return Correlations](https://doi.org/10.1093/rfs/hhab117). *Review of Financial Studies*, 35(7), 3272–3302. Residual-return correlation.
- Avramov, D., S. Cheng, and A. Hameed (2016). [Time-Varying Liquidity and Momentum Profits](https://doi.org/10.1017/S0022109016000764). *Journal of Financial and Quantitative Analysis*, 51(6), 1897–1923. Liquidity predictors.
- Bianchi, D., A. De Polis, and I. Petrella (2026). [Time-Varying Skewness and Momentum Crashes](https://doi.org/10.1093/rapstu/raag010). *Review of Asset Pricing Studies*, online publication. Score-driven scale and shape.
- Jensen, T. I., B. Kelly, and L. H. Pedersen (2023). [Is There a Replication Crisis in Finance?](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13249). *Journal of Finance*, 78(5), 2465–2518. Factor-data definitions and replication context.
- Bailey, D. H., and M. López de Prado (2014). [The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551). *Journal of Portfolio Management*, 40(5), 94–107. Descriptive multiple-strategy performance adjustment.

Mathematical identities and their domains are stated in [Methodology](../METHODOLOGY.md) and tested directly. The transport construction and filtered-state controls are specified there; geometric or statistical labels do not identify economic causes.
