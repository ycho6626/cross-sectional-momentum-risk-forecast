# Methodology

This study tests whether cross-sectional dependence, distribution shape and latent persistence improve forecasts of momentum risk beyond established predictors. The mathematical motivation organizes measurable quantities and identifies redundancy. It does not assert a physical law of markets or claim new mathematical identities.

Write stock returns as $r=Bf+\varepsilon$. If factors and residuals are conditionally orthogonal, signed portfolio weights $a$ give

$$
\operatorname{Var}_t(a^\top r)
=a^\top B\Sigma_{f,t}B^\top a+a^\top\Sigma_{\varepsilon,t}a.
$$

This separates common-factor exposure from residual dependence. Residual-return sorting does not impose $a^\top B=0$. The distinction motivates comparing raw and residual momentum, following [Daniel and Moskowitz (2016)](https://www.nber.org/papers/w20439) and [Blitz, Huij and Martens (2011)](https://repub.eur.nl/pub/22252/ResidualMomentum-2011.pdf).

The primary universe selects the 1,000 largest eligible US common stocks by market capitalization each month. Eligibility includes a price of at least USD 5, 230 daily returns in the preceding 252 trading days, and an available formation history. Permanent security identifiers connect prices and corporate actions. Current delisted status and exchange do not determine historical membership.

At formation month $m$, RAW compounds returns over $m-11,\ldots,m-1$, excluding month $m$. RES regresses excess returns on an intercept and the three Fama–French factors over the 36-calendar-month window $m-35,\ldots,m$, requiring at least 24 complete observations. Its score is the mean divided by the sample standard deviation of residuals over $m-11,\ldots,m-1$; all eleven score months must be observed. RAW_c applies raw rankings to the RES-eligible universe for the market-state contrast.

Primary portfolios buy the top tercile and sell the bottom tercile. Within-leg weights are proportional to market capitalization capped at the universe's 80th percentile; each leg initially sums to one. Academic comparison series follow separate leg-compounding conventions. Forecast targets and trading results use a signed-position account with an explicit net asset value.

A decision at the close of trading day $t$ executes at the close of $t+1$. Existing positions earn that day's return before rebalancing. The forecast target therefore covers the 21 returns on $t+2,\ldots,t+22$:

$$
y_t=\log\!\left(\sum_{d=t+2}^{t+22}P_d^2\right),\qquad
P_d=\frac{\sum_i x_{i,d-1}r_{i,d}}{NAV_{d-1}}.
$$

Positions drift with security returns. Terminal proceeds move to cash until rebalancing. Transaction costs equal $c\sum_i|x_i^{new}-x_i^{pretrade}|/NAV$, so offsetting formation and scaling trades are charged once. The headline cost is 10 basis points per unit of traded notional. Returns use an excess-return convention that excludes collateral interest.

Volatility-targeting rules use a 12% annual target and cap the portfolio multiplier at two, corresponding to gross exposure of four times NAV at rebalancing. The historical-volatility benchmark estimates risk from 126 daily squared book returns. Forecast-based rules substitute their predicted variance, with monthly execution unchanged.

| Representation | Contents and interpretation |
|---|---|
| HAR | Own-book log realized variance at 5, 21 and 126 days. |
| Established predictors, L1 | HAR plus historical risk of current holdings, market beta, market volatility and trend, dispersion, momentum gap, comomentum, illiquidity and a score-driven scale–shape predictor. |
| Cross-sectional features, L2 | Nine features: Bowley skewness and Moors kurtosis for momentum and one-month returns; winner/loser within- and cross-correlations; rank persistence and lagged rank–return correlation. |
| Copula moments, CM | Six pairwise Spearman correlations among momentum, one-month return, beta and volatility. A secondary control adds six squares and fifteen cross-products. |
| Transport, OT | Four distances to training-fitted centroids in a sliced-quantile embedding of the four-dimensional ranked cloud. |
| Latent states | Filtered Gaussian hidden Markov probabilities, compared with emission-only probabilities to isolate persistence. |

Shape measures are robust proxies, not entropy estimates. L2 correlations use complete, positive-variance stocks over a common 126-day window; established comomentum uses 52 weekly factor-residual returns. Shared cross-sectional features use RAW rankings. The transport embedding uses 128 fixed directions and 101 quantiles; its centroids need not represent valid four-dimensional distribution barycenters. HMM inputs are three training-fitted principal components of L1+L2, with diagonal emission covariances, ten initializations and two states; three states are secondary. State features use forward-filtered probabilities or an emission-only control with fitted stationary priors. No smoothed probabilities enter forecasts.

Several identities constrain interpretation and are covered by property tests:

| Identity | Domain and consequence |
|---|---|
| $S=H((1+M)/2)$ | Binary signs, excluding zeros. Sign entropy contains no information beyond mean sign $M$. Median-relative breadth is zero for distinct observations, excluding the median when present. |
| $\overline{D}=(N-1)(\bar V-\bar C)/N+\operatorname{Var}_i(\bar r_i)$ | Balanced finite panels, population moments. Mean daily cross-sectional dispersion depends on mean individual variance, mean off-diagonal covariance and dispersion of individual means. |
| $\bar\rho=(n\operatorname{Var}(\bar z)-1)/(n-1)$ | At least two complete, nonconstant columns standardized with population moments. Cross-leg mean correlation similarly equals $\operatorname{Cov}(\bar z_W,\bar z_L)$. Correlation is not an identified interaction parameter. |
| $\sum_i c(q_i-\bar q)v_i=Nc\operatorname{Cov}(q,v)$ | Finite complete observations and population covariance; signal ranks $q$ use average ranks for ties. Values $v$ are raw returns or betas, not their rank transforms. |
| $C_{label}-W_2^2=2(\overline{xy}_{sorted}-\overline{xy})$ | Equal empirical masses; $C_{label}$ averages squared differences of original pairs. Ties are allowed. For standardized distinct ranks the gap becomes $2(1-\rho_S)$. |

The variance identity concerns a window average, not one day's dispersion. The rank identity describes a rank-weighted portfolio, not the tercile portfolio exactly. These distinctions prevent relabeling known quantities as independent information.

The [research design](docs/research_design.md) specifies estimation, inference, results and limitations. [Data and references](docs/data_and_references.md) attributes inputs and methods; the [results guide](docs/results_guide.md) maps the published evidence. This public package contains the complete pipeline, tests that require no licensed data, and derived research evidence; [reproducibility and scope](docs/reproducibility.md) describes the raw-data-to-results rebuild.
