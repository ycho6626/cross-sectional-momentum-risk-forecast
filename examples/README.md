# Synthetic walk-forward example

From the repository root, after the setup in the [main README](../README.md), run:

```sh
uv run --locked python examples/synthetic_walkforward.py
```

The command generates returns locally and prints its results. It requires no data
subscription, credentials or network access once dependencies are installed, and
writes no output files. It runs in a few seconds, depending on the machine.

The example feeds 2,088 generated weekday returns into the exported numerical
functions. Three trailing own-risk features use 5, 21 and 126 observations; the
second model adds one trailing 126-observation feature from a correlated synthetic
market series. Each feature is `log(21 * mean(return²))`. The response is the log
of the sum of 21 future squared returns, from `t+2` through `t+22`.

The existing walk-forward engine forecasts at month-ends, with annual January
refits and expanding daily training samples. Every training target must finish
before its refit. Penalties are selected using 24 months of inner validation,
with a 22-session gap after the inner training targets. Standardization and
clipping use the relevant training sample and are refitted during inner
validation. The extra market block may be excluded by the existing selection
rule. The released numerical settings are unchanged; only the in-memory
construction label becomes `synthetic`.

Expected output with the pinned environment, rounded to six decimal places:

```text
Synthetic risk forecasting example
Seed: 20261007; synthetic weekday returns: 2088
Monthly origins: 2005-12-30 to 2008-11-28 (36 per model)
Annual refits: 2005-01-03, 2006-01-02, 2007-01-01, 2008-01-01
Target: log(sum of squared returns at t+2 through t+22)
R2 compares each model with its training-mean forecast; lower MSE/QLIKE is better.
own-risk: MSE=0.422242; R2=0.480789; QLIKE=0.215007
own-risk + synthetic-market-risk: MSE=0.420760; R2=0.482610; QLIKE=0.214465
Synthetic market feature retained in 1/4 annual fits.
Illustration only: these synthetic losses are not historical or investment evidence.
```

`R2` measures squared-error improvement over each refit's training-mean forecast.
QLIKE uses the exported variance-loss function, including its training-residual
correction when converting a log forecast to a variance forecast. These losses
describe this one invented sample; no significance claim is made.

The generator uses a fixed seed and persistent synthetic volatility. Its dates
fall within the research design period, and the calendar includes weekdays
without modeling exchange holidays. Both feature sets are limited examples,
not the full historical L1 or L2 specifications. This example demonstrates the
computational path and timing conventions; it does not reconstruct historical
data, portfolios, holdout findings or trading profits.

The focused tests verify a hand-calculated trailing feature and lagged target,
completed target windows, annual fit timing, and the invariance of earlier
features, fitted parameters and forecasts when future returns are changed.
