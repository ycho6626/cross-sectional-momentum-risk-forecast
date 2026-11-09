# Reproducibility and scope

This repository contains the complete research pipeline, from pinned raw downloads to every published table and figure, and verifiable aggregate evidence from the completed study. It separates three claims: what the tests check without data, what the pipeline rebuilds with licensed inputs, and which published records are fixed historical inputs.

## What a public checkout can check without data

`uv sync --locked` installs the Python 3.12 environment. `uv run pytest` executes hand-calculated accounting and regression fixtures, property tests for the mathematical identities, fit-timing and exclusion checks, and synthetic transport and state-recovery tests. It also tests every pipeline stage on small synthetic DuckDB, Parquet and Git fixtures: curation, universes, market-cap repairs, data gates, books, features, the baseline, L2 and L3 studies, strategy accounts, evaluation tables, the freeze check, the holdout runner, the final report, reproduction and robustness. All inputs to those tests are generated in the tests. The real-data tests read the configured data root and skip when it holds no data, so no credentials or market-data subscription are needed.

In a checkout without data, 446 tests pass and the seven real-data tests skip; with the pinned snapshots, those seven pass as well. Lint and the evidence verifier pass. The pinned HMM dependency emits NumPy deprecation warnings whenever it fits a model; they are not failures.

`uv run momenta verify` checks each published evidence file against its SHA-256 digest, checks samples and row uniqueness, and verifies MSE-improvement, incremental-R² and Holm-adjustment arithmetic. It also checks agreement between the primary and headline tables. It does **not** recompute the stored forecast errors, Clark–West standard errors, bootstrap intervals or trading returns from raw observations; the pipeline below does. A digest establishes file integrity relative to the supplied manifest, not scientific correctness by itself.

## How the source relates to the research code

Most modules are an explicit selection from the completed research implementation. Each retained function's executable syntax tree was compared with the original, excluding docstrings and the text and identifier substitutions listed in [code provenance](code_provenance.json): error messages, the approval-tag name, output directory names and local variable names that referred to internal stage labels. Import lists and documentation were adapted to this package.

The study runners, evaluation reports, final report, reproduction and robustness modules were rewritten for this package from their originals. Stage labels became descriptive names (`baseline_study`, `l2_study`, `l3_study`), and code that only carried results between the original study's stages was removed: checkpoint validation, tag-approval runtimes and a one-time identifier migration. Their numerical paths are unchanged, which the reproduction below confirms. The command-line entry point, the stage order in `pipeline.py` and the evidence verifier were written for this package.

## Rebuilding the results

With a Sharadar subscription that includes full history and a free FRED API key, the commands below run the pipeline in its fixed order: data, momentum books, features, design-period fits, freeze check, holdout, strategy accounts and report. Each stage checks the outputs of the stages before it and writes under `data_root`.

```sh
uv run momenta data pull        # pinned snapshots with SHA-256 records
uv run momenta data curate
uv run momenta data check
uv run momenta portfolios
uv run momenta features         # targets, L1, BDP decision, L2, copula clouds, HMM warmup
uv run momenta design           # baseline, L2 and L3 design-period studies
uv run momenta freeze-check
uv run momenta holdout          # frozen models and strategy accounts on the holdout
uv run momenta report           # rewrites results/ and figures/
uv run momenta robustness       # optional pre-specified sensitivities
```

`momenta report` regenerates the released tables and figures under the release rules: strategy IDs and one reconciliation column of review notes are dropped, CSV cells are copied exactly with LF line endings, and image metadata is stripped. After a rebuild on the original vintage, `git diff results figures` shows any difference.

`uv run momenta reproduce --source DIR --output NEW` runs the same order from the pinned downloads in `DIR/raw` without downloading. It links the downloads read-only, works in a fresh directory and compares the 37 portfolio, 25 design and 25 holdout artifacts with [reproduction_hashes.json](reproduction_hashes.json), and every regenerated table and figure with the [evidence manifest](../results/manifest.json). It passes only if all of them match, apart from the run times recorded in the robustness table. Without `--robustness` the two robustness tables are not regenerated; any other missing released file fails the run. A new download is a new vintage: Sharadar revises history, French and JKP reissue their files, and FRED observations are revised, so a rebuild from fresh downloads should agree closely with the published tables but is not expected to match their digests.

The raw snapshots took 1.6 GB as downloaded and 8.4 GB on disk after extraction and staging. DuckDB is limited to 3 GB of memory and processes prices one year at a time.

## The frozen design

[config/frozen.toml](../config/frozen.toml) records the design-period choices: the selected feature prefixes (RAW L1+L2+CM, RES L1), the forecast, strategy and table registries, the origin bounds, the 56 design-period trading specifications and the complete study settings. It was published from the research file by exact string substitution, as for code: the approval-tag name, the private data path, two review-note explanations, the internal path of the trial snapshot and one settings-section name. The provenance record holds the SHA-256 of every replaced string and of both files, and reversing the substitutions restores the original bytes.

Before any holdout-period data are read, `momenta freeze-check` and `momenta holdout` require that the frozen file has its recorded SHA-256, that the data manifest has the frozen hash and that the study settings equal the frozen settings. The only exempt setting is `data_root`, the location of the user's data. No git tag is involved. The `approval_tag`, `base_commit`, `settings_hash` and `source_config_hash` fields of `[freeze]` describe the tag, commit, settings file and source tree with which the original study recorded its freeze in its private repository. The check does not use them; the [research history](research_history.md) records when the freeze happened. The holdout then runs in an isolated stage that links the design-period inputs, rebuilds every feature through the holdout bounds and refits every learned step on its own dates.

## Fixed historical records

Four published records are not rebuilt, because their inputs were fixed when the original study reproduced itself:

- [registered_forecasts_superseded.csv](../results/registered_forecasts_superseded.csv), [registered_strategies_superseded.csv](../results/registered_strategies_superseded.csv) and [registered_placebos_superseded.csv](../results/registered_placebos_superseded.csv) come from the original shared-penalty design, superseded before the holdout;
- [research_attempts.csv](../results/research_attempts.csv) summarizes the private experiment log.

[study_receipt.json](../results/study_receipt.json) remains the record of the original validation. Design-period deflated Sharpe ratios count trading specifications from earlier experiments that a rebuild does not rerun. [design_trial_population.json](../config/design_trial_population.json) publishes that recorded population, strategy identifiers and monthly Sharpe estimates only; the rebuild replaces the estimates of its own accounts. The holdout population is part of the frozen configuration.

## Reproduction result

Rebuilt from the original pinned downloads (data manifest `776526b0…`), the public pipeline matched all 37 portfolio artifacts and 48 of the 50 design and holdout artifacts byte for byte. After the robustness grid, `momenta report` regenerated the 30 published tables that are not fixed records and the five figures byte for byte, apart from the run times recorded in the robustness table.

The two exceptions, the design and holdout `fits.parquet`, differ only in fit-identity hashes inside each row's parameters: the BDP filter's in every row and the HMM's in the rows of the four HMM feature sets. Every other value is identical; with the original identities substituted, both files are byte-identical to the originals and the holdout artifacts give the original combined digest `a69991a6…`. The identities hash `bdp.py`, `baseline_study.py` and `regimes.py` and record the hmmlearn and NumPy versions, so [reproduction_hashes.json](reproduction_hashes.json) holds the public digests of both files, and any change to those files or versions changes them.

The original full suite passed **490 tests**, and the original reproduction took about an hour on the same two-core Intel laptop with 8 GB RAM, excluding downloads and the robustness grid. The sanitized [study receipt](../results/study_receipt.json) preserves its timing, counts, vintages and aggregate hashes.

## Evidence included and excluded

The release includes aggregate model losses, hypothesis tests, strategy metrics, placebo summaries, sensitivity tables, data-quality and universe summaries, reconciliation and replication tables, a summary of the experiment log and research figures. Original numeric strings are preserved in the CSV exports. The reconciliation table omits one free-text column that cited internal review notes; the [results guide](results_guide.md) gives those explanations. Figures retain their original image content. The [manifest](../results/manifest.json) identifies each released artifact.

Security-level prices, volume, actions, fundamentals, identifiers, holdings, feature panels, dated model predictions and downloaded reference series are not distributed. The pipeline creates them locally from the user's own licensed downloads.

The headline preserves the frozen design. Superseded design-period results are labeled explicitly. Post-unblinding sensitivities do not change the primary conclusion, selected models or hypotheses.
