"""Expanding, annually refitted forecasts with purged inner validation (Methodology)."""


from __future__ import annotations

import calendar as month_calendar
import json
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class Standardizer:
    """Training population moments, then clip z to ±clip_sd (Methodology)."""

    mean: np.ndarray
    scale: np.ndarray
    clip_sd: float

    @classmethod
    def fit(cls, values: np.ndarray, clip_sd: float) -> Standardizer:
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Standardization requires a finite nonempty training matrix")
        if not np.isfinite(clip_sd) or clip_sd <= 0:
            raise ValueError("The standardized feature clip must be finite and positive")
        sd = values.std(axis=0)
        return cls(values.mean(axis=0), np.where(sd > 0, sd, 1.0), clip_sd)

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply only the previously fitted training moments (Methodology)."""
        z = (np.asarray(values, dtype=float) - self.mean) / self.scale
        return np.clip(z, -self.clip_sd, self.clip_sd)


@dataclass(frozen=True)
class RegressionFit:
    """A fitted standardized model, with the unpenalized intercept (Methodology)."""

    standardizer: Standardizer
    coefficient: np.ndarray
    intercept: float
    penalty: float | None
    residual_variance: float
    column_indices: np.ndarray
    logistic: bool = False
    constant_prediction: float | None = None

    def predict(self, values: np.ndarray) -> np.ndarray:
        """Predict using this fit alone; sigmoid for mean-log-loss logistic (Methodology)."""
        if self.constant_prediction is not None:
            return np.full(len(values), self.constant_prediction)
        values = np.asarray(values, dtype=float)[:, self.column_indices]
        linear = self.intercept + self.standardizer.transform(values) @ self.coefficient
        if self.logistic:
            from scipy.special import expit

            return expit(linear)
        return linear


def ridge_path(
    x: np.ndarray, y: np.ndarray, penalties: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Minimize mean squared error + λ‖b‖², with alpha=nλ (Methodology).

    x is already standardized on its training set. A shared eigendecomposition
    evaluates the registered grid. λ=0 uses the Moore–Penrose solution.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    grid = np.asarray(penalties, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or not len(x):
        raise ValueError("Ridge needs aligned nonempty training observations")
    if (not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.isfinite(grid).all() or np.any(grid < 0)):
        raise ValueError("Ridge inputs and nonnegative penalties must be finite")
    center, target_mean = x.mean(axis=0), float(y.mean())
    centered = x - center
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    tolerance = np.finfo(float).eps * max(centered.shape) * max(eigenvalues.max(), 1.0)
    eigenvalues = np.maximum(eigenvalues, 0)
    denominators = eigenvalues[:, None] + len(y) * grid[None, :]
    numerators = (eigenvectors.T @ (centered.T @ (y - target_mean)))[:, None]
    inverse = np.divide(
        1.0, denominators, out=np.zeros_like(denominators),
        where=denominators > tolerance,
    )
    coefficients = eigenvectors @ (numerators * inverse)
    intercepts = target_mean - center @ coefficients
    return coefficients, intercepts


def block_ridge(
    x: np.ndarray, y: np.ndarray, penalties: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Solve mean squared error + Σ_j λ_j b_j², free intercept (Methodology).

    Clipped standardized x need not have mean zero. Centering it again here
    accounts for the unpenalized intercept without changing its penalty scale.
    The Moore–Penrose solution covers collinear unpenalized columns.
    """
    center, target_mean = x.mean(axis=0), float(y.mean())
    centered = x - center
    gram = centered.T @ centered / len(y)
    rhs = centered.T @ (y - target_mean) / len(y)
    coefficient = np.linalg.lstsq(gram + np.diag(penalties), rhs, rcond=None)[0]
    return coefficient, float(target_mean - center @ coefficient)


def block_logistic(
    x: np.ndarray, y: np.ndarray, penalties: np.ndarray, settings: dict[str, Any],
) -> tuple[np.ndarray, float]:
    """Minimize mean log-loss + Σ_j λ_j b_j² by exact rescaling (Methodology).

    Put z_j=x_j/sqrt(λ_j) and θ_j=sqrt(λ_j)b_j. sklearn's summed-loss
    convention at C=1/(2n) then penalizes ||θ||² on the mean-loss scale.
    Its intercept is unpenalized. Excluded columns are removed by the caller.
    """
    if len(np.unique(y)) != 2 or np.any(penalties <= 0):
        raise ValueError("Logistic training requires two classes and positive penalties")
    if not x.shape[1]:
        rate = float(y.mean())
        return np.empty(0), float(np.log(rate / (1 - rate)))
    scale = np.sqrt(penalties)
    config = settings["walkforward"]
    model = LogisticRegression(
        C=1.0 / (2.0 * len(y)), solver="lbfgs",
        max_iter=config["logit_max_iter"], tol=config["logit_tolerance"],
        random_state=settings["evaluation"]["seed"],
    ).fit(x / scale, y)
    if int(model.n_iter_.max()) >= config["logit_max_iter"]:
        raise ValueError("Logistic regression did not converge within the fixed limit")
    return model.coef_[0] / scale, float(model.intercept_[0])


def fit_regression(
    values: np.ndarray, target: np.ndarray, penalty: float | Sequence[float], *,
    logistic: bool, settings: dict[str, Any],
) -> RegressionFit:
    """Fit clipped training moments and diagonal penalties; ∞ drops columns (Methodology)."""
    values, target = np.asarray(values, dtype=float), np.asarray(target, dtype=float)
    if (values.ndim != 2 or target.shape != (len(values),) or not len(values)
            or not np.isfinite(values).all() or not np.isfinite(target).all()):
        raise ValueError("Regression needs aligned, finite, nonempty observations")
    penalties = np.broadcast_to(np.asarray(penalty, dtype=float), (values.shape[1],))
    if np.isnan(penalties).any() or (penalties < 0).any():
        raise ValueError("Penalties must be nonnegative or positive infinity for exclusion")
    columns = np.flatnonzero(np.isfinite(penalties))
    standardizer = Standardizer.fit(
        values[:, columns], settings["walkforward"]["feature_clip_sd"],
    )
    x = standardizer.transform(values[:, columns])
    if logistic:
        coefficients, intercept = block_logistic(x, target, penalties[columns], settings)
        residual_variance = float("nan")
    else:
        coefficients, intercept = block_ridge(x, target, penalties[columns])
        residual = target - (x @ coefficients + intercept)
        residual_variance = float(np.mean(residual**2))
    return RegressionFit(
        standardizer, coefficients, intercept,
        float(penalty) if np.ndim(penalty) == 0 else None,
        residual_variance, columns, logistic,
        float(target.mean()) if logistic and not len(columns) else None,
    )


def select_penalty(
    x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray,
    y_validation: np.ndarray, *, logistic: bool, settings: dict[str, Any],
) -> tuple[float, list[float]]:
    """Select λ by validation mean loss; all learned moments use inner training (Methodology)."""
    if not len(y_validation):
        raise ValueError("No validation observations")
    config = settings["walkforward"]
    penalties = config["logit_lambdas" if logistic else "ridge_lambdas"]
    if logistic:
        from scipy.special import xlog1py, xlogy

        losses = []
        for penalty in penalties:
            fit = fit_regression(x_train, y_train, penalty, logistic=True, settings=settings)
            p = fit.predict(x_validation)
            # xlogy handles exact zero probabilities without changing forecasts.
            losses.append(float(-np.mean(xlogy(y_validation, p)
                                         + xlog1py(1 - y_validation, -p))))
    else:
        standardizer = Standardizer.fit(x_train, config["feature_clip_sd"])
        coefficients, intercepts = ridge_path(
            standardizer.transform(x_train), y_train, penalties,
        )
        predictions = standardizer.transform(x_validation) @ coefficients + intercepts
        losses = np.mean((y_validation[:, None] - predictions)**2, axis=0).tolist()
    if not np.isfinite(losses).any():
        raise ValueError("Every registered penalty has nonfinite validation loss")
    best = int(np.argmin(losses))
    return float(penalties[best]), losses


def select_block_penalties(
    x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray,
    y_validation: np.ndarray, *, blocks: dict[str, Sequence[int]], target: str,
    settings: dict[str, Any],
) -> tuple[dict[str, float | str], dict[str, list[dict[str, float | str]]]]:
    """Choose each λ_b with earlier λ fixed and later columns absent (Methodology).

    Grid order is preserved, with exclusion last. Only the y1 HAR benchmark
    cannot be excluded. Exclusion removes columns rather than approximating
    infinity, so an excluded extension reproduces its selected prefix exactly.
    """
    if not len(y_validation):
        raise ValueError("No validation observations")
    if target not in {"y1", "y2", "y3"}:
        raise ValueError(f"Unknown target {target}")
    flattened = [i for indices in blocks.values() for i in indices]
    if flattened != list(range(x_train.shape[1])) or any(not v for v in blocks.values()):
        raise ValueError("Blocks must partition the ordered feature columns")
    logistic = target == "y3"
    config = settings["walkforward"]
    grid = config["logit_lambdas" if logistic else "ridge_lambdas"]
    selected: dict[str, float | str] = {}
    audit: dict[str, list[dict[str, float | str]]] = {}
    previous_columns: list[int] = []
    previous_penalties: list[float] = []
    for block, indices in blocks.items():
        candidates: list[float | str] = [float(value) for value in grid]
        if target != "y1" or block != "HAR":
            candidates.append("exclude")
        stage_columns = previous_columns + list(indices)
        standardizer = Standardizer.fit(
            x_train[:, stage_columns], config["feature_clip_sd"],
        )
        train = standardizer.transform(x_train[:, stage_columns])
        validation = standardizer.transform(x_validation[:, stage_columns])
        # Reuse sufficient statistics across a block's ridge grid. This is the
        # same centered mean-loss normal equation as block_ridge above.
        center, target_mean = train.mean(axis=0), float(y_train.mean())
        centered = train - center
        gram = centered.T @ centered / len(y_train)
        rhs = centered.T @ (y_train - target_mean) / len(y_train)
        losses = []
        for candidate in candidates:
            active = (len(previous_columns) if candidate == "exclude" else len(stage_columns))
            penalties = np.asarray(previous_penalties + (
                [] if candidate == "exclude" else [float(candidate)] * len(indices)
            ))
            if logistic:
                coefficients, intercept = block_logistic(
                    train[:, :active], y_train, penalties, settings,
                )
                linear = intercept + validation[:, :active] @ coefficients
                # logaddexp evaluates finite log-loss even when sigmoid rounds
                # to 0 or 1; it does not clip or otherwise change predictions.
                loss = float(np.mean(np.logaddexp(0.0, linear) - y_validation * linear))
            else:
                coefficients = np.linalg.lstsq(
                    gram[:active, :active] + np.diag(penalties), rhs[:active], rcond=None,
                )[0]
                intercept = target_mean - center[:active] @ coefficients
                predictions = intercept + validation[:, :active] @ coefficients
                loss = float(np.mean((y_validation - predictions)**2))
            losses.append(loss)
        if not np.isfinite(losses).all():
            raise ValueError(f"Nonfinite validation loss for block {block}")
        choice = candidates[int(np.argmin(losses))]
        selected[block] = choice
        audit[block] = [{"penalty": penalty, "loss": loss}
                        for penalty, loss in zip(candidates, losses, strict=True)]
        if choice != "exclude":
            previous_columns.extend(indices)
            previous_penalties.extend([float(choice)] * len(indices))
    return selected, audit


def assert_design_origins(origins: Sequence[date], settings: dict[str, Any]) -> None:
    """Refuse any decision after design_last_origin beyond the design boundary (Methodology)."""
    last = date.fromisoformat(settings["walkforward"]["design_last_origin"])
    if any(origin > last for origin in origins):
        raise ValueError(f"Holdout origins are locked: design_last_origin is {last}")


def month_end_origins(days: Sequence[date]) -> list[date]:
    """Return the last supplied trading session in each calendar month (Methodology)."""
    return [day for i, day in enumerate(days)
            if i == len(days) - 1 or (day.year, day.month) != (days[i + 1].year,
                                                           days[i + 1].month)]


def refit_for_origin(origin: date, days: Sequence[date]) -> date:
    """Latest first-January trading day at or before this decision (Methodology)."""
    refits = [day for i, day in enumerate(days)
              if day.month == 1 and (i == 0 or days[i - 1].year != day.year)]
    available = [day for day in refits if day <= origin]
    if not available:
        raise ValueError(f"No January refit precedes origin {origin}")
    return available[-1]


def subtract_months(day: date, months: int) -> date:
    """Subtract calendar months, retaining the day where possible (Methodology)."""
    index = day.year * 12 + day.month - 1 - months
    year, month = divmod(index, 12)
    month += 1
    return date(year, month, min(day.day, month_calendar.monthrange(year, month)[1]))


@dataclass(frozen=True)
class TrainingSplit:
    """Indices for full training, purged inner training and dated validation (Methodology)."""

    training: np.ndarray
    inner: np.ndarray
    validation: np.ndarray
    inner_cutoff: date


def training_split(
    origins: Sequence[date], target_ends: Sequence[date], days: Sequence[date],
    refit: date, eligible: np.ndarray, settings: dict[str, Any],
) -> TrainingSplit:
    """Require target_end<τ; inner target_end≤first_validation−22 sessions (Methodology)."""
    config = settings["walkforward"]
    first = date.fromisoformat(config["feature_start"])
    assert_design_origins([refit], settings)
    full = np.asarray([bool(ok) and origin >= first and end < refit
                       for origin, end, ok in zip(origins, target_ends, eligible, strict=True)])
    validation_start = subtract_months(refit, config["validation_months"])
    validation = np.flatnonzero(full & np.asarray([d >= validation_start for d in origins]))
    if not len(validation):
        raise ValueError(f"No validation set for refit {refit}")
    day_index = {day: i for i, day in enumerate(days)}
    boundary = day_index[origins[int(validation[0])]] - config["gap_days"]
    if boundary < 0:
        raise ValueError("Trading calendar does not cover the validation embargo")
    last_inner_end = days[boundary]
    inner = np.flatnonzero(full & np.asarray([end <= last_inner_end for end in target_ends]))
    if not len(inner):
        raise ValueError(f"No purged inner training set for refit {refit}")
    # French structural fits use the same completed-data boundary, exclusively.
    cutoff = max(target_ends[int(i)] for i in inner) + timedelta(days=1)
    return TrainingSplit(np.flatnonzero(full), inner, validation, cutoff)


@dataclass(frozen=True)
class ForecastRun:
    """OOS monthly forecasts, learned-fit audit, and training-only strategy inputs."""

    forecasts: pl.DataFrame
    fits: pl.DataFrame
    training_predictions: pl.DataFrame


FeatureTransform = Callable[[pl.DataFrame, date], tuple[pl.DataFrame, dict[str, Any]]]


def feature_block_indices(
    columns: Sequence[str], blocks: dict[str, Sequence[str]] | None = None,
    *, block_order: Sequence[str] | None = None,
) -> dict[str, list[int]]:
    """Resolve registered block order against ordered requested features (Methodology)."""
    if blocks is None:
        from momenta.features import FEATURE_BLOCKS

        order = list(block_order) if block_order is not None else list(FEATURE_BLOCKS)
        if len(order) != len(set(order)):
            raise ValueError("Duplicate registered block names")
        if "FRED" not in order:
            order.append("FRED")
        blocks = {name: FEATURE_BLOCKS[name] for name in order if name in FEATURE_BLOCKS}
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate requested feature columns")
    positions = {column: i for i, column in enumerate(columns)}
    result = {name: [positions[column] for column in names if column in positions]
              for name, names in blocks.items()}
    result = {name: indices for name, indices in result.items() if indices}
    if [i for indices in result.values() for i in indices] != list(range(len(columns))):
        raise ValueError("Blocks must partition the ordered feature columns")
    return result


def walk_forward(
    features: pl.DataFrame, targets: pl.DataFrame, days: Sequence[date],
    settings: dict[str, Any], *, series: str, target: str, feature_set: str,
    columns: Sequence[str], through: date | None = None,
    transform: FeatureTransform | None = None,
    blocks: dict[str, Sequence[str]] | None = None,
) -> ForecastRun:
    """Select ordered block λs and score only allowed month-ends (Methodology).

    Optional structural transformations are separately fitted at the inner and
    full cutoffs. y3 thresholds follow those same two training samples. No scored
    origin or fit can exceed the design bound, including programmatic callers.
    """
    if target not in {"y1", "y2", "y3"}:
        raise ValueError(f"Unknown target {target}")
    config = settings["walkforward"]
    last = through or date.fromisoformat(config["design_last_origin"])
    assert_design_origins([last], settings)
    block_indices = feature_block_indices(columns, blocks, block_order=config["blocks"])
    first = date.fromisoformat(config["design_first_origin"])
    first_month = (first.year, first.month)
    calendar = sorted(days)
    if not calendar or calendar[-1] <= last:
        raise ValueError("Trading calendar must extend beyond the requested origin cutoff")
    score_origins = [d for d in month_end_origins(calendar)
                     if first_month <= (d.year, d.month) and d <= last]
    if not score_origins:
        raise ValueError("No design month-end origins in the requested sample")
    selected = features.filter((pl.col("series") == series) & (pl.col("date") <= last))
    selected_targets = targets.filter(pl.col("series") == series)
    if "construction" in selected_targets.columns:
        selected_targets = selected_targets.filter(
            pl.col("construction") == settings["portfolios"]["primary"],
        )
    joined = selected.join(selected_targets.drop("series"), left_on="date", right_on="origin")
    joined = joined.sort("date")
    if joined["date"].n_unique() != joined.height:
        raise ValueError("Duplicate feature/target origins")
    origins = joined["date"].to_list()
    target_ends = joined["target_end"].to_list()
    index = {day: i for i, day in enumerate(origins)}
    score_origins = [d for d in score_origins if d in index]
    if not score_origins:
        raise ValueError("No complete targets for requested design origins")
    assert_design_origins(score_origins, settings)
    fits, forecasts, training_predictions = [], [], []
    y_raw = joined["y2" if target == "y3" else target].to_numpy()
    logistic = target == "y3"
    january_dates = [d for i, d in enumerate(calendar)
                     if d.month == 1 and (i == 0 or calendar[i - 1].year != d.year)]
    if not january_dates or january_dates[0] > min(score_origins):
        raise ValueError("No January refit precedes the first scored origin")
    assignments = {d: january_dates[bisect_right(january_dates, d) - 1] for d in score_origins}
    fit_dates = sorted(set(assignments.values()))
    monthly_origins = set(month_end_origins(calendar))
    with threadpool_limits(limits=config["blas_threads"]):
        for refit in fit_dates:
            full_frame, structural = transform(joined, refit) if transform else (joined, {})
            if full_frame["date"].to_list() != origins:
                raise ValueError("A structural transform changed origin keys/order")
            values = full_frame.select(columns).to_numpy().astype(float)
            eligible = np.isfinite(values).all(axis=1) & np.isfinite(y_raw)
            split = training_split(origins, target_ends, calendar, refit, eligible, settings)
            inner_frame, inner_structural = (
                transform(joined, split.inner_cutoff) if transform else (joined, {})
            )
            if inner_frame["date"].to_list() != origins:
                raise ValueError("An inner transform changed origin keys/order")
            inner_values = inner_frame.select(columns).to_numpy().astype(float)
            inner_ok = np.isfinite(inner_values).all(axis=1)
            if transform and not np.array_equal(
                np.isfinite(values).all(axis=1), inner_ok,
            ):
                raise ValueError("Inner and full transforms changed feature availability")
            inner = split.inner[inner_ok[split.inner]]
            validation = split.validation[inner_ok[split.validation]]
            training = split.training
            if len(inner) <= len(columns) or not len(validation):
                raise ValueError(
                    f"Insufficient inner observations for {series}/{feature_set}/{refit}",
                )
            inner_q = (float(np.quantile(y_raw[inner], settings["targets"]["tail_quantile"]))
                       if logistic else None)
            inner_y = (y_raw[inner] < inner_q).astype(int) if logistic else y_raw[inner]
            validation_y = ((y_raw[validation] < inner_q).astype(int)
                            if logistic else y_raw[validation])
            block_lambdas, validation_losses = select_block_penalties(
                inner_values[inner], inner_y, inner_values[validation], validation_y,
                blocks=block_indices, target=target, settings=settings,
            )
            penalties = np.full(len(columns), np.inf)
            for block, indices in block_indices.items():
                if block_lambdas[block] != "exclude":
                    penalties[indices] = float(block_lambdas[block])
            full_q = (float(np.quantile(y_raw[training], settings["targets"]["tail_quantile"]))
                      if logistic else None)
            full_y = ((y_raw[training] < full_q).astype(int) if logistic else y_raw[training])
            fit = fit_regression(values[training], full_y, penalties,
                                 logistic=logistic, settings=settings)
            base = float(np.mean(full_y))
            penalty_json = json.dumps(block_lambdas, sort_keys=True)
            losses_json = json.dumps(validation_losses, sort_keys=True)
            audit = {
                "series": series, "target": target, "feature_set": feature_set,
                "refit": refit, "training_count": len(training), "inner_count": len(inner),
                "validation_count": len(validation), "penalty": None,
                "block_lambdas_json": penalty_json, "validation_losses_json": losses_json,
                "training_origin_max": max(origins[int(i)] for i in training),
                "training_target_end_max": max(target_ends[int(i)] for i in training),
                "inner_target_end_max": max(target_ends[int(i)] for i in inner),
                "validation_origin_min": min(origins[int(i)] for i in validation),
                "inner_fit_cutoff": split.inner_cutoff,
                "tail_threshold": full_q, "inner_tail_threshold": inner_q,
                "residual_variance": fit.residual_variance if not logistic else None,
                "parameters_json": json.dumps({
                    "columns": [columns[int(i)] for i in fit.column_indices],
                    "candidate_columns": list(columns), "block_order": list(block_indices),
                    "block_lambdas": block_lambdas,
                    "mean": fit.standardizer.mean.tolist(),
                    "scale": fit.standardizer.scale.tolist(),
                    "clip_sd": fit.standardizer.clip_sd,
                    "coefficient": fit.coefficient.tolist(), "intercept": fit.intercept,
                    "validation_losses": validation_losses, "structural": structural,
                    "inner_structural": inner_structural,
                }, sort_keys=True, default=str),
            }
            fits.append(audit)
            scored = [index[d] for d in score_origins
                      if assignments[d] == refit and eligible[index[d]]]
            predictions = fit.predict(values[scored])
            for i, predicted in zip(scored, predictions, strict=True):
                actual = float(y_raw[i] < full_q) if logistic else float(y_raw[i])
                forecasts.append({
                    "series": series, "target": target, "feature_set": feature_set,
                    "origin": origins[i], "refit": refit,
                    "target_start": joined["target_start"][i], "target_end": target_ends[i],
                    "actual": actual, "prediction": float(predicted), "base_prediction": base,
                    "residual_variance": fit.residual_variance if not logistic else None,
                    "penalty": None, "training_count": len(training),
                    "block_lambdas_json": penalty_json, "validation_losses_json": losses_json,
                    "tail_threshold": full_q, "y2": float(joined["y2"][i]),
                })
            if target == "y1" and feature_set == "L1":
                selected_training = [int(i) for i in training
                                     if origins[int(i)] in monthly_origins]
                predicted = fit.predict(values[selected_training])
                for i, value in zip(selected_training, predicted, strict=True):
                    variance = float(np.exp(value + fit.residual_variance / 2))
                    if not np.isfinite(variance) or variance <= 0:
                        raise ValueError("Nonpositive or nonfinite training variance forecast")
                    training_predictions.append({
                        "series": series, "refit": refit, "origin": origins[i],
                        "target_end": target_ends[i], "variance_prediction": variance,
                        "bear_x_var": float(full_frame["bear_x_var"][i]),
                    })
    frame = pl.DataFrame(forecasts, infer_schema_length=None).sort("origin")
    if frame.is_empty() or not frame["prediction"].is_finite().all():
        raise ValueError("No finite out-of-sample forecasts")
    return ForecastRun(frame, pl.DataFrame(fits, infer_schema_length=None),
                       pl.DataFrame(training_predictions, infer_schema_length=None))
