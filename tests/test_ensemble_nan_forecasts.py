"""Ensembles must never end with NaN forecasts, nor let NaN candidates win a search.

Origin: TemplateWizard re-scores every Ensemble 1/2 template with each entry of
``horizontal_post_processors``. On short histories AlignLastValue got an empty lookback
window, its center/adjustment became NaN and the whole forecast turned NaN, yet such
candidates scored smape 0.0 (nansum over NaN) and were selected as best_model.
"""

import numpy as np
import pandas as pd
import pytest

from autots import AutoTS
from autots.evaluator.auto_model import (
    _eval_prediction_for_template,
    horizontal_post_processors,
)
from autots.models.base import PredictionObject
from autots.tools.transform import AlignLastValue, GeneralTransformer


def _short_history_and_forecast(n_rows=25):
    """Column "a" and "b" have history; column "c" has none (no possible center)."""
    index = pd.date_range("2020-01-31", periods=n_rows, freq="ME")
    history = pd.DataFrame(
        {
            "a": np.arange(n_rows, dtype=float) + 100.0,
            "b": np.arange(n_rows, dtype=float)[::-1] + 50.0,
            "c": np.nan,
        },
        index=index,
    )
    forecast_index = pd.date_range(index[-1] + pd.offsets.MonthEnd(1), periods=3, freq="ME")
    forecast = pd.DataFrame(
        {"a": [200.0, 201.0, 202.0], "b": [10.0, 11.0, 12.0], "c": [5.0, 6.0, 7.0]},
        index=forecast_index,
    )
    return history, forecast


@pytest.mark.parametrize("method", ["additive", "multiplicative"])
def test_align_last_value_window_longer_than_history_gives_no_nan(method):
    history, forecast = _short_history_and_forecast()
    transformer = AlignLastValue(
        rows=4, lag=28, method=method, strength=0.2, threshold=1, threshold_method="max"
    ).fit(history)

    result = transformer.inverse_transform(forecast)

    assert not result.isna().any().any()
    # no center for "c": left exactly as forecast
    pd.testing.assert_series_equal(result["c"], forecast["c"])
    # lag deeper than history falls back to the oldest row: a=100, b=74
    if method == "additive":
        expected_a = forecast["a"] + 0.2 * (100.0 - 200.0)
    else:
        expected_a = forecast["a"] * (1 + (100.0 / 200.0 - 1) * 0.2)
    pd.testing.assert_series_equal(result["a"], expected_a)


def test_align_last_value_nan_or_infinite_threshold_does_not_propagate_nan():
    history, forecast = _short_history_and_forecast(n_rows=60)
    transformer = AlignLastValue(rows=1, lag=1, threshold=1).fit(history)
    # threshold=1 with "max" diffs a single row, so it is NaN for every series
    assert transformer.threshold.isna().all()
    assert not transformer.inverse_transform(forecast).isna().any().any()

    transformer = AlignLastValue(rows=1, lag=1, threshold=1).fit(history)
    transformer.threshold = pd.Series(np.inf, index=history.columns)
    assert not transformer.inverse_transform(forecast).isna().any().any()


@pytest.mark.parametrize(
    "post_processor",
    horizontal_post_processors,
    ids=[str(i) for i in range(len(horizontal_post_processors))],
)
def test_every_horizontal_post_processor_fits_and_inverts(post_processor):
    rng = np.random.default_rng(0)
    index = pd.date_range("2020-01-01", periods=60, freq="D")
    history = pd.DataFrame(
        rng.uniform(50, 150, size=(60, 4)), index=index, columns=list("abcd")
    )
    forecast_index = pd.date_range(index[-1] + pd.Timedelta(days=1), periods=7, freq="D")
    forecast = pd.DataFrame(
        rng.uniform(50, 150, size=(7, 4)), index=forecast_index, columns=history.columns
    )

    transformer = GeneralTransformer(**post_processor, forecast_length=7)
    transformer.fit(history)
    result = transformer.inverse_transform(forecast)

    assert result.shape == forecast.shape


def test_nan_forecast_is_rejected_before_metrics():
    forecast = pd.DataFrame(
        {"a": [1.0, np.nan], "b": [np.nan, np.nan]},
        index=pd.date_range("2024-01-31", periods=2, freq="ME"),
    )
    prediction = PredictionObject(forecast=forecast, model_name="Ensemble")

    with pytest.raises(ValueError, match="NaN or inf"):
        # the guard runs first, the remaining arguments are never touched
        _eval_prediction_for_template(
            prediction, None, 0, None, None, None, [], None, None, None, None, 0,
            "Ensemble", 0, 1, {}, {}, {}, 0, None, None, 0, None,
        )  # fmt: skip


def test_small_simple_ensemble_search_never_selects_nan_forecasts():
    rng = np.random.default_rng(1)
    index = pd.date_range("2020-01-31", periods=25, freq="ME")
    df = pd.DataFrame(
        rng.normal(0, 10, size=(25, 20)).cumsum(axis=0) / 5 + 100,
        index=index,
        columns=[f"series_{i}" for i in range(20)],
    )
    model = AutoTS(
        forecast_length=3,
        frequency="infer",
        ensemble=["simple"],
        model_list=["LastValueNaive", "AverageValueNaive", "SeasonalNaive"],
        max_generations=1,
        num_validations=1,
        validation_method="backwards",
        n_jobs=1,
        verbose=-1,
        random_seed=3,
    ).fit(df)

    validation = model.results("validation")
    assert validation["mae"].notna().all()
    assert validation["Score"].notna().all()
    best_row = validation[validation["ID"] == model.best_model_id]
    assert not best_row.empty and best_row["mae"].notna().all()

    prediction = model.predict()  # raises "returned NaN" if the winner emits NaN
    assert np.isfinite(prediction.forecast.to_numpy()).all()
