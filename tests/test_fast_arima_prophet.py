"""Tests for the vectorised FastARIMA / FastProphet replacements."""

import unittest
import numpy as np
import pandas as pd

from autots.models.fast_arima import FastARIMA
from autots.models.fast_prophet import FastProphet
from autots.evaluator.auto_model import ModelMonster


def _synthetic_wide(n_series=6, n_rows=400, seed=0, weekly=True):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=n_rows, freq="D")
    t = np.arange(n_rows)[:, None]
    slope = rng.uniform(0.05, 0.2, n_series)
    phase = rng.uniform(0, 6, n_series)
    seasonal = 5 * np.sin(2 * np.pi * t / 7 + phase) if weekly else 0
    values = 100 + slope * t + seasonal + rng.normal(0, 0.5, (n_rows, n_series))
    return pd.DataFrame(values, index=index, columns=[f"s{i}" for i in range(n_series)])


class FastARIMATests(unittest.TestCase):
    def test_recovers_ar1_coefficient(self):
        rng = np.random.default_rng(1)
        noise = rng.normal(size=(3000, 4))
        series = np.zeros_like(noise)
        for t in range(1, len(noise)):
            series[t] = 0.7 * series[t - 1] + noise[t]
        df = pd.DataFrame(series, index=pd.date_range("2020-01-01", periods=3000))
        model = FastARIMA(p=1, d=0, q=0).fit(df)
        np.testing.assert_allclose(model.phi[:, 0], 0.7, atol=0.05)

    def test_recovers_arma11(self):
        rng = np.random.default_rng(2)
        noise = rng.normal(size=(4000, 3))
        series = np.zeros_like(noise)
        for t in range(1, len(noise)):
            series[t] = 0.6 * series[t - 1] + noise[t] + 0.3 * noise[t - 1]
        df = pd.DataFrame(series, index=pd.date_range("2020-01-01", periods=4000))
        model = FastARIMA(p=1, d=0, q=1).fit(df)
        np.testing.assert_allclose(model.phi[:, 0], 0.6, atol=0.08)
        np.testing.assert_allclose(model.theta[:, 0], 0.3, atol=0.08)

    def test_forecast_shape_ordering_and_finite(self):
        df = _synthetic_wide()
        for order in [(0, 1, 0), (2, 1, 2), (3, 0, 1), (1, 2, 1)]:
            pred = FastARIMA(p=order[0], d=order[1], q=order[2]).fit(df).predict(14)
            self.assertEqual(pred.forecast.shape, (14, df.shape[1]))
            self.assertTrue(np.isfinite(pred.forecast.to_numpy()).all(), order)
            self.assertTrue((pred.lower_forecast <= pred.forecast + 1e-9).all().all())
            self.assertTrue((pred.upper_forecast >= pred.forecast - 1e-9).all().all())
            self.assertEqual(list(pred.forecast.columns), list(df.columns))
            # intervals must widen with the horizon for integrated models
            width = (pred.upper_forecast - pred.lower_forecast).to_numpy()
            if order[1] > 0:
                self.assertTrue((width[-1] >= width[0] - 1e-9).all())

    def test_close_to_statsmodels_on_random_walk(self):
        try:
            from autots.models.statsmodels import ARIMA
        except Exception:
            self.skipTest("statsmodels unavailable")
        df = _synthetic_wide(n_series=3, n_rows=300, weekly=False)
        fast = FastARIMA(p=1, d=1, q=0).fit(df).predict(10).forecast
        slow = ARIMA(p=1, d=1, q=0, n_jobs=1).fit(df).predict(10).forecast
        relative = (fast - slow).abs().to_numpy() / df.abs().mean().to_numpy()
        self.assertLess(relative.max(), 0.02)

    def test_missing_values_and_all_nan_column(self):
        df = _synthetic_wide()
        df.iloc[:20, 0] = np.nan
        df.iloc[100:110, 1] = np.nan
        df.iloc[:, 2] = np.nan
        pred = FastARIMA(p=2, d=1, q=1).fit(df).predict(7)
        self.assertTrue(np.isfinite(pred.forecast.to_numpy()).all())

    def test_regressors(self):
        df = _synthetic_wide(n_rows=300)
        horizon = 10
        regressor = pd.DataFrame(
            {"a": np.sin(np.arange(310) / 5.0)},
            index=pd.date_range(df.index[0], periods=310),
        )
        model = FastARIMA(p=1, d=1, q=1, regression_type="User")
        model.fit(df, future_regressor=regressor.iloc[:300])
        pred = model.predict(horizon, future_regressor=regressor.iloc[300:])
        self.assertTrue(np.isfinite(pred.forecast.to_numpy()).all())
        model = FastARIMA(p=1, d=0, q=0, regression_type="Holiday", holiday_country="US")
        pred = model.fit(df).predict(horizon)
        self.assertTrue(np.isfinite(pred.forecast.to_numpy()).all())
        with self.assertRaises(ValueError):
            FastARIMA(regression_type="User").fit(df)

    def test_get_new_params_roundtrip(self):
        df = _synthetic_wide(n_rows=200)
        for _ in range(10):
            params = FastARIMA().get_new_params()
            params.pop("regression_type")
            self.assertEqual(FastARIMA(**params).get_params()["p"], params["p"])
            FastARIMA(**params).fit(df).predict(5)


class FastProphetTests(unittest.TestCase):
    def test_learns_trend_and_weekly_season(self):
        df = _synthetic_wide(n_rows=500)
        train, test = df.iloc[:-28], df.iloc[-28:]
        pred = FastProphet().fit(train).predict(28)
        mae = (pred.forecast - test).abs().mean().mean()
        self.assertLess(mae, 1.5)  # noise sd 0.5, seasonal amplitude 5
        coverage = ((test >= pred.lower_forecast) & (test <= pred.upper_forecast)).mean().mean()
        self.assertGreater(coverage, 0.7)

    def test_matches_reference_prophet_when_installed(self):
        try:
            from prophet import Prophet  # noqa: F401
            from autots.models.prophet import FBProphet
        except Exception:
            self.skipTest("prophet not installed")
        df = _synthetic_wide(n_series=2, n_rows=400)
        ours = FastProphet().fit(df).predict(14).forecast
        reference = FBProphet(n_jobs=1).fit(df).predict(14).forecast
        relative = (ours - reference).abs().to_numpy() / df.abs().mean().to_numpy()
        self.assertLess(relative.mean(), 0.03)

    def test_option_matrix_runs_and_is_finite(self):
        df = _synthetic_wide(n_rows=450)
        df.iloc[:30, 0] = np.nan
        df.iloc[200:210, 1] = np.nan
        options = [
            {},
            {"seasonality_mode": "multiplicative"},
            {"growth": "flat"},
            {"holiday_country": "US"},
            {"holiday_country": ["US", "DE"], "holidays_prior_scale": 1.0},
            {"trend_phi": 0.95},
            {"changepoint_range": 60, "changepoint_spacing": 30},
            {"changepoint_spacing": 40, "changepoint_distance_end": 20},
            {"yearly_seasonality_prior_scale": 5.0, "yearly_seasonality_order": 4,
             "weekly_seasonality_prior_scale": 1.0},
            {"yearly_seasonality": False, "weekly_seasonality": False},
        ]
        for kwargs in options:
            pred = FastProphet(**kwargs).fit(df).predict(21)
            self.assertEqual(pred.forecast.shape, (21, df.shape[1]), kwargs)
            self.assertTrue(np.isfinite(pred.forecast.to_numpy()).all(), kwargs)
            self.assertTrue((pred.lower_forecast <= pred.upper_forecast).all().all())

    def test_holiday_effect_is_learned(self):
        df = _synthetic_wide(n_series=2, n_rows=800, weekly=False)
        holiday = pd.DataFrame(
            {"ds": pd.to_datetime(["2022-07-04", "2023-07-04", "2024-07-04"]),
             "holiday": "boost", "lower_window": 0, "upper_window": 0}
        )
        df.loc[holiday["ds"].iloc[:2], :] += 40
        model = FastProphet(holidays=holiday).fit(df.loc[:"2024-03-01"])
        pred = model.predict(10)
        self.assertEqual(model._holiday_design.names, ["boost"])
        self.assertEqual(pred.forecast.shape[0], 10)

    def test_user_regressor_signal_used(self):
        df = _synthetic_wide(n_series=2, n_rows=300, weekly=False)
        regressor = pd.DataFrame(
            {"r": np.random.default_rng(3).normal(size=310)},
            index=pd.date_range(df.index[0], periods=310),
        )
        df = df + 3 * regressor["r"].iloc[:300].to_numpy()[:, None]
        model = FastProphet(regression_type="User", yearly_seasonality=False)
        pred = model.fit(df, future_regressor=regressor.iloc[:300]).predict(
            10, future_regressor=regressor.iloc[300:]
        )
        base = FastProphet(yearly_seasonality=False).fit(df).predict(10).forecast
        truth_shift = 3 * regressor["r"].iloc[300:].to_numpy()[:, None]
        with_regressor_error = np.abs(pred.forecast.to_numpy() - base.to_numpy() - truth_shift)
        self.assertLess(with_regressor_error.mean(), np.abs(truth_shift).mean())
        with self.assertRaises(ValueError):
            FastProphet(regression_type="User").fit(df)

    def test_components_frame(self):
        df = _synthetic_wide(n_series=3, n_rows=400)
        model = FastProphet(holiday_country="US")
        model.fit(df).predict(12)
        components = model.process_components()
        self.assertEqual(components.shape[0], 12)
        self.assertEqual(components.columns.nlevels, 2)
        self.assertIn("trend", components["s0"].columns)

    def test_get_new_params_roundtrip(self):
        df = _synthetic_wide(n_rows=300)
        for _ in range(8):
            params = FastProphet().get_new_params()
            if params.get("regression_type") == "User":
                params["regression_type"] = None
            if isinstance(params.get("holiday"), dict):
                params["holiday"] = False  # detector search is slow and unrelated
            FastProphet(**params).fit(df).predict(5)

    def test_registered_in_model_monster(self):
        for name, cls in [("FastARIMA", FastARIMA), ("FastProphet", FastProphet)]:
            model = ModelMonster(name, parameters={}, frequency="D", n_jobs=1)
            self.assertIsInstance(model, cls)


if __name__ == "__main__":
    unittest.main()
