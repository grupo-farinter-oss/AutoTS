"""FastTVVAR must reproduce TVVAR forecasts exactly (it only removes redundant work)."""

import unittest
import numpy as np
import pandas as pd

from autots.models.basics import TVVAR
from autots.models.fast_tvvar import FastTVVAR
from autots.evaluator.auto_model import ModelMonster

FORECAST_LENGTH = 14


def _synthetic_wide(n_series=8, n_rows=500, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=n_rows, freq="D")
    t = np.arange(n_rows)[:, None]
    values = (
        np.cumsum(rng.normal(size=(n_rows, n_series)), axis=0) * 0.3
        + 10 * np.sin(2 * np.pi * t / 7 + rng.uniform(0, 6, n_series))
        + 100
    )
    return pd.DataFrame(values, index=index, columns=[f"s{i}" for i in range(n_series)])


PARITY_CONFIGS = {
    "no_lags": {},
    "lags_rolling": dict(lags=[1, 7], rolling_means=[7]),
    "forgetting_factor": dict(lags=[1, 2], rolling_means=[4], phi=0.99),
    "forgetting_factor_short_window": dict(lags=[1], phi=0.5, max_cycles=100, lambda_=1.0),
    "x_scaled": dict(lags=[1, 7], x_scaled=True, base_scaled=False),
    "pca": dict(lags=[1, 7], rolling_means=[4], apply_pca=True, pca_n_components=0.9),
    "multiplicative": dict(lags=[1], mode="multiplicative"),
    "trend_phi": dict(lags=[7], trend_phi=0.98),
    "threshold": dict(lags=[1, 7], threshold_value=0.05),
}


class FastTVVARParityTests(unittest.TestCase):
    def assert_same_prediction(self, reference, fast):
        for attribute in ("forecast", "upper_forecast", "lower_forecast"):
            np.testing.assert_allclose(
                getattr(fast, attribute).to_numpy(),
                getattr(reference, attribute).to_numpy(),
                rtol=1e-8,
                atol=1e-8,
            )

    def test_matches_tvvar(self):
        df = _synthetic_wide()
        for name, params in PARITY_CONFIGS.items():
            with self.subTest(config=name):
                reference = TVVAR(**params).fit(df).predict(FORECAST_LENGTH)
                fast = FastTVVAR(**params).fit(df).predict(FORECAST_LENGTH)
                self.assert_same_prediction(reference, fast)

    def test_matches_tvvar_with_user_regressor(self):
        df = _synthetic_wide()
        regressor = pd.DataFrame(
            {"r": np.random.default_rng(1).normal(size=len(df) + FORECAST_LENGTH)},
            index=pd.date_range(df.index[0], periods=len(df) + FORECAST_LENGTH),
        )
        train_regr, future_regr = regressor.iloc[: len(df)], regressor.iloc[len(df) :]
        params = dict(lags=[1], regression_type="User")
        reference = TVVAR(**params).fit(df, future_regressor=train_regr)
        fast = FastTVVAR(**params).fit(df, future_regressor=train_regr)
        self.assert_same_prediction(
            reference.predict(FORECAST_LENGTH, future_regressor=future_regr),
            fast.predict(FORECAST_LENGTH, future_regressor=future_regr),
        )

    def test_var_dampening_is_finite(self):
        # intentionally differs from TVVAR (see module docstring), so only sanity checked
        df = _synthetic_wide()
        for params in (dict(lags=[1, 7], var_dampening=0.98), dict(var_dampening=0.98)):
            forecast = FastTVVAR(**params).fit(df).predict(FORECAST_LENGTH).forecast
            self.assertTrue(np.isfinite(forecast.to_numpy()).all())

    def test_components_match_tvvar(self):
        df = _synthetic_wide(n_series=5)
        configs = {
            name: PARITY_CONFIGS[name]
            for name in ("no_lags", "lags_rolling", "x_scaled", "pca", "multiplicative")
        }
        configs["multiplicative_pca"] = dict(
            lags=[1, 7], mode="multiplicative", apply_pca=True, pca_n_components=0.9
        )
        for name, params in configs.items():
            with self.subTest(config=name):
                reference = TVVAR(**params).fit(df).predict(FORECAST_LENGTH).components
                fast = (
                    FastTVVAR(store_components=True, **params)
                    .fit(df)
                    .predict(FORECAST_LENGTH)
                    .components
                )
                if reference is None:  # TVVAR raises internally when base_scaled=False
                    self.assertFalse(params.get("base_scaled", True))
                    self.assertIsNotNone(fast)
                    continue
                self.assertEqual(list(fast.columns), list(reference.columns))
                np.testing.assert_allclose(
                    fast.to_numpy(), reference.to_numpy(), rtol=1e-8, atol=1e-8
                )

    def test_var_postprocessing_matches_tvvar(self):
        # TVVAR discards the postprocessor output, so forecasts must be unchanged
        df = _synthetic_wide()
        postprocessing = {
            "fillna": "ffill",
            "transformations": {"0": "AlignLastValue"},
            "transformation_params": {"0": {}},
        }
        params = dict(lags=[1, 7], var_postprocessing=postprocessing)
        reference = TVVAR(**params).fit(df).predict(FORECAST_LENGTH)
        self.assert_same_prediction(
            reference, FastTVVAR(**params).fit(df).predict(FORECAST_LENGTH)
        )

    def test_components_opt_in(self):
        df = _synthetic_wide(n_series=3)
        params = dict(lags=[1, 7])
        self.assertIsNone(FastTVVAR(**params).fit(df).predict(FORECAST_LENGTH).components)
        model = FastTVVAR(store_components=True, **params).fit(df)
        self.assertIsNotNone(model.predict(FORECAST_LENGTH).components)
        self.assertEqual(model.process_components().shape[0], FORECAST_LENGTH)

    def test_random_params_run(self):
        df = _synthetic_wide(n_rows=300)
        for seed in range(6):
            import random

            random.seed(seed)
            params = FastTVVAR().get_new_params()
            params["regression_type"] = None
            forecast = FastTVVAR(**params).fit(df).predict(7).forecast
            self.assertEqual(forecast.shape, (7, df.shape[1]))

    def test_registered_in_model_monster(self):
        model = ModelMonster("FastTVVAR", parameters={}, frequency="D", n_jobs=1)
        self.assertIsInstance(model, FastTVVAR)


if __name__ == "__main__":
    unittest.main()
