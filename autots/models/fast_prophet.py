"""FastProphet: vectorised drop-in replacement for ``FBProphet`` (no Stan, no prophet).

Same model family as Prophet: piecewise-linear trend with changepoints, Fourier
seasonality, holidays and regressors, additive or multiplicative. The differences
that buy the speed:
  * posterior mode via batched IRLS ridge solves (see prophet_style_solver) for every
    series at once, instead of one Stan optimisation per series
  * analytic prediction intervals (observation noise + Prophet's future-changepoint
    trend variance) instead of Monte Carlo simulation
Known deviations: time axis / changepoints use the shared frame index (series with
leading NaNs are masked rather than rescaled individually); 'logistic' growth is not
supported; multiplicative mode is solved by backfitting rather than joint MAP.
"""

import random
import datetime
import numpy as np
import pandas as pd
from autots.models.base import ModelObject, PredictionObject
from autots.tools.batched_linear_solvers import normal_quantile
from autots.tools.prophet_style_solver import apply_design, penalised_least_squares
from autots.tools.prophet_style_features import (
    auto_seasonality_flags,
    changepoint_basis,
    country_holiday_events,
    expand_holiday_windows,
    fourier_matrix,
    HolidayDesign,
    index_to_days,
    trend_design,
)

_MAX_USER_REGRESSOR_COLUMNS = 8
_CHUNK_ELEMENT_BUDGET = 4e7  # (series * rows * features) floats materialised at once


class FastProphet(ModelObject):
    """Prophet-style forecaster solved in closed form for all series simultaneously.

    Accepts the same constructor arguments as ``FBProphet`` so parameter templates are
    interchangeable. ``n_jobs`` is accepted but unused (BLAS handles threading).
    """

    def __init__(
        self,
        name: str = "FastProphet",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday: bool = False,
        regression_type: str = None,
        holiday_country: str = None,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        growth: str = "linear",
        n_changepoints: int = 25,
        changepoint_prior_scale: float = 0.05,
        seasonality_mode: str = "additive",
        changepoint_range: float = 0.8,
        changepoint_spacing: int = None,
        changepoint_distance_end: int = None,
        seasonality_prior_scale: float = 10.0,
        weekly_seasonality_prior_scale: float = None,
        weekly_seasonality_order: int = 4,
        yearly_seasonality_prior_scale: float = None,
        yearly_seasonality_order: int = None,
        holidays_prior_scale: float = 10.0,
        trend_phi: float = 1,
        phi: float = None,
        holidays=None,
        random_seed: int = 2024,
        verbose: int = 0,
        n_jobs: int = None,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            regression_type=regression_type,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.holiday = holiday
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.growth = growth
        self.n_changepoints = n_changepoints
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_mode = seasonality_mode
        self.changepoint_range = changepoint_range
        self.changepoint_spacing = changepoint_spacing
        self.changepoint_distance_end = changepoint_distance_end
        self.seasonality_prior_scale = seasonality_prior_scale
        self.weekly_seasonality_prior_scale = weekly_seasonality_prior_scale
        self.weekly_seasonality_order = weekly_seasonality_order
        self.yearly_seasonality_prior_scale = yearly_seasonality_prior_scale
        self.yearly_seasonality_order = yearly_seasonality_order
        self.holidays_prior_scale = holidays_prior_scale
        self.trend_phi = phi if phi is not None else trend_phi
        self.holidays = holidays
        self.regressor_name = []

    # ------------------------------------------------------------------ features
    @staticmethod
    def _is_set(value):
        return value not in [None, "None"]

    def _seasonality_specs(self, days):
        """List of (name, period_days, fourier_order, prior_scale)."""
        auto = auto_seasonality_flags(days)
        median_step = float(np.median(np.diff(days))) if len(days) > 1 else 1.0
        specs = []
        custom_weekly = self._is_set(self.weekly_seasonality_prior_scale)
        custom_yearly = self._is_set(self.yearly_seasonality_prior_scale)

        def enabled(setting, key):
            return auto[key] if setting == "auto" else bool(setting)

        if custom_yearly:
            order = 12 if not self._is_set(self.yearly_seasonality_order) else self.yearly_seasonality_order
            specs.append(('yearly', 365.25, int(order), float(self.yearly_seasonality_prior_scale)))
        elif enabled(self.yearly_seasonality, 'yearly'):
            specs.append(('yearly', 365.25, 10, self.seasonality_prior_scale))
        # weekly cycle cannot be resolved by data sampled coarser than ~3.5 days
        if median_step < 3.5:
            if custom_weekly:
                specs.append(('weekly', 7.0, int(self.weekly_seasonality_order), float(self.weekly_seasonality_prior_scale)))
            elif enabled(self.weekly_seasonality, 'weekly'):
                specs.append(('weekly', 7.0, 3, self.seasonality_prior_scale))
        if median_step < 0.5 and enabled(self.daily_seasonality, 'daily'):
            specs.append(('daily', 1.0, 4, self.seasonality_prior_scale))
        return specs

    def _holiday_events(self, index, detector=None):
        """Long (date, name, series) frame from countries, user table and detector."""
        pieces = []
        countries = self.holiday_country
        if self._is_set(countries):
            countries = [countries] if isinstance(countries, str) else list(countries)
            years = range(index[0].year - 1, index[-1].year + 2)
            events = country_holiday_events(countries, years)
            if not events.empty:
                pieces.append(events.assign(series=np.nan))
        if isinstance(self.holidays, pd.DataFrame) and not self.holidays.empty:
            user = self.holidays.rename(columns={'ds': 'date', 'holiday': 'name'})
            pieces.append(expand_holiday_windows(user))
        if detector is not None:
            detected = detector.dates_to_holidays(index, style="prophet")
            if len(detected):
                pieces.append(
                    detected.rename(columns={'ds': 'date', 'holiday': 'name'})[
                        ['date', 'name', 'series']
                    ]
                )
        if not pieces:
            return None
        return pd.concat(pieces, ignore_index=True)

    def _regressor_matrix(self, values):
        """Standardised (n, r) regressors using training statistics."""
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        centred = (values - self._regr_mean) / self._regr_scale
        return centred @ self._regr_projection if self._regr_projection is not None else centred

    def _fit_regressor_stats(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        self._regr_mean = values.mean(axis=0)
        spread = values.std(axis=0)
        self._regr_scale = np.where(spread > 0, spread, 1.0)
        self._regr_projection = None
        if values.shape[1] > _MAX_USER_REGRESSOR_COLUMNS:
            z = (values - self._regr_mean) / self._regr_scale
            _, _, vt = np.linalg.svd(z, full_matrices=False)
            self._regr_projection = vt[:_MAX_USER_REGRESSOR_COLUMNS].T

    def _shared_features(self, index, regressor_values):
        """Series-independent feature columns: Fourier blocks + regressors."""
        days = index_to_days(index)
        parts, slices, std, start = [], [], [], 0
        for name, period, order, prior in self._season_specs:
            block = fourier_matrix(days, period, order)
            parts.append(block)
            slices.append((name, slice(start, start + block.shape[1])))
            std.extend([prior] * block.shape[1])
            start += block.shape[1]
        if regressor_values is not None:
            block = self._regressor_matrix(regressor_values)
            parts.append(block)
            slices.append(('extra_regressors', slice(start, start + block.shape[1])))
            std.extend([self.seasonality_prior_scale] * block.shape[1])
            start += block.shape[1]
        matrix = np.concatenate(parts, axis=1) if parts else np.zeros((len(index), 0))
        return matrix, slices, std, start

    def _feature_chunk(self, shared, holiday_design, index, series_slice, start):
        """Append holiday columns: result is 2D if global/none, else (c, n, k)."""
        if holiday_design is None or not holiday_design.names:
            return shared, []
        dense = holiday_design.dense(index, series_slice)
        slices = [('holidays', slice(start, start + len(holiday_design.names)))]
        if dense.ndim == 2:
            return np.concatenate([shared, dense], axis=1), slices
        n_chunk = dense.shape[0]
        tiled = np.broadcast_to(shared[None], (n_chunk,) + shared.shape)
        return np.concatenate([tiled, dense], axis=2), slices

    def _chunk_bounds(self, n_series, n_rows, n_features):
        per_series = max(n_rows * max(n_features, 1), 1)
        size = int(max(min(_CHUNK_ELEMENT_BUDGET // per_series, n_series), 1))
        return [slice(i, min(i + size, n_series)) for i in range(0, n_series, size)]

    # ---------------------------------------------------------------------- fit
    def fit(self, df, future_regressor=None):
        df = self.basic_profile(df)
        self.regressor_train = None
        if self.regression_type in ['User', 'user']:
            if future_regressor is None:
                raise ValueError("regression_type='User' but no future_regressor passed")
            self.regressor_train = np.asarray(future_regressor, dtype=float)
            self._fit_regressor_stats(self.regressor_train)
            self.regressor_name = ['user_regressor']
        self.df_train = df
        if self.growth not in ['linear', 'flat']:
            raise ValueError(f"FastProphet growth='{self.growth}' not supported")

        index = df.index
        days = index_to_days(index)
        self._origin_day = days[0]
        self._span_days = max(days[-1] - days[0], 1e-9)
        time_scaled = (days - self._origin_day) / self._span_days
        observed = ~np.isnan(df.to_numpy(dtype=float))
        y = df.to_numpy(dtype=float)
        scale = np.nanmax(np.abs(np.where(observed, y, np.nan)), axis=0)
        self._y_scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        targets = np.where(observed, y / self._y_scale, 0.0)

        if self.growth == 'flat':
            self._changepoints = np.zeros(0)
            trend_x = np.ones((len(index), 1))
        else:
            self._changepoints = changepoint_basis(
                days, time_scaled, days, self.n_changepoints, self.changepoint_range,
                self.changepoint_spacing, self.changepoint_distance_end,
            )
            trend_x = trend_design(time_scaled, self._changepoints)
        n_trend = trend_x.shape[1]
        trend_std = np.full(n_trend, 5.0)
        trend_std[2:] = np.inf  # changepoint deltas: Laplace, handled by the solver
        laplace = slice(2, n_trend) if n_trend > 2 else None

        self._season_specs = self._seasonality_specs(days)
        shared, _, shared_std, n_shared = self._shared_features(
            index, self.regressor_train
        )
        self._holiday_detector = None
        if isinstance(self.holiday, dict):
            from autots.evaluator.anomaly_detector import HolidayDetector

            self._holiday_detector = HolidayDetector(**self.holiday)
            self._holiday_detector.detect(df)
        events = self._holiday_events(index, self._holiday_detector)
        self._holiday_design = None
        if events is not None and len(events):
            design = HolidayDesign(events, df.columns).restrict_to_active(index)
            self._holiday_design = design if design.names else None
        n_holiday = len(self._holiday_design.names) if self._holiday_design else 0
        feature_std = np.array(
            shared_std + [self.holidays_prior_scale] * n_holiday, dtype=float
        )
        multiplicative = self.seasonality_mode == 'multiplicative'

        n_series = df.shape[1]
        self._trend_coef = np.zeros((n_series, n_trend))
        self._feature_coef = np.zeros((n_series, n_shared + n_holiday))
        self._sigma = np.zeros(n_series)
        for chunk in self._chunk_bounds(n_series, len(index), n_trend + n_shared + n_holiday):
            feats, _ = self._feature_chunk(shared, self._holiday_design, index, chunk, n_shared)
            y_c, m_c = targets[:, chunk], observed[:, chunk]
            if feats.shape[-1] == 0:
                coef, sigma2 = penalised_least_squares(
                    trend_x, y_c, m_c, trend_std, laplace, self.changepoint_prior_scale
                )
                self._trend_coef[chunk], self._sigma[chunk] = coef, np.sqrt(sigma2)
                continue
            if not multiplicative:
                if feats.ndim == 2:
                    design = np.concatenate([trend_x, feats], axis=1)
                else:
                    tiled = np.broadcast_to(trend_x[None], (feats.shape[0],) + trend_x.shape)
                    design = np.concatenate([tiled, feats], axis=2)
                std = np.concatenate([trend_std, feature_std])
                coef, sigma2 = penalised_least_squares(
                    design, y_c, m_c, std, laplace, self.changepoint_prior_scale
                )
                self._trend_coef[chunk] = coef[:, :n_trend]
                self._feature_coef[chunk] = coef[:, n_trend:]
            else:
                trend_coef, sigma2 = penalised_least_squares(
                    trend_x, y_c, m_c, trend_std, laplace, self.changepoint_prior_scale
                )
                for _ in range(3):  # backfit g(t) * (1 + s(t))
                    trend = apply_design(trend_x, trend_coef)
                    trend = np.where(np.abs(trend) < 1e-3, np.where(trend < 0, -1e-3, 1e-3), trend)
                    ratio = np.where(m_c, y_c / trend - 1.0, 0.0)
                    # exact objective (y - g(1+s))^2 = g^2 (y/g - 1 - s)^2 -> weight g^2
                    feature_coef, _ = penalised_least_squares(
                        feats, ratio, m_c, feature_std, weights=trend**2 * m_c
                    )
                    season = np.clip(apply_design(feats, feature_coef), -0.95, None)
                    trend_coef, sigma2 = penalised_least_squares(
                        trend_x, np.where(m_c, y_c / (1.0 + season), 0.0), m_c,
                        trend_std, laplace, self.changepoint_prior_scale,
                        weights=(1.0 + season) ** 2 * m_c,
                    )
                self._trend_coef[chunk] = trend_coef
                self._feature_coef[chunk] = feature_coef
                fitted = apply_design(trend_x, trend_coef) * (1.0 + season)
                sigma2 = (((y_c - fitted) * m_c) ** 2).sum(0) / np.maximum(m_c.sum(0), 1)
            self._sigma[chunk] = np.sqrt(np.maximum(sigma2, 1e-10))
        self._n_hist = len(index)
        self._median_step_scaled = float(np.median(np.diff(days))) / self._span_days if len(days) > 1 else 1.0
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, forecast_length: int, future_regressor=None, just_point_forecast: bool = False):
        predictStartTime = datetime.datetime.now()
        test_index = self.create_forecast_index(forecast_length=forecast_length)
        days_future = index_to_days(test_index)
        time_future = (days_future - self._origin_day) / self._span_days
        if self.growth == 'flat':
            trend_x = np.ones((forecast_length, 1))
        else:
            trend_x = trend_design(time_future, self._changepoints)
        regressor_future = None
        if self.regression_type in ['User', 'user']:
            if future_regressor is None:
                raise ValueError("regression_type='User' but no future_regressor passed")
            regressor_future = np.asarray(future_regressor, dtype=float)
        shared, shared_slices, _, n_shared = self._shared_features(test_index, regressor_future)
        design = None
        if self._holiday_design is not None:
            fixed_names = self._holiday_design.names
            full_index = self.df_train.index.append(test_index)
            events = self._holiday_events(full_index, self._holiday_detector)
            design = HolidayDesign(events, self.column_names, names=fixed_names)

        n_series = len(self.column_names)
        n_hol = len(design.names) if design is not None else 0
        trend = np.zeros((forecast_length, n_series))
        multiplicative = self.seasonality_mode == 'multiplicative'
        components = {}
        season_total = np.zeros((forecast_length, n_series))
        slices = shared_slices + ([('holidays', slice(n_shared, n_shared + n_hol))] if n_hol else [])
        for chunk in self._chunk_bounds(n_series, forecast_length, n_shared + n_hol + 1):
            trend[:, chunk] = apply_design(trend_x, self._trend_coef[chunk])
            feats, _ = self._feature_chunk(shared, design, test_index, chunk, n_shared)
            if feats.shape[-1] == 0:
                continue
            coef = self._feature_coef[chunk]
            season_total[:, chunk] = apply_design(feats, coef)
            for name, sl in slices:
                part = apply_design(feats[..., sl], coef[:, sl])
                components.setdefault(name, np.zeros((forecast_length, n_series)))[:, chunk] = part

        undamped_trend = trend
        phi = self.trend_phi
        if phi is not None and phi != 1 and forecast_length > 2:
            powers = phi ** np.arange(forecast_length - 1)
            increments = np.diff(trend, axis=0) * powers[:, None]
            trend = np.concatenate([trend[:1], increments], axis=0).cumsum(axis=0)
        if multiplicative:
            point_scaled = trend * (1.0 + season_total)
        else:
            point_scaled = trend + season_total

        # Prophet's future-changepoint trend uncertainty, in closed form
        horizon = np.arange(1, forecast_length + 1)[:, None].astype(float)
        n_cp = len(self._changepoints)
        if n_cp:
            delta_scale = np.abs(self._trend_coef[:, 2:]).mean(axis=1)
            rate = n_cp / self._n_hist
            trend_var = (
                rate * 2.0 * delta_scale[None, :] ** 2 * self._median_step_scaled**2
                * (horizon - 1) * horizon * (2 * horizon - 1) / 6.0
            )
            if multiplicative:
                trend_var = trend_var * (1.0 + season_total) ** 2
        else:
            trend_var = 0.0
        z_value = normal_quantile(1 - (1 - self.prediction_interval) / 2)
        half_width = z_value * np.sqrt(self._sigma[None, :] ** 2 + trend_var)

        scale = self._y_scale[None, :]
        columns = self.column_names
        forecast = pd.DataFrame(point_scaled * scale, index=test_index, columns=columns)
        if just_point_forecast:
            return forecast
        lower_forecast = pd.DataFrame((point_scaled - half_width) * scale, index=test_index, columns=columns)
        upper_forecast = pd.DataFrame((point_scaled + half_width) * scale, index=test_index, columns=columns)

        components['trend'] = undamped_trend * scale
        key = 'multiplicative_terms' if multiplicative else 'additive_terms'
        components[key] = season_total if multiplicative else season_total * scale
        if not multiplicative:
            for name in list(components):
                if name not in ('trend', key):
                    components[name] = components[name] * scale
        self._components_arrays = (components, test_index)
        return PredictionObject(
            model_name=self.name,
            forecast_length=forecast_length,
            forecast_index=test_index,
            forecast_columns=forecast.columns,
            lower_forecast=lower_forecast,
            forecast=forecast,
            upper_forecast=upper_forecast,
            prediction_interval=self.prediction_interval,
            predict_runtime=datetime.datetime.now() - predictStartTime,
            fit_runtime=self.fit_runtime,
            model_parameters=self.get_params(),
        )

    def process_components(self):
        """Wide frame with MultiIndex columns (series, component), like FBProphet."""
        components, index = self._components_arrays
        names = list(components)
        stacked = np.stack([components[n] for n in names], axis=2)  # (h, N, C)
        columns = pd.MultiIndex.from_product([self.column_names, names])
        return pd.DataFrame(stacked.reshape(stacked.shape[0], -1), index=index, columns=columns)

    # ---------------------------------------------------------------- parameters
    def get_new_params(self, method: str = 'random'):
        """Identical search space to FBProphet so both can share templates."""
        from autots.models.prophet import FBProphet

        return FBProphet().get_new_params(method=method)

    def get_params(self):
        return {
            'holiday': self.holiday,
            'regression_type': self.regression_type,
            "growth": self.growth,
            "seasonality_mode": self.seasonality_mode,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "n_changepoints": self.n_changepoints,
            "changepoint_range": self.changepoint_range,
            "changepoint_spacing": self.changepoint_spacing,
            "changepoint_distance_end": self.changepoint_distance_end,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "yearly_seasonality_prior_scale": self.yearly_seasonality_prior_scale,
            "yearly_seasonality_order": self.yearly_seasonality_order,
            "weekly_seasonality_prior_scale": self.weekly_seasonality_prior_scale,
            "holidays_prior_scale": self.holidays_prior_scale,
            "trend_phi": self.trend_phi,
        }
