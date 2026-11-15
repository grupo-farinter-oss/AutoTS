"""FastTVVAR: same model as ``TVVAR`` with the slow paths rewritten in numpy.

Profiling TVVAR showed four hotspots, none of them intrinsic to the model:
  * fit with a forgetting factor ``phi`` solved a full linear system at every time step
    but only kept the last one. The final solution is a single exponentially weighted
    ridge regression, so it is solved once (identical result).
  * predict rebuilt the lag / rolling-mean features over the *entire* history with
    pandas at every forecast step to read one row. Here each step indexes numpy arrays.
  * component breakdown (pandas loops over every series x feature) dominated predict.
    Category totals are linear in the features, so they are matrix products here, and
    only built when ``store_components=True``.
  * ``var_postprocessing`` refit a GeneralTransformer at every forecast step, but TVVAR
    discards the inverse_transform result (``inverse_transform(df)`` is not assigned),
    so it never changes the forecast (verified on 82 random postprocessing draws). It
    is skipped here: identical forecasts, except draws where TVVAR's postprocessor
    raised (e.g. Constraint / FIRFilter on one-row frames) now return a forecast.

Deviations from TVVAR:
  * ``var_dampening`` is applied to every VAR feature column. TVVAR slices
    ``[-n_var:-1]``, which skips the last VAR column and, when there are no lag
    features, dampens nearly all exogenous columns instead.
  * components are also returned when ``base_scaled=False`` (TVVAR indexes the float
    0.0 scaler mean by series name, raises, and silently returns no components).
Falls back to TVVAR's fit only for ``store_phi`` (needs every intermediate solve;
not part of the parameter search).
"""

import datetime
import numpy as np
import pandas as pd
from autots.models.base import PredictionObject, stack_component_frames
from autots.models.basics import TVVAR
from autots.tools.seasonal import date_part
from autots.tools.changepoints import changepoint_fcst_from_last_row
from autots.tools.batched_linear_solvers import normal_quantile


def _dampen_cumulative(block_contribution: np.ndarray, factor: float) -> np.ndarray:
    """Scale step-to-step increments by factor**k (same rule TVVAR applies per column).

    The rule is linear, so applying it to the sum of a block's columns equals applying
    it column by column and summing, which avoids the (horizon, features, series) tensor.
    """
    powers = factor ** np.arange(block_contribution.shape[0] - 1)
    increments = np.diff(block_contribution, axis=0) * powers[:, None]
    return np.cumsum(np.vstack([block_contribution[:1], increments]), axis=0)


class FastTVVAR(TVVAR):
    """Time Varying VAR with vectorised fit (forgetting factor) and predict.

    Args:
        store_components (bool): build the per-feature component breakdown during
            predict (slow for many series). TVVAR always does this.
        All other arguments are identical to ``TVVAR``.
    """

    def __init__(self, name: str = "FastTVVAR", store_components: bool = False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.store_components = store_components

    # ---------------------------------------------------------------------- fit
    def fit(self, df, future_regressor=None):
        if self.phi is None or self.store_phi:
            return super().fit(df, future_regressor=future_regressor)
        # reuse TVVAR's feature pipeline (closed-form branch), then replace its beta
        forgetting_factor = self.phi
        self.phi = None
        try:
            super().fit(df, future_regressor=future_regressor)
        finally:
            self.phi = forgetting_factor
        self._fit_with_forgetting_factor(df)
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def _scaled_target(self, df):
        """Reproduce TVVAR's target transform (log for multiplicative, then scaling)."""
        target = df
        if self.mode == 'multiplicative':
            target = (
                np.log(df.replace(0, np.nan)).replace(-np.inf, np.nan).bfill().ffill()
            )
        target = (target - self.scaler_mean) / self.scaler_std
        return target.loc[self.X.index].to_numpy(dtype=float)

    def _fit_with_forgetting_factor(self, df):
        """argmin sum_t phi^(T-1-t) ||y_t - x_t beta||^2 + alpha ||beta||^2, solved once.

        Equal to TVVAR's recursive S = phi*S + x x', r = phi*r + x y' after the last step.
        """
        features = np.asarray(self.x_scaler.transform(self.X), dtype=float)
        targets = self._scaled_target(df)
        n_samples, n_features = features.shape
        first_row = 0
        if self.max_cycles is not None:
            first_row = max(n_samples - int(self.max_cycles), 0)
        window_features = features[first_row:]
        ages = np.arange(n_samples - first_row - 1, -1, -1)
        weighted = window_features * (self.phi**ages)[:, None]
        gram = weighted.T @ window_features
        cross = weighted.T @ targets[first_row:]
        ridge_alpha = self.lambda_ if self.lambda_ is not None else 0.0001
        gram[np.diag_indices(n_features)] += ridge_alpha
        try:
            beta = np.linalg.solve(gram, cross)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(gram) @ cross
        self.beta = self.apply_beta_threshold(beta)
        residuals = targets - features @ self.beta
        degrees_of_freedom = targets.shape[0] - n_features
        self.sigma = np.sqrt(np.sum(residuals**2, axis=0) / degrees_of_freedom)

    # ------------------------------------------------------------------ predict
    def _future_exogenous(self, test_index, future_regressor):
        """Same exogenous design as TVVAR.predict (dateparts, changepoints, regressors)."""
        seasonal_part = date_part(
            test_index,
            method=self.datepart_method,
            set_index=True,
            holiday_country=self.holiday_country,
            holiday_countries_used=self.holiday_countries_used,
        )
        trend_part = changepoint_fcst_from_last_row(self.last_row, len(test_index))
        trend_part.index = test_index
        exogenous = pd.concat([seasonal_part, trend_part], axis=1)
        if str(self.regression_type).lower() == "user":
            regressors = future_regressor.reindex(test_index).rename(
                columns=lambda x: "regr_" + str(x)
            )
            exogenous = pd.concat([exogenous, regressors], axis=1)
        exogenous["constant"] = 1
        trend_columns = slice(
            seasonal_part.shape[1], seasonal_part.shape[1] + trend_part.shape[1]
        )
        return exogenous, trend_columns

    def _affine_feature_scaling(self):
        """x_scaler is per-column affine (or identity): recover offset and slope once."""
        columns = self.X.columns
        zeros = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
        offset = np.asarray(self.x_scaler.transform(zeros), dtype=float)[0]
        slope = np.asarray(self.x_scaler.transform(zeros + 1.0), dtype=float)[0] - offset
        return offset, slope

    def _var_feature_row(self, extended, row, lags, windows, n_series):
        """Lag then rolling-mean blocks, same column order as create_VAR_features."""
        blocks = [extended[row - lag] for lag in lags]
        for window in windows:
            if row >= window:  # pandas rolling(min_periods=window) gives NaN otherwise
                blocks.append(extended[row - window : row].mean(axis=0))
            else:
                blocks.append(np.full(n_series, np.nan))
        return np.concatenate(blocks) if blocks else np.zeros(0)

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        predict_start_time = datetime.datetime.now()
        test_index = self.create_forecast_index(forecast_length=forecast_length)
        exogenous_frame, trend_columns = self._future_exogenous(
            test_index, future_regressor
        )
        feature_columns = self.X.columns
        n_exogenous = exogenous_frame.shape[1]
        if list(exogenous_frame.columns) != list(feature_columns[:n_exogenous]):
            raise ValueError("FastTVVAR: forecast features do not match training features")
        exogenous = exogenous_frame.to_numpy(dtype=float)
        offset, slope = self._affine_feature_scaling()

        exogenous_contribution = (
            exogenous * slope[:n_exogenous] + offset[:n_exogenous]
        ) @ self.beta[:n_exogenous]
        var_beta = self.beta[n_exogenous:]
        var_slope, var_offset = slope[n_exogenous:], offset[n_exogenous:]

        history = self.var_history.to_numpy(dtype=float)
        n_history, n_series = history.shape
        extended = np.vstack([history, np.full((forecast_length, n_series), np.nan)])
        lags = list(self.lags_list or [])
        windows = list(self.rolling_avg_list or [])
        var_inputs = np.zeros((forecast_length, len(feature_columns) - n_exogenous))
        predictions = np.zeros((forecast_length, n_series))
        # recursion: each step's VAR features include the previous steps' predictions
        for step in range(forecast_length):
            row = n_history + step
            var_row = self._var_feature_row(extended, row, lags, windows, n_series)
            if self.apply_pca:
                var_row = self.pca.transform(
                    pd.DataFrame(var_row[None, :], columns=self.VAR_feature_columns)
                )[0]
            var_inputs[step] = var_row
            predictions[step] = exogenous_contribution[step] + (
                var_row * var_slope + var_offset
            ) @ var_beta
            extended[row] = predictions[step]

        raw_features = np.hstack([exogenous, var_inputs])
        self.X_pred = pd.DataFrame(raw_features, index=test_index, columns=feature_columns)
        if forecast_length >= 2:
            scaled_features = raw_features * slope + offset
            dampened_blocks = []
            if self.trend_phi is not None and self.trend_phi != 1:
                dampened_blocks.append((trend_columns, self.trend_phi))
            if (
                self.var_dampening is not None
                and self.var_dampening != 1
                and var_inputs.shape[1] > 0
            ):
                dampened_blocks.append((slice(n_exogenous, None), self.var_dampening))
            for columns, factor in dampened_blocks:
                block = scaled_features[:, columns] @ self.beta[columns]
                predictions = predictions - block + _dampen_cumulative(block, factor)

        forecast = (
            pd.DataFrame(predictions, index=test_index, columns=self.column_names)
            * self.scaler_std
            + self.scaler_mean
        )
        if self.mode == 'multiplicative':
            forecast = np.exp(forecast)
        if just_point_forecast:
            return forecast
        z_value = normal_quantile(1 - (1 - self.prediction_interval) / 2)
        margin_of_error = pd.DataFrame(
            z_value * np.tile(self.sigma, (forecast_length, 1)) * self.scaler_std.values,
            columns=self.column_names,
            index=forecast.index,
        )
        if self.mode == 'multiplicative':  # kept identical to TVVAR
            margin_of_error = np.exp(margin_of_error)
        components = None
        if self.store_components:
            components = self._components_frame(
                forecast.index, raw_features * slope + offset, n_exogenous
            )
        return PredictionObject(
            model_name=self.name,
            forecast_length=forecast_length,
            forecast_index=forecast.index,
            forecast_columns=forecast.columns,
            lower_forecast=(forecast - margin_of_error).astype(float),
            forecast=forecast.astype(float),
            upper_forecast=(forecast + margin_of_error).astype(float),
            prediction_interval=self.prediction_interval,
            predict_runtime=datetime.datetime.now() - predict_start_time,
            fit_runtime=self.fit_runtime,
            model_parameters=self.get_params(),
            components=components,
        )

    def _var_source_projection(self, n_var_inputs):
        """(n_var_inputs, n_sources) map from VAR model inputs to the series they come from.

        Without PCA each lag / rolling column belongs to one source series. With PCA the
        input is a principal component, mapped back through ``pca.components_`` exactly
        like TVVAR.process_components does before grouping by source series.
        """
        source_names = [name.split('___')[0] for name in self.VAR_feature_columns]
        unique_sources = list(dict.fromkeys(source_names))
        source_index = {name: i for i, name in enumerate(unique_sources)}
        membership = np.zeros((len(source_names), len(unique_sources)))
        membership[np.arange(len(source_names)), [source_index[n] for n in source_names]] = 1.0
        if self.apply_pca:
            return np.asarray(self.pca.components_, dtype=float) @ membership
        return membership[:n_var_inputs]

    def _components_frame(self, forecast_index, scaled_features, n_exogenous):
        """Per-series category totals, same values/layout as TVVAR's components.

        TVVAR groups contributions (feature * beta) by feature (exogenous) or by source
        series (VAR), rescales each group by the target's std (adding the mean to the
        constant), exponentiates groups in multiplicative mode, then sums by category.
        """
        feature_names = list(self.X.columns[:n_exogenous])
        n_series = len(self.column_names)
        scale = np.asarray(self.scaler_std, dtype=float)
        level = (
            np.asarray(self.scaler_mean, dtype=float)
            if np.ndim(self.scaler_mean)
            else np.zeros(n_series)
        )
        multiplicative = self.mode == 'multiplicative'
        category_of = {name: 'trend' for name in self.trend_columns}
        category_of.update({name: 'seasonality' for name in self.seasonal_columns})
        if str(self.regression_type).lower() == "user":
            category_of.update({name: 'regressors' for name in self.regressor_columns})
        category_of.update({'constant': 'constant', 'intercept': 'constant'})

        totals = {}
        # exogenous features: each column is its own group
        groups = (
            scaled_features[:, :n_exogenous, None] * self.beta[None, :n_exogenous, :]
        ) * scale
        for i, name in enumerate(feature_names):
            if category_of.get(name) == 'constant':
                groups[:, i, :] += level
        if multiplicative:
            groups = np.exp(groups)
        for i, name in enumerate(feature_names):
            category = category_of.get(name, 'var')
            totals[category] = totals.get(category, 0.0) + groups[:, i, :]

        n_var_inputs = scaled_features.shape[1] - n_exogenous
        if n_var_inputs > 0:
            projection = self._var_source_projection(n_var_inputs)
            var_inputs = scaled_features[:, n_exogenous:]
            var_beta = self.beta[n_exogenous:]
            if multiplicative:  # exp is applied per source series, so keep that axis
                per_source = np.einsum('hi,in,is->hsn', var_inputs, var_beta, projection)
                var_total = np.exp(per_source * scale).sum(axis=1)
            else:
                var_total = ((var_inputs * projection.sum(axis=1)) @ var_beta) * scale
            totals['var'] = totals.get('var', 0.0) + var_total

        order = ['trend', 'seasonality', 'regressors', 'var', 'constant', 'other']
        ordered = [c for c in order if c in totals] + [c for c in totals if c not in order]
        frames = {
            category: pd.DataFrame(
                totals[category], index=forecast_index, columns=self.column_names
            )
            for category in ordered
        }
        return stack_component_frames(frames)
