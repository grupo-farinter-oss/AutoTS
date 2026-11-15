"""FastARIMA: drop-in, vectorised replacement for the statsmodels-based ``ARIMA``.

Why it is fast: instead of one numerical maximum-likelihood fit per series (SARIMAX +
Kalman filter), every series is estimated simultaneously with batched least squares:
    1. difference d times (exog differenced the same way, i.e. regression with ARIMA errors)
    2. long-AR regression -> proxy innovations           (Hannan-Rissanen stage 1)
    3. regress on lags of the series and of the proxy innovations (stage 2)
    4. Levenberg-Marquardt on conditional sum of squares (per-series damping, only
       accepts improving, stable steps)
The only python loops are over time steps / horizon (never over series).

Trade-off: estimates are conditional least squares, not exact Gaussian MLE, so they
differ slightly from statsmodels but are consistent and typically within noise on
forecast accuracy, at a fraction of the cost. Series where the fit is non-finite
degrade to a zero-parameter model (random walk / mean) rather than failing the batch.
"""

import random
import datetime
import numpy as np
import pandas as pd
from autots.models.base import ModelObject, PredictionObject
from autots.tools.holiday import holiday_flag
from autots.tools.batched_linear_solvers import (
    fill_missing_columnwise,
    normal_quantile,
    sliding_lag_design,
    solve_ridge_batched,
)

_RIDGE_RELATIVE = 1e-8  # numerical guard only; not a shrinkage prior
_ROOT_MARGIN = 0.995  # radius that unstable fits are pulled back to


def _difference(values: np.ndarray, order: int):
    """Return d-times differenced array and the last value of every intermediate level."""
    last_values = []
    current = values
    for _ in range(order):
        last_values.append(current[-1].copy())
        current = np.diff(current, axis=0)
    return current, last_values


def _integrate(forecast_differenced: np.ndarray, last_values: list) -> np.ndarray:
    result = forecast_differenced
    for last in reversed(last_values):
        result = last[None, :] + np.cumsum(result, axis=0)
    return result


def _lagged_normal_equations(design: np.ndarray, target: np.ndarray):
    """(N, rows, k), (N, rows) -> per-series gram and cross-product."""
    gram = np.matmul(design.transpose(0, 2, 1), design)
    cross = np.einsum('nrk,nr->nk', design, target)
    return gram, cross


def _least_squares_per_series(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    gram, cross = _lagged_normal_equations(design, target)
    scale = np.trace(gram, axis1=1, axis2=2) / max(design.shape[2], 1)
    penalty = (_RIDGE_RELATIVE * scale + 1e-12)[:, None]
    return solve_ridge_batched(gram, cross, penalty)


def _stabilise_polynomial(coefficients: np.ndarray, sign: float) -> np.ndarray:
    """Pull coefficients inside the unit circle by scaling all roots uniformly.

    sign=+1 for AR (stationarity), sign=-1 for MA (invertibility: companion of -theta).
    """
    n_series, order = coefficients.shape
    if order == 0:
        return coefficients
    companion = np.zeros((n_series, order, order))
    companion[:, 0, :] = sign * coefficients
    if order > 1:
        companion[:, np.arange(1, order), np.arange(order - 1)] = 1.0
    finite = np.isfinite(coefficients).all(axis=1)
    companion[~finite] = 0.0
    radius = np.abs(np.linalg.eigvals(companion)).max(axis=1)
    shrink = np.where(radius > _ROOT_MARGIN, _ROOT_MARGIN / np.maximum(radius, 1e-12), 1.0)
    powers = shrink[:, None] ** np.arange(1, order + 1)[None, :]
    stabilised = coefficients * powers
    stabilised[~finite] = 0.0
    return stabilised


def _arma_innovations(u: np.ndarray, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Exact conditional ARMA recursion, vectorised over series (loop over time only)."""
    n_rows, n_series = u.shape
    p, q = phi.shape[1], theta.shape[1]
    innovations = np.zeros((n_rows, n_series))
    for t in range(n_rows):
        prediction = np.zeros(n_series)
        if p and t:
            span = min(p, t)
            prediction += np.einsum('ni,in->n', phi[:, :span], u[t - span : t][::-1])
        if q and t:
            span = min(q, t)
            prediction += np.einsum(
                'ni,in->n', theta[:, :span], innovations[t - span : t][::-1]
            )
        innovations[t] = u[t] - prediction
    return np.clip(innovations, -1e12, 1e12)


def fit_arma_batched(u: np.ndarray, p: int, q: int, refinement_iterations: int = 2):
    """Estimate ARMA(p, q) for every column of ``u`` (already differenced / demeaned)."""
    n_rows, n_series = u.shape
    if p == 0 and q == 0:
        return np.zeros((n_series, 0)), np.zeros((n_series, 0))
    long_order = int(min(max(p + q + 2, 8), max(n_rows // 4, 1)))
    if q == 0:  # pure AR: one exact conditional least squares solve
        lags = sliding_lag_design(u, p)
        return _least_squares_per_series(lags, u[p:].T), np.zeros((n_series, 0))

    # stage 1: long AR -> proxy innovations
    lags_long = sliding_lag_design(u, long_order)
    ar_long = _least_squares_per_series(lags_long, u[long_order:].T)
    proxy = np.zeros_like(u)
    proxy[long_order:] = u[long_order:] - np.einsum('nrk,nk->rn', lags_long, ar_long)

    def stage_two(innovations, start):
        design = np.concatenate(
            [
                sliding_lag_design(u, p)[:, start - p :, :] if p else
                np.zeros((n_series, n_rows - start, 0)),
                sliding_lag_design(innovations, q)[:, start - q :, :],
            ],
            axis=2,
        )
        return _least_squares_per_series(design, u[start:].T)

    start = max(long_order + q, p, q)
    if n_rows - start < p + q + 2:  # too short for HR: fall back to AR-only
        lags = sliding_lag_design(u, max(p, 1))
        phi = _least_squares_per_series(lags, u[max(p, 1) :].T)
        return _stabilise_polynomial(phi, 1.0)[:, :p], np.zeros((n_series, q))
    coefficients = stage_two(proxy, start)
    phi = _stabilise_polynomial(coefficients[:, :p], 1.0)
    theta = _stabilise_polynomial(coefficients[:, p:], -1.0)
    return _refine_css_levenberg_marquardt(u, phi, theta, refinement_iterations)


def _css_residuals_and_jacobian(u, phi, theta, with_jacobian=True):
    """Conditional innovations e_t and d e_t / d(phi, theta), vectorised over series.

    The derivative of e w.r.t. each parameter obeys the same MA recursion as e itself,
    so all p+q derivative channels are propagated together in one pass over time.
    """
    n_rows, n_series = u.shape
    p, q = phi.shape[1], theta.shape[1]
    n_params = p + q
    innovations = np.zeros((n_rows, n_series))
    jacobian = np.zeros((n_rows, n_series, n_params)) if with_jacobian else None
    for t in range(n_rows):
        prediction = np.zeros(n_series)
        span_p, span_q = min(p, t), min(q, t)
        if span_p:
            prediction += np.einsum('ni,in->n', phi[:, :span_p], u[t - span_p : t][::-1])
        if span_q:
            prediction += np.einsum(
                'ni,in->n', theta[:, :span_q], innovations[t - span_q : t][::-1]
            )
        innovations[t] = u[t] - prediction
        if with_jacobian:
            row = jacobian[t]
            if span_p:
                row[:, :span_p] = -u[t - span_p : t][::-1].T
            if span_q:
                row[:, p : p + span_q] = -innovations[t - span_q : t][::-1].T
                row -= np.einsum(
                    'ni,ink->nk', theta[:, :span_q], jacobian[t - span_q : t][::-1]
                )
    return innovations, jacobian


def _refine_css_levenberg_marquardt(u, phi, theta, n_iterations):
    """Polish Hannan-Rissanen starts by minimising conditional sum of squares.

    Per-series damping, so stubborn series do not stall the rest of the batch. A step
    is accepted only if it lowers that series' SSE after re-imposing stationarity /
    invertibility, so the result can never be worse than the HR starting point.
    """
    p, q = phi.shape[1], theta.shape[1]
    burn = max(p, q)
    n_series = u.shape[1]
    damping = np.full(n_series, 1e-3)
    innovations, jacobian = _css_residuals_and_jacobian(u, phi, theta)
    sse = (innovations[burn:] ** 2).sum(axis=0)
    for _ in range(n_iterations):
        j_used, e_used = jacobian[burn:], innovations[burn:]
        normal = np.einsum('tnk,tnl->nkl', j_used, j_used)
        gradient = np.einsum('tnk,tn->nk', j_used, e_used)
        diag_idx = np.arange(p + q)
        trial_normal = normal.copy()
        trial_normal[:, diag_idx, diag_idx] *= 1.0 + damping[:, None]
        trial_normal[:, diag_idx, diag_idx] += 1e-12
        step = -solve_ridge_batched(trial_normal, gradient)
        phi_new = _stabilise_polynomial(phi + step[:, :p], 1.0)
        theta_new = _stabilise_polynomial(theta + step[:, p:], -1.0)
        new_innovations, _ = _css_residuals_and_jacobian(
            u, phi_new, theta_new, with_jacobian=False
        )
        new_sse = (new_innovations[burn:] ** 2).sum(axis=0)
        accept = np.isfinite(new_sse) & (new_sse < sse * (1.0 - 1e-9))
        if not accept.any() and (damping > 1e6).all():
            break
        converged = accept & (sse - new_sse < 1e-7 * sse)
        phi = np.where(accept[:, None], phi_new, phi)
        theta = np.where(accept[:, None], theta_new, theta)
        damping = np.where(accept, np.maximum(damping / 4.0, 1e-9), damping * 8.0)
        if accept.any():
            innovations, jacobian = _css_residuals_and_jacobian(u, phi, theta)
            sse = (innovations[burn:] ** 2).sum(axis=0)
        if converged.all() or (~accept & (damping > 1e6)).all():
            break
    return phi, theta


def _ar_filter(values: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """phi(B) applied per series. values (T, N) or (T, N, r); phi (N, p). Drops first p rows."""
    p = phi.shape[1]
    if p == 0:
        return values
    n_rows = values.shape[0]
    filtered = values[p:].copy()
    for lag in range(1, p + 1):
        weights = phi[:, lag - 1].reshape((1, -1) + (1,) * (values.ndim - 2))
        filtered -= weights * values[p - lag : n_rows - lag]
    return filtered


class FastARIMA(ModelObject):
    """Vectorised ARIMA (conditional least squares / Hannan-Rissanen), all series at once.

    Same parameters as ``ARIMA`` so templates are interchangeable.

    Args:
        p, d, q (int): AR order, differencing order, MA order
        regression_type (str): None, 'User' or 'Holiday' (regression with ARIMA errors)
        demean_when_stationary (bool): d=0 only; if True revert to the sample mean
            instead of zero (SARIMAX default behaviour, kept as default for parity)
        refinement_iterations (int): Levenberg-Marquardt CSS iterations when q > 0
    """

    def __init__(
        self,
        name: str = "FastARIMA",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        p: int = 0,
        d: int = 1,
        q: int = 0,
        regression_type: str = None,
        holiday_country: str = 'US',
        refinement_iterations: int = 8,
        demean_when_stationary: bool = False,
        random_seed: int = 2020,
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
        self.p = p
        self.d = d
        self.q = q
        self.order = (p, d, q)
        self.refinement_iterations = refinement_iterations
        self.demean_when_stationary = demean_when_stationary

    def fit(self, df, future_regressor=None):
        df = self.basic_profile(df)
        self.regressor_train = None
        method_str = str(self.regression_type).lower()
        if method_str == 'holiday':
            self.regressor_train = holiday_flag(
                df.index, country=self.holiday_country
            ).to_numpy(dtype=float)
        elif method_str == "user":
            if future_regressor is None:
                raise ValueError(
                    "regression_type='User' but future_regressor not supplied"
                )
            self.regressor_train = np.asarray(future_regressor, dtype=float)
            if self.regressor_train.ndim == 1:
                self.regressor_train = self.regressor_train[:, None]
        self.df_train = df
        self._fit_parameters(fill_missing_columnwise(df.to_numpy(dtype=float)))
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def _fit_parameters(self, levels: np.ndarray):
        n_series = levels.shape[1]
        w, self._last_levels = _difference(levels, self.d)
        if w.shape[0] < max(self.p, self.q) + 5:
            raise ValueError("FastARIMA: not enough history for requested order")
        # statsmodels SARIMAX default is a zero-mean process; keep that for parity
        use_mean = self.demean_when_stationary and self.d == 0
        self._mean = w.mean(axis=0) if use_mean else np.zeros(n_series)
        u = w - self._mean
        exog = None
        if self.regressor_train is not None:
            exog_d, _ = _difference(self.regressor_train, self.d)
            self._exog_mean = exog_d.mean(0) if use_mean else np.zeros(exog_d.shape[1])
            exog = exog_d - self._exog_mean
            spread = exog.std(axis=0)
            self._exog_scale = np.where(spread > 0, spread, 1.0)
            exog = exog / self._exog_scale
        self.beta = np.zeros((n_series, 0 if exog is None else exog.shape[1]))
        phi = np.zeros((n_series, 0))
        theta = np.zeros((n_series, 0))
        residual = u
        # feasible-GLS style alternation: beta | ARMA filter, then ARMA | beta
        rounds = 2 if exog is not None else 1
        for _ in range(rounds):
            if exog is not None:
                self.beta = self._regression_coefficients(u, exog, phi)
                residual = u - exog @ self.beta.T
            phi, theta = fit_arma_batched(
                residual, self.p, self.q, self.refinement_iterations
            )
        self.phi, self.theta = phi, theta
        innovations = _arma_innovations(residual, phi, theta)
        burn = min(max(self.p, self.q), residual.shape[0] // 2)
        self.sigma2 = np.mean(innovations[burn:] ** 2, axis=0)
        self._history_tail = residual[-max(self.p, 1) :]
        self._innovation_tail = innovations[-max(self.q, 1) :]

    @staticmethod
    def _regression_coefficients(u, exog, phi):
        n_series = u.shape[1]
        n_regressors = exog.shape[1]
        if phi.shape[1]:  # GLS-ish: filter both sides with the current AR polynomial
            y_f = _ar_filter(u, phi)  # (T-p, N)
            x_f = _ar_filter(
                np.broadcast_to(exog[:, None, :], (exog.shape[0], n_series, n_regressors)),
                phi,
            )  # (T-p, N, r)
            design = x_f.transpose(1, 0, 2)
            return _least_squares_per_series(design, y_f.T)
        design = np.broadcast_to(exog[None], (n_series,) + exog.shape)
        return _least_squares_per_series(design, u.T)

    def _future_exog(self, test_index, future_regressor):
        method_str = str(self.regression_type).lower()
        if method_str == 'holiday':
            future = holiday_flag(test_index, country=self.holiday_country).to_numpy(
                dtype=float
            )
        else:
            if future_regressor is None:
                raise ValueError("future_regressor required for regression_type='User'")
            future = np.asarray(future_regressor, dtype=float)
            if future.ndim == 1:
                future = future[:, None]
        # exog entered the fit differenced; forecast in the differenced domain by
        # differencing the (train tail + future) stitched series d times.
        stitched = np.vstack([self.regressor_train, future])
        stitched_d, _ = _difference(stitched, self.d)
        future_d = stitched_d[-future.shape[0] :]
        return (future_d - self._exog_mean) / self._exog_scale

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        predictStartTime = datetime.datetime.now()
        test_index = self.create_forecast_index(forecast_length=forecast_length)
        n_series = self.phi.shape[0]
        p, q = self.phi.shape[1], self.theta.shape[1]

        u_buffer = self._history_tail[::-1][:p].T.copy()  # (N, p) newest first
        e_buffer = self._innovation_tail[::-1][:q].T.copy()
        arma_forecast = np.zeros((forecast_length, n_series))
        for step in range(forecast_length):
            value = np.zeros(n_series)
            if p:
                value += (self.phi * u_buffer).sum(axis=1)
            if q:
                value += (self.theta * e_buffer).sum(axis=1)
                e_buffer = np.concatenate([np.zeros((n_series, 1)), e_buffer[:, :-1]], 1)
            if p:
                u_buffer = np.concatenate([value[:, None], u_buffer[:, :-1]], axis=1)
            arma_forecast[step] = value
        # AR errors persist across the exog fit, so exog effect is added on top
        differenced_forecast = arma_forecast + self._mean
        if self.regressor_train is not None:
            exog_future = self._future_exog(test_index, future_regressor)
            differenced_forecast = differenced_forecast + exog_future @ self.beta.T
        point = _integrate(differenced_forecast, self._last_levels)

        # psi weights of the integrated process -> forecast error variance
        psi = np.zeros((forecast_length, n_series))
        psi[0] = 1.0
        for j in range(1, forecast_length):
            value = self.theta[:, j - 1].copy() if j <= q else np.zeros(n_series)
            for i in range(1, min(j, p) + 1):
                value += self.phi[:, i - 1] * psi[j - i]
            psi[j] = value
        for _ in range(self.d):
            psi = np.cumsum(psi, axis=0)
        standard_error = np.sqrt(np.cumsum(psi**2, axis=0) * self.sigma2[None, :])
        z_value = normal_quantile(1 - (1 - self.prediction_interval) / 2)

        columns = self.column_names
        forecast = pd.DataFrame(point, index=test_index, columns=columns)
        if just_point_forecast:
            return forecast
        lower_forecast = pd.DataFrame(
            point - z_value * standard_error, index=test_index, columns=columns
        )
        upper_forecast = pd.DataFrame(
            point + z_value * standard_error, index=test_index, columns=columns
        )
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

    def get_new_params(self, method: str = 'random'):
        """Same search space as ARIMA, so tuned templates transfer one-to-one."""
        p_choice = random.choices(
            [0, 1, 2, 3, 4, 5, 7, 12], [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01]
        )[0]
        d_choice = random.choices([0, 1, 2, 3], [0.4, 0.3, 0.2, 0.1])[0]
        q_choice = random.choices(
            [0, 1, 2, 3, 4, 5, 7, 12], [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01]
        )[0]
        if "regressor" in method:
            regression_choice = "User"
        else:
            regression_choice = random.choices(
                [None, 'User', 'Holiday'], [0.5, 0.3, 0.2]
            )[0]
        return {
            'p': p_choice,
            'd': d_choice,
            'q': q_choice,
            'regression_type': regression_choice,
        }

    def get_params(self):
        return {
            'p': self.p,
            'd': self.d,
            'q': self.q,
            'regression_type': self.regression_type,
        }
