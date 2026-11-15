"""Batched (many-series-at-once) linear algebra helpers for the Fast* models.

Why: statsmodels/Stan fit one series at a time in Python. All of the fast models
reduce a fit to (penalised) least squares, which numpy can solve for every column
of a wide dataframe in a handful of BLAS calls.
"""

import numpy as np

try:
    from scipy.stats import norm as _scipy_norm

    def normal_quantile(probability: float) -> float:
        return float(_scipy_norm.ppf(probability))

except Exception:  # scipy is optional in AutoTS
    from autots.tools.mocks import norm as _mock_norm

    def normal_quantile(probability: float) -> float:
        return float(_mock_norm().ppf(probability))


def fill_missing_columnwise(values: np.ndarray) -> np.ndarray:
    """Linear interpolate interior NaNs, edge-fill the ends, all-NaN columns -> 0.

    Vectorised across columns by index arithmetic (no per-column python loop for the
    common no-NaN case; columns that contain NaN are handled with np.interp).
    """
    values = np.asarray(values, dtype=float)
    nan_mask = np.isnan(values)
    if not nan_mask.any():
        return values
    filled = values.copy()
    row_positions = np.arange(values.shape[0])
    for col in np.flatnonzero(nan_mask.any(axis=0)):
        valid = ~nan_mask[:, col]
        if not valid.any():
            filled[:, col] = 0.0
        else:
            filled[:, col] = np.interp(
                row_positions, row_positions[valid], values[valid, col]
            )
    return filled


def solve_ridge_batched(
    gram: np.ndarray, cross: np.ndarray, penalty: np.ndarray = None
) -> np.ndarray:
    """Solve (G_j + diag(penalty_j)) beta_j = c_j for every series j at once.

    Args:
        gram: (k, k) shared or (n_series, k, k) per-series X'WX
        cross: (n_series, k) X'Wy
        penalty: (k,) or (n_series, k) ridge diagonal, or None
    Falls back to lstsq per failing series so one singular fit cannot kill the batch.
    """
    n_series, k = cross.shape
    if gram.ndim == 2:
        lhs = np.broadcast_to(gram, (n_series, k, k)).copy()
    else:
        lhs = gram.copy()
    if penalty is not None:
        diag_idx = np.arange(k)
        lhs[:, diag_idx, diag_idx] += penalty
    try:
        return np.linalg.solve(lhs, cross[..., None])[..., 0]
    except np.linalg.LinAlgError:
        beta = np.zeros((n_series, k))
        for j in range(n_series):
            beta[j] = np.linalg.lstsq(lhs[j], cross[j], rcond=None)[0]
        return beta


def sliding_lag_design(series_matrix: np.ndarray, n_lags: int) -> np.ndarray:
    """Lag matrix for all series. (T, N) -> (N, T - n_lags, n_lags); lag 1 first.

    Uses a strided view then one copy, so cost is a single pass over the data.
    """
    n_rows, n_series = series_matrix.shape
    windows = np.lib.stride_tricks.sliding_window_view(
        series_matrix, window_shape=n_lags, axis=0
    )  # (T - n_lags + 1, N, n_lags) with lag n_lags .. 1 ordering (oldest first)
    lagged = windows[: n_rows - n_lags][:, :, ::-1]  # drop last window; newest first
    return np.ascontiguousarray(lagged.transpose(1, 0, 2))
