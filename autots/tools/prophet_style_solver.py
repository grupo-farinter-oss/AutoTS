"""Closed-form (IRLS) MAP solver replacing Stan's L-BFGS for Prophet-style models.

Prophet's posterior mode has Normal priors on seasonality/holiday/regressor weights
and a Laplace prior on trend changepoint deltas. Normal priors are ridge penalties.
The Laplace term |d| is majorised by d^2 / (2|d0|) so each iteration is a weighted
ridge solve; all series are solved in one batched ``np.linalg.solve``.
"""

import numpy as np
from autots.tools.batched_linear_solvers import solve_ridge_batched

_LAPLACE_FLOOR = 1e-4  # keeps IRLS finite; deltas below this are effectively zero


def _normal_equations(design, targets, weights):
    """Return gram (k,k)|(c,k,k) and cross (c,k); ``weights`` is (n,c) or None (all 1)."""
    if weights is None:
        if design.ndim == 2:
            return design.T @ design, (design.T @ targets).T
        weights = np.ones(targets.shape)
    if design.ndim == 2:
        # one BLAS matmul over the k(k+1)/2 unique column products instead of a
        # (c, n, k) intermediate: weights.T @ [x_a * x_b]
        k = design.shape[1]
        row_idx, col_idx = np.triu_indices(k)
        pair_products = design[:, row_idx] * design[:, col_idx]
        packed = weights.T @ pair_products
        gram = np.empty((targets.shape[1], k, k))
        gram[:, row_idx, col_idx] = packed
        gram[:, col_idx, row_idx] = packed
        return gram, (weights * targets).T @ design
    weighted = design * weights.T[:, :, None]
    gram = np.matmul(weighted.transpose(0, 2, 1), design)
    return gram, np.einsum('cnk,nc->ck', weighted, targets)


def apply_design(design, coefficients):
    """(n,k)|(c,n,k) times (c,k) -> (n,c)."""
    if design.ndim == 2:
        return design @ coefficients.T
    return np.einsum('cnk,ck->nc', design, coefficients)


def penalised_least_squares(
    design,
    targets,
    observed,
    normal_prior_std,
    laplace_columns=None,
    laplace_scale=0.05,
    n_iterations=6,
    weights=None,
):
    """MAP fit for all series.

    Args:
        design: (n, k) shared or (c, n, k) per-series regressors
        targets: (n, c) with zeros where unobserved
        observed: (n, c) boolean mask
        normal_prior_std: (k,) prior std per column (np.inf == no ridge)
        laplace_columns: slice of columns under a Laplace(0, laplace_scale) prior
        weights: optional (n, c) observation weights (already zero where unobserved);
            used by the multiplicative backfit, where the exact objective is weighted
    Returns:
        coefficients (c, k), noise variance (c,)
    """
    n_series = targets.shape[1]
    n_columns = design.shape[-1]
    if weights is None and not observed.all():
        weights = observed.astype(float)
    gram, cross = _normal_equations(design, targets, weights)
    n_obs = np.maximum(observed.sum(axis=0), 1)
    sigma2 = np.maximum(0.01 * (targets**2).sum(axis=0) / n_obs, 1e-8)
    with np.errstate(divide='ignore'):
        ridge_weight = np.where(np.isfinite(normal_prior_std), 1.0 / normal_prior_std**2, 0.0)
    delta_magnitude = None
    if laplace_columns is not None:
        n_laplace = len(range(*laplace_columns.indices(n_columns)))
        if n_laplace:
            delta_magnitude = np.full((n_series, n_laplace), laplace_scale)
    coefficients = np.zeros((n_series, n_columns))
    iterations = n_iterations if delta_magnitude is not None else 2
    for _ in range(iterations):
        penalty = sigma2[:, None] * ridge_weight[None, :]
        if delta_magnitude is not None:
            penalty[:, laplace_columns] = sigma2[:, None] / (
                laplace_scale * np.maximum(delta_magnitude, _LAPLACE_FLOOR)
            )
        coefficients = solve_ridge_batched(gram, cross, penalty)
        residual = (targets - apply_design(design, coefficients)) * observed
        if weights is not None:
            residual = residual * np.sqrt(weights)
        sigma2 = np.maximum((residual**2).sum(axis=0) / n_obs, 1e-10)
        if delta_magnitude is not None:
            delta_magnitude = np.abs(coefficients[:, laplace_columns])
    return coefficients, sigma2
