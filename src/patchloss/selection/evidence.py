"""Profile Gaussian linear evidence, with explicit numerical convergence records."""

import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from sklearn.covariance import ledoit_wolf
from .linear import normalize_fit
from .diagnostics import class_contrasts


def profile_objective(log_ratio, eigen, rhs, target_norm, n):
    ratio = np.exp(log_ratio)
    residual = target_norm - np.sum(rhs * rhs / (eigen + ratio))
    if residual <= 0:
        raise FloatingPointError("Nonpositive profiled residual")
    return float(-.5 * (n * (np.log(2 * np.pi) + 1 + np.log(residual / n))
                        + np.log1p(eigen / ratio).sum()) / n)


def profile_logme(features, labels):
    x, _, _ = normalize_fit(features)
    n = len(x)
    eigen, vectors = eigh(x.T @ x, check_finite=False)
    eigen = np.maximum(eigen, 0)
    targets = (np.asarray(labels)[:, None] == np.unique(labels)).astype(float)
    rhs = vectors.T @ x.T @ targets
    center = np.log(max(eigen.max(), 1e-12))
    grid = np.linspace(center - 30, center + 35, 131)
    details = []
    for j in range(targets.shape[1]):
        norm = float(targets[:, j] @ targets[:, j])
        def objective(t):
            return profile_objective(t, eigen, rhs[:, j], norm, n)
        values = np.array([objective(t) for t in grid])
        null = float(-.5 * (np.log(2 * np.pi) + 1 + np.log(norm / n)))
        best, ratio, mode = null, None, "analytic_null"
        refinements = []
        for i in range(1, len(grid) - 1):
            if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
                # Ignore machine-precision plateaus at the null boundary.
                if values[i] <= null + 1e-12:
                    continue
                fit = minimize_scalar(lambda t: -objective(t), bounds=(grid[i-1], grid[i+1]),
                                      method="bounded", options={"xatol": 1e-9, "maxiter": 300})
                if not fit.success:
                    raise RuntimeError("Profile evidence optimizer did not converge")
                refinements.append(dict(success=bool(fit.success), evaluations=int(fit.nfev)))
                if -fit.fun > best:
                    best, ratio, mode = float(-fit.fun), float(fit.x), "interior_profile"
        if values[0] > best + 1e-9 or values[-1] > best + 1e-9:
            raise RuntimeError("Profile evidence maximum at finite search boundary")
        beta = n / (norm - np.sum(rhs[:, j]**2 / (eigen + np.exp(ratio)))) if ratio is not None else n / norm
        details.append(dict(evidence=best, mode=mode, log_ratio=ratio, beta=float(beta),
                            search_bounds=[float(grid[0]), float(grid[-1])],
                            refinements=refinements, converged=True))
    return float(np.mean([d["evidence"] for d in details])), details


def adaptive_hscore(features, labels):
    x, _, _ = normalize_fit(features)
    covariance, shrinkage = ledoit_wolf(x, assume_centered=True)
    contrasts, _, _, _ = class_contrasts(x, labels)
    value = (1 - shrinkage) * np.sum(contrasts.T * cho_solve(cho_factor(covariance), contrasts.T))
    return float(value), float(shrinkage)
