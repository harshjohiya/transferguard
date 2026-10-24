"""Classification LogME using the official later fixed-point formulation.

This follows ``LogME._fit_fixed_point`` in thuml/LogME (the JMLR 2022
extension), rather than its separate ICML 2021 ``_fit_icml`` routine. We use
direct float64 SVD, a rank threshold, convergence tolerance and bounded
positive precisions for numerical safety. The Bayesian evidence itself is
unchanged. No feature normalization or intercept is added.
"""

import numpy as np


def _evidence_for_class(y, singular, projected, rank, dimension, count):
    sigma = singular * singular
    projection_square = projected * projected
    perpendicular = max(0.0, float(y @ y - projection_square.sum()))
    alpha = beta = 1.0
    for _ in range(128):
        ratio = alpha / beta
        gamma = float(np.sum(sigma / (sigma + ratio)))
        mean_square = float(np.sum(sigma * projection_square / (sigma + ratio) ** 2))
        residual = perpendicular + float(np.sum(projection_square * (ratio / (sigma + ratio)) ** 2))
        next_alpha = np.clip(gamma / max(mean_square, 1e-12), 1e-12, 1e12) if rank else 1.0
        next_beta = np.clip((count - gamma) / max(residual, 1e-12), 1e-12, 1e12)
        change = max(abs(np.log(next_alpha / alpha)), abs(np.log(next_beta / beta)))
        alpha, beta = float(next_alpha), float(next_beta)
        if change < 1e-7:
            break

    ratio = alpha / beta
    mean_square = float(np.sum(sigma * projection_square / (sigma + ratio) ** 2))
    residual = perpendicular + float(np.sum(projection_square * (ratio / (sigma + ratio)) ** 2))
    # D-r null directions contribute log(alpha) to log|alpha I + beta X'X|.
    log_determinant = (dimension - rank) * np.log(alpha) + np.log(alpha + beta * sigma).sum()
    evidence = (dimension * np.log(alpha) + count * np.log(beta) - log_determinant
                - beta * residual - alpha * mean_square - count * np.log(2 * np.pi)) / (2 * count)
    if not np.isfinite(evidence):
        raise FloatingPointError("LogME evidence is non-finite")
    return float(evidence)


def logme_score(features, labels):
    """Mean per-example, class-wise one-vs-rest log evidence (higher is better)."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    if x.ndim != 2 or min(x.shape) == 0 or y.ndim != 1 or len(y) != len(x):
        raise ValueError("Expected nonempty features [N,D] and labels [N]")
    if not np.isfinite(x).all() or not np.issubdtype(y.dtype, np.integer) or (y < 0).any():
        raise ValueError("Features must be finite and labels nonnegative integers")
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("LogME classification requires at least two classes")
    try:
        u, singular, _ = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise FloatingPointError("LogME SVD did not converge") from error
    tolerance = max(x.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    u, singular = u[:, :rank], singular[:rank]
    scores = []
    for label in classes:
        binary = (y == label).astype(np.float64)
        scores.append(_evidence_for_class(binary, singular, u.T @ binary,
                                          rank, x.shape[1], x.shape[0]))
    result = float(np.mean(scores))
    if not np.isfinite(result):
        raise FloatingPointError("Mean LogME score is non-finite")
    return result
