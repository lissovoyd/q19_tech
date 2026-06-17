"""Pick a duplicate/not-duplicate cut from the *shape* of the score
distribution -- never from the gold labels. This is what keeps threshold
selection honest (cf. the rejected "tune tau on the eval set" shortcut).

Otsu's method: assume the scores are a mix of two classes (duplicates high,
non-duplicates low) and choose the cut that maximizes between-class variance.
No labels involved -- it only looks at the scores themselves.
"""
import numpy as np


def otsu_threshold(scores, bins: int = 512) -> float:
    scores = np.asarray(scores, dtype=float)
    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        return lo

    hist, edges = np.histogram(scores, bins=bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    w = hist.astype(float)
    total = w.sum()

    omega = np.cumsum(w) / total                # class-0 weight up to each cut
    mu = np.cumsum(w * centers) / total          # class-0 cumulative mean*weight
    mu_t = mu[-1]                                 # global mean

    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    sigma_b2[~np.isfinite(sigma_b2)] = -1.0
    return float(centers[int(np.argmax(sigma_b2))])
