"""Closed-form quantities of the paper, independent of any dataset.

All angles are in radians unless a name says otherwise. A *direction* is a
unit vector in the space of whitened channels; a *path* is an array of shape
(n, p) whose rows are directions at successive ages.

    loss(psi, gamma)            information loss of a fixed indicator, eq. (7)
    variance_inflation(...)     e^{2 loss}: error-variance factor, eq. (8)
    margin_inflation(...)       e^{loss}: safety-margin factor of a
                                predictive replacement policy (Section 2.3)
    arc_length(path)            length of the path on the sphere (Section 3.2)
    arc_bound(phi, gamma=None)  largest e^{2 loss} for an indicator placed at
                                the midpoint of an arc of length phi, eq. (12)
    barycentre(path, w=None)    renormalised (weighted) mean direction, the
                                fixed indicator of step 3
    karcher_mean(path, w)       intrinsic mean on the sphere (Supplement S2)
    max_angle(v, path)          largest angle between v and the path
"""
import numpy as np


def _unit(v, axis=-1):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=axis, keepdims=True)


def angle(a, b):
    """Angle between directions (or rows of two paths), in radians."""
    a, b = _unit(a), _unit(b)
    return np.arccos(np.clip((a * b).sum(-1), -1.0, 1.0))


def loss(psi, gamma):
    """Information loss (nats) of a fixed indicator at misalignment psi when
    the record has signal-to-noise ratio gamma = ||d||^2 Var[R]:

        l = 1/2 log[(1 + gamma) / (1 + gamma cos^2 psi)]
          = -1/2 log(1 - w sin^2 psi),   w = gamma / (1 + gamma).
    """
    psi = np.asarray(psi, dtype=float)
    g = np.asarray(gamma, dtype=float)
    w = g / (1.0 + g)
    return -0.5 * np.log(np.maximum(1.0 - w * np.sin(psi) ** 2, 1e-300))


def variance_inflation(psi, gamma):
    """Var[R | z] / Var[R | record] = e^{2 l}."""
    return np.exp(2.0 * loss(psi, gamma))


def margin_inflation(psi, gamma):
    """Factor by which the safety margin zeta_alpha s_R of a predictive
    replacement policy widens when the fixed indicator replaces the record:
    e^{l}, the ratio of the posterior standard deviations."""
    return np.exp(loss(psi, gamma))


def arc_length(path):
    """Length on the unit sphere of a path given as rows of directions."""
    P = _unit(path)
    return float(np.arccos(np.clip((P[1:] * P[:-1]).sum(1), -1.0, 1.0)).sum())


def arc_bound(phi, gamma=None):
    """Upper bound on e^{2 l} for an indicator within phi/2 of every point of
    a path of length phi (an indicator at the path's midpoint qualifies):

        e^{2 l} <= 1 / (1 - w sin^2(phi/2)) <= 1 / cos^2(phi/2).

    gamma=None gives the right-hand, gamma-free bound.
    """
    h = np.asarray(phi, dtype=float) / 2.0
    if gamma is None:
        return 1.0 / np.cos(h) ** 2
    w = np.asarray(gamma, dtype=float) / (1.0 + np.asarray(gamma, dtype=float))
    return 1.0 / (1.0 - w * np.sin(h) ** 2)


def barycentre(path, weights=None):
    """Renormalised (weighted) mean of the directions of a path: the fixed
    indicator of step 3 of the design procedure."""
    P = _unit(path)
    w = np.ones(len(P)) if weights is None else np.asarray(weights, float)
    return _unit((w[:, None] * P).sum(0))


def karcher_mean(path, weights, iters=500):
    """Weighted intrinsic (Karcher) mean of directions on the sphere."""
    DH = _unit(path)
    mu = np.asarray(weights, dtype=float)
    b = (mu[:, None] * DH).sum(0)
    b /= np.linalg.norm(b)
    for _ in range(iters):
        cth = np.clip(DH @ b, -1, 1)
        R = DH - np.outer(cth, b)
        nr = np.linalg.norm(R, axis=1, keepdims=True)
        V = np.where(nr > 1e-14, R / np.maximum(nr, 1e-300), 0.0) \
            * np.arccos(cth)[:, None]
        v = (mu[:, None] * V).sum(0)
        nv = np.linalg.norm(v)
        if nv < 1e-15:
            break
        b = np.cos(nv) * b + np.sin(nv) * v / nv
        b /= np.linalg.norm(b)
    return b


def max_angle(v, path):
    """Largest angle (radians) between the indicator v and the path."""
    return float(angle(np.asarray(v)[None, :], path).max())


def path_midpoint(path):
    """The direction halfway along the path by arc length (interpolated along
    the great circle between the two samples that bracket it). Every point
    of the path lies within arc/2 of it, so an indicator placed there is
    always covered by the arc bound (12)."""
    P = _unit(path)
    seg = np.arccos(np.clip((P[1:] * P[:-1]).sum(1), -1.0, 1.0))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    half = cum[-1] / 2.0
    k = int(min(np.searchsorted(cum, half, side="right") - 1, len(seg) - 1))
    if seg[k] < 1e-15:
        return P[k]
    s = (half - cum[k]) / seg[k]
    om = seg[k]
    m = (np.sin((1 - s) * om) * P[k] + np.sin(s * om) * P[k + 1]) / np.sin(om)
    return m / np.linalg.norm(m)


def bound_for(v, path):
    """The bound for a GIVEN indicator v: e^{2 l} <= 1 / cos^2(psi_max), with
    psi_max its largest angle to the path (the arc bound is this bound for
    an indicator within arc/2)."""
    return 1.0 / np.cos(max_angle(v, path)) ** 2
