"""Steps 1-3 of the design procedure on run-to-failure data.

A *fleet* is a list of units; a unit is an array of shape (n, p): one row per
cycle (or inspection), from the start of service to failure, one column per
monitoring channel. Each unit's life is normalised to tau = cycle / (n - 1).

Step 1  observed_direction(): smooth every channel with a Savitzky-Golay
        filter, differentiate with respect to normalised life, whiten each
        channel by the scatter of its readings about the smoothed trend,
        orient the channels by the sign of the fleet median, normalise each
        unit's derivative to a unit vector, and average over the units on a
        common grid of tau.
Step 2  FleetDirection.arc / .bound: length of the path the observed
        direction traverses and the bound 1/cos^2(arc/2) on the error-variance
        inflation of a well-placed fixed indicator.
Step 3  FleetDirection.indicator: the barycentre (renormalised mean of the
        path), with its largest angle to the path, which must not exceed
        arc/2.

The defaults are those of the paper's battery case study (Section 4): a
cubic filter, window min(301, 2 floor(n/4) + 1) cycles, tau in [0.15, 0.90]
on 40 points.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter

from . import core

DEFAULT_GRID = np.linspace(0.15, 0.90, 40)


def unit_window(n, window, cap_divisor=4):
    """Savitzky-Golay window for a unit of n cycles: the requested window,
    capped at 2 floor(n / cap_divisor) + 1 so short units are not
    over-smoothed."""
    return min(window, (n // cap_divisor) * 2 + 1)


def unit_derivative(Y, window=301, grid=DEFAULT_GRID, polyorder=3,
                    cap_divisor=4):
    """Whitened derivative of one unit's channels with respect to normalised
    life, interpolated on grid.

    Returns (D, sigma, w): D of shape (len(grid), p), the per-channel noise
    levels sigma (standard deviation of the readings about the smoothed
    trend), and the window used.
    """
    Y = np.asarray(Y, dtype=float)
    n = len(Y)
    tau = np.arange(n) / (n - 1.0)
    w = unit_window(n, window, cap_divisor)
    D, S = [], []
    for j in range(Y.shape[1]):
        y = Y[:, j]
        D.append(savgol_filter(y, w, polyorder, deriv=1) * (n - 1))
        S.append(np.std(y - savgol_filter(y, w, polyorder)) + 1e-12)
    Dm = np.stack(D, 1) / np.array(S)[None, :]
    return (np.stack([np.interp(grid, tau, Dm[:, c]) for c in range(Y.shape[1])], 1),
            np.array(S), w)


def orientation(R):
    """Sign of each channel that makes the fleet's median derivative
    positive: an array of +1/-1 of length p, for R of shape (units, grid, p)."""
    sg = np.sign(np.median(R.reshape(-1, R.shape[2]), axis=0))
    sg[sg == 0] = 1.0
    return sg


def fleet_mean_path(DH):
    """Renormalised mean over units of unit directions DH (units, grid, p)."""
    M = DH.mean(0)
    return M / np.linalg.norm(M, axis=1, keepdims=True)


@dataclass
class FleetDirection:
    """The observed direction of a fleet and what steps 2 and 3 derive
    from it."""
    grid: np.ndarray
    path: np.ndarray                 # (len(grid), p) fleet-mean direction
    unit_directions: np.ndarray      # (units, len(grid), p), oriented, unit
    signs: np.ndarray                # channel orientation, +1/-1
    channels: tuple = field(default=())

    @property
    def arc(self):
        """Length of the path traversed, radians."""
        return core.arc_length(self.path)

    @property
    def arc_deg(self):
        return float(np.degrees(self.arc))

    @property
    def end_to_end_deg(self):
        """Angle between the first and last direction of the path."""
        return float(np.degrees(core.angle(self.path[0], self.path[-1])))

    def bound(self, gamma=None):
        """Largest error-variance inflation e^{2 l} of a well-placed fixed
        indicator, eq. (12); gamma=None gives the gamma-free 1/cos^2 bound."""
        return float(core.arc_bound(self.arc, gamma))

    @property
    def indicator(self):
        """Step 3: the barycentre of the path (in whitened, oriented
        coordinates)."""
        return core.barycentre(self.path)

    def max_angle_deg(self, v=None):
        """Largest angle between v (default: the barycentre) and the path."""
        v = self.indicator if v is None else np.asarray(v, float)
        return float(np.degrees(core.max_angle(v, self.path)))

    @property
    def midpoint(self):
        """The direction halfway along the path by arc length: always within
        arc/2 of every point, so always covered by the bound."""
        return core.path_midpoint(self.path)

    def step3(self):
        """Step 3 with its check. The barycentre is used when its largest
        angle to the path is at most arc/2; otherwise the arc midpoint,
        which always passes. Returns dict(indicator, choice, passes,
        barycentre_max_deg, midpoint_max_deg, half_arc_deg, bound)."""
        half = self.arc_deg / 2.0
        b, m = self.max_angle_deg(), self.max_angle_deg(self.midpoint)
        ok = b <= half + 1e-9
        v = self.indicator if ok else self.midpoint
        return dict(indicator=v, choice="barycentre" if ok else "midpoint",
                    passes=ok, barycentre_max_deg=b, midpoint_max_deg=m,
                    half_arc_deg=half, bound=float(core.bound_for(v, self.path)))

    def rotation_deg(self):
        """Angle of the path from its starting direction, per grid point
        (the curve of Fig. 4(b))."""
        return np.degrees(core.angle(self.path[0][None, :], self.path))

    def raw_weights(self, sigma):
        """The indicator as weights on the RAW (unwhitened, unoriented)
        channels: a = signs * v / sigma, normalised."""
        a = self.signs * self.indicator / np.asarray(sigma, float)
        return a / np.linalg.norm(a)

    def summary(self):
        return dict(arc_deg=self.arc_deg, end_to_end_deg=self.end_to_end_deg,
                    bound=self.bound(), excess_variance_pct=100 * (self.bound() - 1),
                    margin_pct=100 * (np.sqrt(self.bound()) - 1),
                    indicator=self.indicator.tolist(),
                    max_angle_deg=self.max_angle_deg(),
                    all_positive_frac=float((self.path > 0).all(axis=1).mean()))


def observed_direction(units, window=301, grid=DEFAULT_GRID, polyorder=3,
                       cap_divisor=4, channels=()):
    """Step 1: the fleet's observed direction. units: list of (n, p) arrays,
    each a whole life from start of service to failure."""
    R = np.array([unit_derivative(Y, window, grid, polyorder, cap_divisor)[0]
                  for Y in units])
    sg = orientation(R)
    DH = R * sg[None, None, :]
    DH = DH / np.linalg.norm(DH, axis=2, keepdims=True)
    return FleetDirection(grid=np.asarray(grid), path=fleet_mean_path(DH),
                          unit_directions=DH, signs=sg, channels=tuple(channels))


def bootstrap_rotation(fd, nboot=400, seed=4, q=(5, 95)):
    """Percentile band of the rotation curve over units resampled with
    replacement (the band of Fig. 4(b))."""
    rng = np.random.default_rng(seed)
    DH = fd.unit_directions
    B = []
    for _ in range(nboot):
        M = fleet_mean_path(DH[rng.choice(len(DH), len(DH), True)])
        B.append(np.degrees(core.angle(M[0][None, :], M)))
    return np.percentile(np.array(B), q, axis=0)
