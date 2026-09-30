"""Step 4: a leave-one-out, similarity-based remaining-life estimator.

For a test unit, everything the estimator uses is computed from the OTHER
units: the reference path (median over units of the smoothed channels, each
referred to its own baseline, against normalised life), the channel noise
levels, the orientation, and every indicator. At elapsed cycle n0 the
estimator sees the last `back` fraction of the unit's elapsed cycles,
referred to the unit's own baseline (median of its first `nbase` cycles),
and returns the life L in [n0 + 1, lmax] whose reference path, stretched to
L cycles, is closest in whitened least squares. The remaining life is
L - n0.

The estimator can read all channels or any fixed linear indicator, so the
same machinery compares a fixed indicator with the full record (Table 1).

    model = FleetModel(units)                   # units: list of (n, p) arrays
    sig = model.signals(k)                      # indicators built without k
    L = model.predict_life(k, n0, sig["barycentre"])
    errors = leave_one_out(model, ages=(0.5, 0.7, 0.9))
"""
import numpy as np
from scipy.signal import savgol_filter

from . import core
from .direction import DEFAULT_GRID, unit_window


class FleetModel:
    """Per-unit ingredients of the estimator, computed once.

    units      list of (n_i, p) arrays, each a whole life, row = cycle
    window     Savitzky-Golay window (capped per unit, see unit_window)
    nbase      cycles whose median is a unit's baseline
    nref       points of the reference path on normalised life [0, 1]
    grid       normalised-life grid of the observed direction
    """

    def __init__(self, units, window=301, nbase=50, nref=1001,
                 grid=DEFAULT_GRID, polyorder=3, cap_divisor=4):
        self.nref, self.grid = nref, np.asarray(grid)
        TG = np.linspace(0.0, 1.0, nref)
        self.cells = []
        for Y in units:
            Y = np.asarray(Y, dtype=float)
            n, p = Y.shape
            tau = np.arange(n) / (n - 1.0)
            w = unit_window(n, window, cap_divisor)
            S = np.array([np.std(Y[:, j] - savgol_filter(Y[:, j], w, polyorder)) + 1e-12
                          for j in range(p)])
            D = np.stack([savgol_filter(Y[:, j], w, polyorder, deriv=1) * (n - 1)
                          for j in range(p)], 1) / S[None, :]
            B = np.median(Y[:nbase], axis=0)
            sm = np.stack([savgol_filter(Y[:, j], w, polyorder) for j in range(p)], 1) - B
            self.cells.append(dict(
                n=n, Y=Y, S=S, B=B,
                Dg=np.stack([np.interp(self.grid, tau, D[:, j]) for j in range(p)], 1),
                ref=np.stack([np.interp(TG, tau, sm[:, j]) for j in range(p)], 1)))
        self.DG = np.array([c["Dg"] for c in self.cells])
        self.SG = np.sign(np.median(self.DG.reshape(-1, self.DG.shape[2]), axis=0))
        self.p = self.DG.shape[2]

    @property
    def lives(self):
        return np.array([c["n"] for c in self.cells], dtype=float)

    def path(self, idx):
        """Fleet-mean observed direction of the units idx (whitened,
        oriented), on the grid."""
        R = self.DG[idx] * self.SG[None, None, :]
        R = R / np.linalg.norm(R, axis=2, keepdims=True)
        M = R.mean(0)
        return M / np.linalg.norm(M, axis=1, keepdims=True)

    def signals(self, k, extra=None):
        """Leave-one-out ingredients for test unit k.

        Returns dict(path, sigma, ref, weights) where weights maps a signal
        name to a (p, m) matrix the whitened, oriented record is multiplied
        by: 'all' (identity), 'barycentre' (the fixed indicator of step 3),
        'midpoint' (the arc midpoint, the fallback of step 3), 'fusion' (a
        data-level fusion indicator in the spirit of Liu et al. 2013:
        weights ~ C^-1 dmu at failure), and 'channel:j' for each single
        channel. extra may add named weight vectors (whitened, oriented
        coordinates).
        """
        cells, SG = self.cells, self.SG
        tr = [i for i in range(len(cells)) if i != k]
        M = self.path(tr)
        vb = M.mean(0)
        vb = vb / np.linalg.norm(vb)
        sig = np.median([cells[i]["S"] for i in tr], axis=0)
        refw = np.median([cells[i]["ref"] for i in tr], axis=0) / sig * SG
        Zf = np.array([cells[i]["ref"][-1] / sig * SG for i in tr])
        wf = np.linalg.solve(np.cov(Zf.T), Zf.mean(0))
        wf = wf / np.linalg.norm(wf)
        if wf @ Zf.mean(0) < 0:
            wf = -wf
        W = {"all": np.eye(self.p), "barycentre": vb[:, None], "fusion": wf[:, None],
             "midpoint": core.path_midpoint(M)[:, None]}
        for j in range(self.p):
            e = np.zeros((self.p, 1))
            e[j, 0] = 1.0
            W[f"channel:{j}"] = e
        for name, v in (extra or {}).items():
            v = np.asarray(v, float)
            W[name] = (v / np.linalg.norm(v))[:, None]
        return dict(path=M, sigma=sig, ref=refw, weights=W)

    def predict_life(self, k, n0, sig, name, back=0.30, lmax=4000):
        """Estimated total life of unit k seen up to cycle n0 through signal
        `name` of sig = self.signals(k)."""
        Wm = sig["weights"][name]
        return _fit(self, self.cells[k], n0, Wm, sig["ref"] @ Wm, sig["sigma"],
                    back, lmax)


def _fit(model, c, n0, Wm, refz, sig, back, lmax):
    nref = model.nref
    ns = np.arange(int(np.floor((1 - back) * n0)), n0 + 1)
    z = ((c["Y"][ns] - c["B"]) / sig * model.SG) @ Wm
    Ls = np.arange(n0 + 1, lmax + 1)
    cost = np.empty(len(Ls))
    for i0 in range(0, len(Ls), 500):
        Lb = Ls[i0:i0 + 500]
        pos = np.clip(ns[None, :] / (Lb[:, None] - 1.0) * (nref - 1),
                      0, nref - 1.000001)
        lo = pos.astype(int)
        fr = pos - lo
        r = refz[lo] * (1 - fr[..., None]) + refz[lo + 1] * fr[..., None]
        cost[i0:i0 + 500] = ((z[None] - r) ** 2).sum(axis=(1, 2))
    return float(Ls[int(np.argmin(cost))])


def leave_one_out(model, ages=(0.5, 0.7, 0.9),
                  names=("all", "barycentre", "channel:0", "channel:2", "fusion"),
                  back=0.30, lmax=4000):
    """Relative errors (L - n) / n of every signal at every age, leave-one-out.

    Returns dict(err={name: {age: [..]}}, err_cycles={age: [..]} for 'all',
    fusion_weights=[..] per test unit).
    """
    err = {m: {t: [] for t in ages} for m in names}
    ecyc = {t: [] for t in ages}
    fus = []
    for k, c in enumerate(model.cells):
        sig = model.signals(k)
        fus.append(sig["weights"]["fusion"][:, 0])
        for t in ages:
            n0 = int(round(t * (c["n"] - 1)))
            for m in names:
                L = model.predict_life(k, n0, sig, m, back, lmax)
                err[m][t].append((L - c["n"]) / c["n"])
                if m == "all":
                    ecyc[t].append(L - c["n"])
    return dict(err=err, err_cycles=ecyc, fusion_weights=np.array(fus),
                ages=tuple(ages))


def mse(a):
    return float(np.mean(np.asarray(a) ** 2))


def mse_ratios(res, reference="all"):
    """Mean squared error of each signal relative to the reference signal,
    per age and pooled over ages."""
    err, ages = res["err"], res["ages"]
    per_age = {m: {t: mse(err[m][t]) / mse(err[reference][t]) for t in ages}
               for m in err if m != reference}
    pooled = {m: mse(sum((err[m][t] for t in ages), []))
              / mse(sum((err[reference][t] for t in ages), []))
              for m in err if m != reference}
    return per_age, pooled
