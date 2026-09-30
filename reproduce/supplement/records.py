"""Summaries of the recorded supplementary computations (Supplementary
Section S4). The records sit beside this file; rho_sweep.py,
downstream_sweep.py and dinkelbach_grids.py regenerate them (hours each).

    python reproduce/supplement/records.py
"""
import io
import os

import numpy as np


def rho_sweep_stats():
    """Summary of rho_sweep.json, the screening sweep over 300 random systems
    (Supplementary Section S4): how well the reachable-arc length, rho and
    their combination screen for large steering gains."""
    # os is imported here rather than at the top because the source copy of
    # this script does not import it; the repository copy does, added by the
    # $HI_DATA patch. Importing locally keeps the two copies differing by
    # that patch alone, which is what the packaging audit checks.
    import json
    import os
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "rho_sweep.json")
    if not os.path.exists(fn):
        return None
    rows = json.load(io.open(fn, encoding="utf-8"))
    rho = np.array([r["rho"] for r in rows])
    arc = np.array([r["arc"] for r in rows])
    gain = np.array([r["gain"] for r in rows])

    def sp(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean()
        rb -= rb.mean()
        return float((ra * rb).sum()
                     / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))

    big = gain >= 5.0
    low = rho < 1.0
    veto = rho >= 1.0
    near = np.abs(np.log(rho) - np.log(0.88)) < 0.35
    return dict(
        n=float(len(rows)),
        sp=sp(rho, gain),
        sp_arc=sp(arc, gain),
        # rho = drift / arc by construction, so the drift is recovered
        # rather than stored.
        sp_drift=sp(rho * arc, gain),
        **_rho_classifier(rho, arc, gain),
        **_rho_nested(rho, arc, gain,
                      np.array([r["m"] for r in rows])),
        share_low=100.0 * big[low].mean(),
        base=100.0 * big.mean(),
        miss=100.0 * big[veto].mean(),
        med_near=float(np.median(gain[near])),
    )


def _rho_classifier(rho, arc, gain, B=2000, seed=31):
    """Bootstrap AUC of rho, the arc and their logistic combination as screens
    for a gain above the median."""
    y = gain >= 5.0
    drift = rho * arc

    def auc(sc, yy):
        pos, neg = sc[yy], sc[~yy]
        gt = (pos[:, None] > neg[None, :]).sum()
        eq = (pos[:, None] == neg[None, :]).sum()
        return float((gt + 0.5 * eq) / (len(pos) * len(neg)))

    rng = np.random.default_rng(seed)
    out = {}
    for name, sc in (("auc_rho", -np.log(rho)), ("auc_arc", arc),
                     ("auc_drift", -drift)):
        out[name] = auc(sc, y)
        bs = []
        for _ in range(B):
            i = rng.integers(0, len(y), len(y))
            if y[i].all() or (~y[i]).all():
                continue
            bs.append(auc(sc[i], y[i]))
        out[name + "_lo"] = float(np.percentile(bs, 2.5))
        out[name + "_hi"] = float(np.percentile(bs, 97.5))
    out["p_below"] = 100.0 * float(y[rho < 2.6].mean())
    out["p_above"] = 100.0 * float(y[rho >= 2.6].mean())
    d1 = rho <= np.percentile(rho, 10)
    out["p_lowest_decile"] = 100.0 * float(y[d1].mean())
    out["drift_lowest_decile"] = float(np.median(drift[d1]))
    out["drift_all"] = float(np.median(drift))
    rs = np.random.default_rng(seed + 1)
    ranks = []
    for _ in range(B):
        i = rs.integers(0, len(y), len(y))
        a, g = rho[i], gain[i]
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(g)).astype(float)
        ra -= ra.mean()
        rb -= rb.mean()
        ranks.append(float((ra * rb).sum()
                           / np.sqrt((ra ** 2).sum() * (rb ** 2).sum())))
    out["sp_lo"] = float(np.percentile(ranks, 2.5))
    out["sp_hi"] = float(np.percentile(ranks, 97.5))
    return out


def _rho_nested(rho, arc, gain, m):
    """Nested logistic models of a large gain on log arc, log drift and their
    interaction (likelihood-ratio tests)."""
    from scipy import stats as _st
    from scipy.optimize import minimize as _min
    y = (gain >= 5.0).astype(float)
    x1, x2 = np.log(arc), np.log(rho * arc)

    def fit(X, yy):
        X = np.column_stack([np.ones(len(yy)), X])
        r = _min(lambda b: np.sum(np.logaddexp(0, X @ b) - yy * (X @ b)),
                 np.zeros(X.shape[1]), method="BFGS")
        return r.x, -r.fun

    def lls(sel):
        a, b, i = x1[sel], x2[sel], y[sel]
        return dict(arc=fit(np.column_stack([a]), i)[1],
                    rho=fit(np.column_stack([b - a]), i)[1],
                    full=fit(np.column_stack([a, b]), i)[1],
                    inter=fit(np.column_stack([a, b, a * b]), i)[1])

    L = lls(np.ones(len(y), bool))
    out = dict(
        p_drift_main=float(_st.chi2.sf(2 * (L["full"] - L["arc"]), 1)),
        p_ratio=float(_st.chi2.sf(2 * (L["full"] - L["rho"]), 1)),
        p_interaction=float(_st.chi2.sf(2 * (L["inter"] - L["full"]), 1)),
        aic_arc=4.0 - 2 * L["arc"], aic_rho=4.0 - 2 * L["rho"])
    ok = True
    for mm in (1, 2):
        Lm = lls(m == mm)
        ok &= (_st.chi2.sf(2 * (Lm["full"] - Lm["arc"]), 1) > 0.05
               and _st.chi2.sf(2 * (Lm["full"] - Lm["rho"]), 1) < 0.05)
    out["strata_agree"] = float(ok)

    qa = np.quantile(x1, [1 / 3, 2 / 3])
    qd = np.quantile(x2, [1 / 3, 2 / 3])
    ia, idd = np.digitize(x1, qa), np.digitize(x2, qd)
    for tag, a_, d_ in (("lo_lo", 0, 0), ("lo_hi", 0, 2),
                        ("hi_lo", 2, 0), ("hi_hi", 2, 2)):
        out["t_" + tag] = float(y[(ia == a_) & (idd == d_)].mean())

    def auc(sc, yy):
        pos, neg = sc[yy == 1], sc[yy == 0]
        return float(((pos[:, None] > neg[None, :]).sum()
                      + 0.5 * (pos[:, None] == neg[None, :]).sum())
                     / (len(pos) * len(neg)))

    MOD = {"arc": np.column_stack([x1]), "rho": np.column_stack([x2 - x1]),
           "inter": np.column_stack([x1, x2, x1 * x2])}
    rng = np.random.default_rng(41)
    cv = {k: [] for k in MOD}
    n = len(y)
    for _ in range(20):
        idx = rng.permutation(n)
        folds = np.array_split(idx, 10)
        for k, X in MOD.items():
            sc = np.empty(n)
            for te in folds:
                tr = np.setdiff1d(idx, te)
                b, _l = fit(X[tr], y[tr])
                sc[te] = b[0] + X[te] @ b[1:]
            cv[k].append(auc(sc, y))
    for k in MOD:
        out["cv_" + k] = float(np.mean(cv[k]))
    out["cv_inter_always_better"] = float(
        all(a > b for a, b in zip(cv["inter"], cv["arc"])))
    return out


def downstream_sweep_stats():
    """Summary of downstream_sweep.json: the downstream estimator test of
    Supplementary Section S4 repeated on 120 random systems."""
    import json
    import os
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "downstream_sweep.json")
    if not os.path.exists(fn):
        return None
    J = json.load(io.open(fn, encoding="utf-8"))
    ok = [r for r in J["plants"] if r["ok"]]
    sp = np.array([r["sp"] for r in ok])
    rl = np.array([r["random"]["sp_loss"] for r in ok])
    la = np.array([r["random"]["sp_loss_angle"] for r in ok])
    return dict(
        paper=J["paper"], n=float(len(ok)),
        sp_min=float(sp.min()), sp_max=float(sp.max()),
        rand_median=float(np.median(rl)),
        rand_excl=100.0 * float(np.mean([r["random"]["lo"] > 0 for r in ok])),
        identical=100.0 * float(np.mean(la > 0.9999)))


def dinkelbach_record(k=4):
    """{NU: dict(lam, p, lam_prev, N, v)} from dinkelbach_grids_NU*.json: the
    computed fractional optimum at q* on state grids of 25 to 137 points per
    axis for control grids NU = 41, 81, 161, and its limit lam from a fit v
    = lam + C h^p over the k finest grids (lam_prev: one grid coarser). None
    if a record is missing."""
    import json as _json
    import os as _os
    from scipy.optimize import curve_fit
    here = _os.path.dirname(_os.path.abspath(__file__))
    out = {}
    for NU in (41, 81, 161):
        p = _os.path.join(here, f"dinkelbach_grids_NU{NU}.json")
        if not _os.path.exists(p):
            return None
        rec = _json.load(io.open(p, encoding="utf-8"))
        N = np.array(sorted(int(n) for n in rec))
        if len(N) < k + 1:
            return None
        v = np.array([rec[str(n)] for n in N])
        h = 1.0 / (N - 1)

        def fit(sl):
            po, _ = curve_fit(lambda hh, lam, c, pp: lam + c * hh ** pp,
                              h[sl], v[sl], p0=(v[-1] * 0.7, 5e-3, 1.0),
                              maxfev=20000)
            return float(po[0]), float(po[2])
        # the finest k grids give the reported value; the k grids one step
        # coarser say how much it moves with the choice of window, which is
        # what the text reports as its spread
        lam, pw = fit(slice(-k, None))
        lam_prev, _ = fit(slice(-k - 1, -1))
        out[NU] = dict(lam=lam, p=pw, lam_prev=lam_prev, N=N, v=v)
    return out


if __name__ == "__main__":
    r = rho_sweep_stats()
    print("rho sweep:", r)
    print("downstream sweep:", downstream_sweep_stats())
    dk = dinkelbach_record()
    if dk:
        for NU, d in dk.items():
            print(f"fractional optimum, control grid {NU}: limit {d['lam']:.4g}"
                  f" (one grid coarser {d['lam_prev']:.4g}), order {d['p']:.2f},"
                  f" finest value {d['v'][-1]:.4g} at N = {d['N'][-1]}")
