"""Section 6: what public run-to-failure fleets establish (Table 4).

    severson_stats(301)             rotation on the Severson cells
    cmapss_stats("train_FD004.txt") between- minus within-regime contrast
    ncmapss_sep(), ncmapss_sweep()  two-way separability test, N-CMAPSS DS02
    ncmapss_sep(units=(1, 2, 3, 4, 5, 6), key="ncmapss01")        ... DS01
    ncmapss_health_constant()       damage constant within every flight?
    xjtu_stats(), pronostia_stats(), nasa_stats()

Data are read from the paths in hiloss.datasets.DATA (the directory given by
$HI_DATA, else ./data; see DATA.md). A function whose data are missing
returns None or raises, and says which file it wanted.
"""
import numpy as np

from .datasets import DATA


def ncmapss_health_constant(key="ncmapss"):
    """Are the health parameters N-CMAPSS imposes constant within every flight?
    Returns (constant in every flight, number of (unit, cycle) flights). If
    so, a within-flight contrast between operating points reads the sensor
    map, not the damage."""
    import h5py
    with h5py.File(DATA[key], "r") as f:
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in np.array(f["A_var"])]
        A = np.array(f["A_dev"], dtype=float)
        T = np.array(f["T_dev"], dtype=float)
    k = A[:, names.index("unit")] * 10000 + A[:, names.index("cycle")]
    order = np.argsort(k, kind="stable")
    k, T = k[order], T[order]
    cuts = np.flatnonzero(np.diff(k)) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(k)]
    const = all(np.all(T[a:b].max(0) == T[a:b].min(0))
                for a, b in zip(starts, ends))
    return bool(const), int(len(starts))


def ncmapss_sep(nclust=6, nstage=2, minpts=400, units=(2, 5, 10, 16, 18, 20),
                key="ncmapss"):
    """The two-way separability test of Section 6 on N-CMAPSS: operating points
    clustered on (altitude, Mach, throttle), life split into nstage stages,
    each sensor regressed on remaining life within every (stage, cluster)
    cell, each cell estimated twice from interleaved halves to measure the
    noise floor. Returns (operating-point effect, stage effect and
    interaction, each over its noise floor; interaction over the operating-
    point main effect; units with a complete layout)."""
    import h5py

    def load(unit):
        with h5py.File(DATA[key], "r") as f:
            A = np.array(f["A_dev"][:, :2])
            m = A[:, 0].astype(int) == unit
            idx = np.where(m)[0]
            lo, hi = idx.min(), idx.max() + 1
            keep = m[lo:hi]
            W = np.array(f["W_dev"][lo:hi])[keep]
            X = np.array(f["X_s_dev"][lo:hi])[keep]
            Y = np.array(f["Y_dev"][lo:hi])[keep].ravel().astype(float)
            cyc = A[lo:hi][keep][:, 1].astype(int)
        return W, X, Y, cyc

    def clusters(W, k, seed=0):
        Z = (W[:, :3] - W[:, :3].mean(0)) / (W[:, :3].std(0) + 1e-12)
        rng = np.random.default_rng(seed)
        C = Z[rng.choice(len(Z), k, replace=False)]
        for _ in range(25):
            lab = np.argmin(((Z[:, None, :] - C[None]) ** 2).sum(2), axis=1)
            for j in range(k):
                if (lab == j).any():
                    C[j] = Z[lab == j].mean(0)
        return lab

    def direction(X, Y):
        if len(Y) < 30 or Y.std() < 1e-9:
            return None
        y = (Y - Y.mean()) / (Y.std() + 1e-12)
        sl, rs = [], []
        for j in range(X.shape[1]):
            x = X[:, j]
            b = float(np.dot(y, x - x.mean()) / len(y))
            sl.append(b)
            rs.append(np.std(x - x.mean() - b * y) + 1e-12)
        v = np.array(sl) / np.array(rs)
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n < 1e-12:
            return None
        z = np.log(np.abs(v / n) + 1e-12)
        return z - z.mean()

    def twoway(M, ns, nc):
        keys = list(M)
        Y = np.array([M[k] for k in keys])
        g = Y.mean(0)
        a = {t: np.mean([M[k] for k in keys if k[0] == t], 0) - g
             for t in range(ns)}
        b = {c: np.mean([M[k] for k in keys if k[1] == c], 0) - g
             for c in range(nc)}
        it = np.array([M[k] - g - a[k[0]] - b[k[1]] for k in keys])
        return np.array([float(np.sum(np.array(list(b.values())) ** 2)),
                         float(np.sum(np.array(list(a.values())) ** 2)),
                         float(np.sum(it ** 2))])

    tm, td, used = np.zeros(3), np.zeros(3), 0
    for u in units:
        W, X, Y, cyc = load(u)
        lab = clusters(W, nclust)
        edges = np.linspace(cyc.min(), cyc.max() + 1, nstage + 1)
        M, N, ok = {}, {}, True
        for t in range(nstage):
            ins = (cyc >= edges[t]) & (cyc < edges[t + 1])
            for c in range(nclust):
                sel = ins & (lab == c)
                if sel.sum() < minpts:
                    ok = False
                    break
                z1 = direction(X[sel][0::2], Y[sel][0::2])
                z2 = direction(X[sel][1::2], Y[sel][1::2])
                if z1 is None or z2 is None:
                    ok = False
                    break
                M[(t, c)] = 0.5 * (z1 + z2)
                N[(t, c)] = 0.5 * (z1 - z2)
            if not ok:
                break
        if not ok or len(M) != nstage * nclust:
            continue
        tm += twoway(M, nstage, nclust)
        td += twoway(N, nstage, nclust)
        used += 1
    r = tm / np.maximum(td, 1e-12)
    return (float(r[0]), float(r[1]), float(r[2]),
            float(tm[2] / tm[0]) if tm[0] > 0 else float("nan"), int(used))


def ncmapss_sweep():
    """The interaction-over-main-effect ratio over the clusterings and stagings
    the paper reports: (smallest and largest interaction over its floor,
    smallest interaction over the main effect)."""
    out = []
    for nc, ns in ((3, 2), (4, 2), (6, 2), (8, 2), (6, 3), (4, 4), (10, 2)):
        try:
            r = ncmapss_sep(nc, ns, minpts=300)
        except Exception:
            continue
        if r[4] > 0:
            out.append((r[2], r[3]))
    return (min(x[0] for x in out), max(x[0] for x in out),
            min(x[1] for x in out))


CM_COLS = (["unit", "cycle"] + [f"op{i}" for i in (1, 2, 3)]
           + [f"s{i}" for i in range(1, 22)])


CM_GRID = np.linspace(0.30, 0.85, 10)


def _cm_load(fname, guard_constant=False):
    """Read one C-MAPSS training file, label operating regimes, keep the
    sensors that vary within a regime and z-score them per regime."""
    import os
    import pandas as pd
    d = pd.read_csv(os.path.join(DATA["cmapss"], fname), sep=r"\s+",
                    header=None, names=CM_COLS)
    d["tau"] = d["cycle"] / d.groupby("unit")["cycle"].transform("max")
    X = d[["op1", "op2", "op3"]].to_numpy(float)
    if guard_constant and not ((X.max(0) - X.min(0)) > 0.1).any():
        # a subset with one operating condition: standardising here would
        # divide by a near-zero std and turn numerical jitter into hundreds
        # of spurious regimes
        d["cond"] = 0
    else:
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
        _, d["cond"] = np.unique(np.round(Xs, 1), axis=0,
                                 return_inverse=True)
    sens = [c for c in CM_COLS if c.startswith("s")]
    sd = d.groupby(["unit", "cond"])[sens].std().median()
    sens = [c for c in sens if sd[c] > 1e-9]          # live within a regime
    g = d.groupby("cond")
    for c in sens:                                     # z-score per regime
        d[c] = (d[c] - g[c].transform("mean")) / g[c].transform("std")
    return d.dropna(subset=sens), sens


def _cm_dhat(tau, Y, w):
    """Whitened unit derivative direction of the sensors Y against normalised
    life tau, on the C-MAPSS grid (quadratic Savitzky-Golay filter, window
    w)."""
    from scipy.signal import savgol_filter
    if len(tau) < w + 3:
        return None
    dt = np.gradient(tau).mean()
    D, S = [], []
    for j in range(Y.shape[1]):
        y = Y[:, j]
        D.append(savgol_filter(y, w, 2, deriv=1) / dt)
        S.append(np.std(y - savgol_filter(y, w, 2)) + 1e-12)
    M = np.stack(D, 1) / np.array(S)[None, :]
    if not np.isfinite(M).all():
        return None
    V = np.stack([np.interp(CM_GRID, tau, M[:, j])
                  for j in range(M.shape[1])], 1)
    nv = np.linalg.norm(V, axis=1, keepdims=True)
    if (nv < 1e-12).any():
        return None
    return V / nv


def _cm_ang(a, b):
    """Mean angle, in degrees, between two direction paths."""
    return float(np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1))).mean())


def cmapss_stats(fname, windows=(7, 9, 11)):
    """The between-regime minus within-regime contrast B - W of Section 6 on
    one C-MAPSS file, each engine serving as its own control, for each
    window in `windows`, with a negative control that cuts fake regimes from
    one real regime. Returns {'real': {w: (mean B - W in degrees, engines
    with B > W, engines)}, 'ctrl': {w: (|B - W|, p-value)}, 'nregime':
    regimes}."""
    import os
    from scipy import stats as _st
    if not os.path.isdir(DATA["cmapss"]):
        return None
    d, sens = _cm_load(fname)
    G = {k: g.sort_values("cycle") for k, g in d.groupby(["unit", "cond"])}
    units = sorted({u for u, _ in G})
    out = dict(real={}, ctrl={}, nregime=int(d["cond"].nunique()))
    for w in windows:
        real, ctrl = [], []
        for u in units:
            cs = [c for (uu, c) in G if uu == u]

            # --- real test, every regime resampled to a common count -----
            if len(cs) >= 2:
                m = min(len(G[(u, c)]) for c in cs)
                if m >= 2 * (w + 3):
                    H = {}
                    for c in cs:
                        g = G[(u, c)]
                        idx = np.unique(np.linspace(0, len(g) - 1, m)
                                        .round().astype(int))
                        g = g.iloc[idx]
                        tau = g["tau"].to_numpy(float)
                        Y = g[sens].to_numpy(float)
                        a = _cm_dhat(tau[0::2], Y[0::2], w)
                        b = _cm_dhat(tau[1::2], Y[1::2], w)
                        if a is not None and b is not None:
                            H[c] = (a, b)
                    ks = sorted(H)
                    if len(ks) >= 2:
                        ww = [_cm_ang(*H[c]) for c in ks]
                        bb = [_cm_ang(H[c][0], H[c2][1])
                              for i, c in enumerate(ks) for c2 in ks[i + 1:]]
                        real.append((np.mean(ww), np.mean(bb)))

            # --- negative control: fake regimes cut from one real regime --
            wf, bf = [], []
            for c in cs:
                g = G[(u, c)]
                if len(g) < 4 * (w + 3):
                    continue
                tau = g["tau"].to_numpy(float)
                Y = g[sens].to_numpy(float)
                q = [_cm_dhat(tau[i::4], Y[i::4], w) for i in range(4)]
                if any(x is None for x in q):
                    continue
                wf.append(_cm_ang(q[0], q[1]))       # fake A against itself
                bf.append(_cm_ang(q[0], q[3]))       # fake A against fake B
            if wf:
                ctrl.append((np.mean(wf), np.mean(bf)))

        R = np.array(real)
        C = np.array(ctrl)
        dr = R[:, 1] - R[:, 0]
        dc = C[:, 1] - C[:, 0]
        out["real"][w] = (float(dr.mean()), int((dr > 0).sum()), len(dr))
        out["ctrl"][w] = (float(abs(dc.mean())),
                          float(_st.ttest_rel(C[:, 1], C[:, 0]).pvalue))
    return out


def _dirs_from(mat, win, grid_):
    """Whitened derivative direction on a normalised-life grid from an (n, p)
    feature record (quadratic filter), or None for short or constant
    records."""
    from scipy.signal import savgol_filter
    a = np.asarray(mat, dtype=float)
    n, pdim = a.shape
    if n < 40 or not np.isfinite(a).all():
        return None
    w = min(win, (n // 3) * 2 + 1)
    if w < 7:
        return None
    tau = np.arange(n) / (n - 1.0)
    D, S = [], []
    for j in range(pdim):
        y = a[:, j]
        if np.std(y) < 1e-12:
            return None
        D.append(savgol_filter(y, w, 2, deriv=1) * (n - 1))
        S.append(np.std(y - savgol_filter(y, w, 2)) + 1e-12)
    D = np.stack(D, 1) / np.array(S)[None, :]
    return np.stack([np.interp(grid_, tau, D[:, k]) for k in range(pdim)], 1)


def _orient(R):
    """Orient channels by the sign of the fleet median and normalise."""
    sg = np.sign(np.median(R.reshape(-1, R.shape[2]), axis=0))
    sg[sg == 0] = 1.0
    D = R * sg[None, None, :]
    return D / np.linalg.norm(D, axis=2, keepdims=True)


def _travel(DH):
    """Arc length, in degrees, of the fleet-mean direction path."""
    M = DH.mean(0)
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    return float(np.degrees(np.arccos(np.clip((M[:-1] * M[1:]).sum(1),
                                              -1, 1))).sum())


def _perm_p(DH, lab, nperm=3000, seed=3):
    """Permutation p-value of the between-condition spread of the fleet-mean
    directions: (p, observed spread in degrees)."""
    rng = np.random.default_rng(seed)
    conds = sorted(set(lab))

    def stat(labels):
        M = []
        for c in conds:
            m = DH[labels == c].mean(0)
            M.append(m / np.linalg.norm(m, axis=1, keepdims=True))
        s = [np.degrees(np.arccos(np.clip((M[i] * M[j]).sum(1), -1, 1))).mean()
             for i in range(len(conds)) for j in range(i + 1, len(conds))]
        return float(np.mean(s))
    obs = stat(lab)
    null = np.array([stat(rng.permutation(lab)) for _ in range(nperm)])
    return float((null >= obs).mean()), obs


def xjtu_stats(win=21):
    """XJTU-SY bearings: rotation and a permutation test across operating
    conditions (condition is fixed per bearing, so condition and group of
    units cannot be separated)."""
    import os
    if not os.path.exists(DATA["xjtu"]):
        return None
    Z = np.load(DATA["xjtu"], allow_pickle=True)
    names = sorted({k.split("__")[0] for k in Z.keys()
                    if k.startswith("Bearing")})
    G = np.linspace(0.20, 0.90, 30)
    D, lab = [], []
    for b in names:
        d = _dirs_from(Z[b + "__X"], win, G)
        if d is not None and np.isfinite(d).all():
            D.append(d)
            lab.append(str(Z[b + "__cond"]))
    DH = _orient(np.array(D))
    lab = np.array(lab)
    p, _ = _perm_p(DH, lab)
    sizes = sorted({int((lab == c).sum()) for c in set(lab)})
    return dict(p=p, travel=_travel(DH), n=len(DH),
                ncond=len(set(lab)), sizes=sizes)


def pronostia_stats(win=201):
    """PRONOSTIA bearings: rotation and a permutation test across operating
    conditions (fixed per bearing)."""
    import os
    if not os.path.exists(DATA["pronostia"]):
        return None
    Z = np.load(DATA["pronostia"], allow_pickle=True)
    G = np.linspace(0.20, 0.90, 30)
    D, lab = [], []
    for k, nm in enumerate(Z["names"]):
        d = _dirs_from(np.asarray(Z["data"][k], dtype=float), win, G)
        if d is not None and np.isfinite(d).all():
            D.append(d)
            lab.append("c" + str(nm)[7])
    DH = _orient(np.array(D))
    p, _ = _perm_p(DH, np.array(lab))
    return dict(p=p, travel=_travel(DH), n=len(DH))


def nasa_stats(win=21):
    """NASA batteries: per-cell and fleet rotation of the observed direction
    (only four cells)."""
    import os
    if not os.path.exists(DATA["nasa"]):
        return None
    import pandas as pd
    df = pd.read_csv(DATA["nasa"])
    cols = ["t_CV", "tau", "area_CV"]
    G = np.linspace(0.15, 0.90, 30)
    D = []
    for _, g in df.groupby("battery"):
        d = _dirs_from(g.sort_values("cycle_seq")[cols].to_numpy(float),
                       win, G)
        if d is not None and np.isfinite(d).all():
            D.append(d)
    DH = _orient(np.array(D))
    per = [float(np.degrees(np.arccos(np.clip((DH[i, :-1] * DH[i, 1:]).sum(1),
                                              -1, 1))).sum())
           for i in range(len(DH))]
    M = DH.mean(0)
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    return dict(n=len(DH), per_lo=min(per), per_hi=max(per),
                allpos=float((M > 0).all(axis=1).mean()),
                travel=_travel(DH))


def severson_stats(win):
    """Severson cells, Table 4: fleet-mean observed direction over tau in
    [0.15, 0.90] with smoothing window `win`: cells, arc in degrees, angle
    between first and last direction, fraction of the path with all weights
    positive, information shares at mid-path. Same computation as
    hiloss.direction.observed_direction."""
    import os
    if not os.path.exists(DATA["severson"]):
        return None
    from scipy.signal import savgol_filter
    Z = np.load(DATA["severson"], allow_pickle=True)
    CH = [str(c) for c in Z["channels"]]
    USE = [CH.index("QDischarge"), CH.index("IR"), CH.index("Tavg")]
    G = np.linspace(0.15, 0.90, 40)
    rows = []
    for k in range(len(Z["data"])):
        if not np.isfinite(Z["lives"][k]):
            continue
        aa = np.asarray(Z["data"][k], dtype=float)
        n = len(aa)
        if n < 400 or not np.isfinite(aa[:, USE]).all():
            continue
        tau = np.arange(n) / (n - 1.0)
        w = min(win, (n // 4) * 2 + 1)
        Dl, Sl = [], []
        for j in USE:
            y = aa[:, j]
            Dl.append(savgol_filter(y, w, 3, deriv=1) * (n - 1))
            Sl.append(np.std(y - savgol_filter(y, w, 3)) + 1e-12)
        Dm = np.stack(Dl, 1) / np.array(Sl)[None, :]
        rows.append(np.stack([np.interp(G, tau, Dm[:, c])
                              for c in range(3)], 1))
    R = np.array(rows)
    sg = np.sign(np.median(R.reshape(-1, 3), axis=0))
    DH = R * sg[None, None, :]
    DH = DH / np.linalg.norm(DH, axis=2, keepdims=True)
    M = DH.mean(0)
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    travel = float(np.degrees(np.arccos(np.clip((M[:-1] * M[1:]).sum(1),
                                                -1, 1))).sum())
    e2e = float(np.degrees(np.arccos(np.clip(M[0] @ M[-1], -1, 1))))
    P = M ** 2
    mid = P[len(P) // 2]
    return dict(n=len(R), travel=travel, e2e=e2e,
                allpos=float((M > 0).all(axis=1).mean()),
                shares=mid)
