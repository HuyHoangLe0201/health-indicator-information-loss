"""The synthetic system of Section 5 and the supplementary analyses on it.

Three mechanisms with Arrhenius stress factors and one adjustable temperature:

    x_i' = phi_i(u) h_i(x_i),  phi_i(u) = exp(a_i - E_i u),  h_i(x) = 1 + c_i x,

with u a scaled reciprocal temperature (u = 0 is 60 C, u = 3 is 25 C; see
celsius() and activation_energy_eV()), unit channel noise and Var[R] = 10^3,
from x_i(0) = 5e-3 to the condemnation limit x_f = 0.9. Every loss is
evaluated at the system's own signal-to-noise ratio.

Main-text quantities (Section 5):
    greedy(QS)          myopic policy at the steering indicator q*
    best_constant(QS)   best constant input at q*            -> factor 68
    fair_constant()     best constant design, own indicator   -> factor 47
    arc_bound_plant()   the arc bound checked per constant input (5.2)
    policy_table()      Table 2
    misspec_table()     Table 3
Supplementary quantities: levelset_table (Table S2), regime_row (Table S3),
record_identity (S2), one_input_family and family_mistuned (S3),
two_input_floor, segment_counterexample, floor_shape, threshold_sensitivity
and downstream_rul (S4).

The functions share module state (E, XFAIL) exactly as the code that
produced the paper's numbers did; regime_row() and threshold_sensitivity()
change it temporarily and restore it.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

CFG = dict(
    a=np.array([0.0, 0.3, -0.2]),          # log pre-exponentials
    E=np.array([0.6, 1.1, 0.8]),           # activation energies (scaled)
    c=np.array([-0.5, 2.0, 0.8]),          # shape coefficients h_i = 1 + c_i x
    U=(0.0, 3.0),                          # input box
    w=0.9,                                 # fixed weight, generic families only
    xfail=0.9, x0=5e-3, tmax=200.0,
    NU=41,                                 # control grid used throughout
    qstar=np.array([0.466109, 0.770508, 0.434809]),   # steering indicator
    gain=1.0e3,                            # Var[R] / sigma^2
)
A, E, HC = CFG["a"], CFG["E"], CFG["c"]
WS, XFAIL, X0, TMAX = CFG["w"], CFG["xfail"], CFG["x0"], CFG["tmax"]
UMIN, UMAX = CFG["U"]
QS = CFG["qstar"] / np.linalg.norm(CFG["qstar"])
GAIN = CFG["gain"]
BALANCED = np.ones(3) / np.sqrt(3.0)


def ell(b, d2=None):
    """Loss (7) at misalignment b (radians). With d2 = ||d||^2 the weight uses
    the system's own gamma = GAIN d2, as every loss of Section 5 does;
    d2=None uses the fixed weight CFG['w'], which only the generic families
    of Supplementary Section S3 use."""
    if d2 is None:
        w = WS
    else:
        g = CFG["gain"] * np.asarray(d2, dtype=float)
        w = g / (1.0 + g)
    return -0.5 * np.log(np.maximum(1 - w * np.sin(b) ** 2, 1e-300))


def grid(NU):
    """Control grid of NU points on U and the stress factors phi_i(u) on it."""
    US = np.linspace(UMIN, UMAX, NU)
    return US, np.exp(A[None, :] - US[:, None] * E[None, :])


def traj(law, n=2000, rtol=1e-10):
    """Integrate x' = phi(u) h(x) from x0 under the feedback u = law(x) until
    the first mechanism reaches x_f. Returns (states at n equally spaced
    times, lifetime T_f), or (None, None) if x_f is not reached by TMAX."""
    def rhs(t, x):
        return np.exp(A - law(x) * E) * (1.0 + HC * x)

    def ev(t, x):
        return x.max() - XFAIL
    ev.terminal, ev.direction = True, 1
    s = solve_ivp(rhs, [0, TMAX], np.full(3, X0), events=ev, rtol=rtol,
                  atol=rtol * 1e-2, dense_output=True)
    if s.t_events[0].size == 0:
        return None, None
    tg = np.linspace(0, s.t[-1], n)
    return s.sol(tg).T, float(s.t[-1])


def greedy(q, NU=None, fast=False):
    """The myopic policy of Section 5.1 aimed at indicator q: at every state,
    the input on the NU-point control grid with the smallest misalignment.
    Returns (average loss (9), lifetime, trajectory). fast=True loosens the
    tolerances for use inside optimisers; report values computed at full
    accuracy."""
    q = np.abs(q) / np.linalg.norm(q)
    US, PHI = grid(NU or CFG["NU"])
    n_s, rt = (400, 1e-7) if fast else (2000, 1e-10)

    def bb(x):
        D = PHI * (1.0 + HC[None, :] * x[None, :])
        nr = np.linalg.norm(D, axis=1, keepdims=True)
        D = D / nr
        b = np.arccos(np.clip(D @ q, -1, 1))
        j = int(np.argmin(b))
        return US[j], float(b[j]), float(nr[j, 0] ** 2)

    X, Tf = traj(lambda x: bb(x)[0], n_s, rt)
    if X is None:
        return np.inf, np.inf, None
    v = np.array([bb(x)[1:] for x in X])        # columns: beta, ||d||^2
    return float(np.mean(ell(v[:, 0], v[:, 1]))), Tf, X


def best_constant(q, NU=None):
    """Best constant input on the control grid for a GIVEN indicator q:
    (average loss, u)."""
    q = np.abs(q) / np.linalg.norm(q)
    US, _ = grid(NU or CFG["NU"])
    best = (np.inf, None)
    for u in US:
        X, _ = traj(lambda x, u=u: u)
        if X is None:
            continue
        D = np.exp(A - u * E)[None, :] * (1 + HC[None, :] * X)
        nr = np.linalg.norm(D, axis=1, keepdims=True)
        D = D / nr
        L = float(np.mean(ell(np.arccos(np.clip(D @ q, -1, 1)),
                              nr[:, 0] ** 2)))
        if L < best[0]:
            best = (L, float(u))
    return best


def const_curve(u, n=600):
    """Constant input u: (states, informative-direction path, ||d||^2,
    lifetime)."""
    X, Tf = traj(lambda x, u=u: u, n)
    D = np.exp(A - u * E)[None, :] * (1.0 + HC[None, :] * X)
    nr = np.linalg.norm(D, axis=1)
    return X, D / nr[:, None], nr ** 2, Tf


def beta_min_range(q, X):
    """Smallest and largest floor along X, in degrees: the least misalignment
    to q that any input on the grid attains at each state."""
    US, PHI = grid(CFG["NU"])
    out = []
    for x in X:
        D = PHI * (1.0 + HC[None, :] * x[None, :])
        D = D / np.linalg.norm(D, axis=1, keepdims=True)
        out.append(np.degrees(np.arccos(np.clip((D @ q).max(), -1, 1))))
    return float(np.min(out)), float(np.max(out))


def levelset_table():
    """Supplementary Table S2: at five constant inputs scaled to a common
    record Fisher score, rows (u, delta, record score, indicator information
    I_z, rms misalignment in degrees, loss)."""
    rows = [const_curve(u) for u in (0.0, 1.0, 1.5, 2.0, 3.0)]
    F = float(np.mean(rows[2][2]))
    out = []
    for u, (X, DH, d2, Tf) in zip((0.0, 1.0, 1.5, 2.0, 3.0), rows):
        dl = float(np.sqrt(F / np.mean(d2)))
        g = CFG["gain"] * dl ** 2 * d2
        wv = g / (1 + g)
        b = np.arccos(np.clip(DH @ QS, -1, 1))
        L = float(np.mean(-0.5 * np.log(np.maximum(
            1 - wv * np.sin(b) ** 2, 1e-300))))
        Iz = float(np.mean(dl ** 2 * d2 * np.cos(b) ** 2))
        rms = float(np.degrees(np.sqrt(np.mean(b ** 2))))
        out.append((u, dl, F, Iz, rms, L))
    return out


def karcher(DH, mu, iters=500):
    """Weighted intrinsic (Karcher) mean of the directions DH; the same as
    hiloss.core.karcher_mean."""
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


def barycentre_scan():
    """Myopic loss when the indicator is the gamma-weighted barycentre of the
    constant-input path at u0, over 20 values of u0: ((best loss, u0),
    (worst loss, u0))."""
    best, worst = (np.inf, None), (0.0, None)
    for u0 in np.linspace(0.1, 3.0, 20):
        X, DH, d2, Tf = const_curve(u0)
        g = CFG["gain"] * d2
        wv = g / (1 + g)
        q = karcher(DH, wv / wv.sum())
        L, _, _ = greedy(q)
        if L < best[0]:
            best = (L, float(u0))
        if L > worst[0]:
            worst = (L, float(u0))
    return best, worst


def _ea(e, t_hot=60.0, t_cold=25.0, span=3.0):
    """Activation energy in eV of the scaled energy e of Section 5.2, in which
    u = 0 is 60 C and u = 3 is 25 C."""
    k = 8.617333262e-5
    d = 1.0 / (k * (t_cold + 273.15)) - 1.0 / (k * (t_hot + 273.15))
    return float(e * span / d)


def _t_of_u(u, t_hot=60.0, t_cold=25.0, span=3.0):
    """Temperature in degrees Celsius at scaled reciprocal temperature u
    (Section 5.2)."""
    k = 8.617333262e-5
    a = 1.0 / (k * (t_hot + 273.15))
    d = 1.0 / (k * (t_cold + 273.15)) - a
    return float(1.0 / (k * (a + u * d / span)) - 273.15)


def floor_along(X):
    """The floor to q*, in degrees, at every state of a trajectory."""
    US, PHI = grid(CFG["NU"])
    out = []
    for x in X:
        D = PHI * (1.0 + HC[None, :] * x[None, :])
        D = D / np.linalg.norm(D, axis=1, keepdims=True)
        out.append(np.degrees(np.arccos(np.clip((D @ QS).max(), -1, 1))))
    return np.array(out)


def floor_shape():
    """Supplementary Section S4: (tau where the floor to q* peaks and where it
    dips along the myopic trajectory, and 1.0 if the mid-life reachable arc
    drawn by make_fig1.py is the closest of the three it draws)."""
    b = floor_along(greedy(QS)[2])
    tau = np.linspace(0.0, 1.0, len(b))
    bn = floor_along(const_curve(1.2)[0])
    taun = np.linspace(0.0, 1.0, len(bn))
    three = [float(np.interp(f, taun, bn)) for f in (0.05, 0.45, 0.9)]
    return (float(tau[b.argmax()]), float(tau[b.argmin()]),
            float(three[1] < min(three[0], three[2])))


def one_input_family(dmax=8):
    """Supplementary Section S3: the single-input family that holds the
    informative direction fixed, integrated for d = 2..dmax. Returns
    (largest misalignment in degrees, largest loss under the ramp, best and
    worst constant-input loss over d)."""
    U0, XF = 0.4, CFG["xfail"]
    umin, umax = CFG["U"]
    worst_ang, worst_ramp, cbest, cworst = 0.0, 0.0, np.inf, 0.0
    for d in range(2, dmax + 1):
        rng = np.random.default_rng(100 + d)
        a = rng.uniform(-0.5, 0.5, d)
        E = rng.uniform(0.4, 1.4, d)
        x0 = np.full(d, 0.02)
        k = np.exp(a - E * U0)
        tf0 = (XF - x0[0]) / k.max()
        r = (umax - U0) / tf0 * 0.98
        g = r * E / k

        def dh(x, u):
            v = np.exp(a - E * u) * np.exp(g * (x - x0))
            return v / np.linalg.norm(v)

        def run(pol, tmax):
            def rhs(t, x):
                return np.exp(a - E * pol(t, x)) * np.exp(g * (x - x0))

            def hit(t, x):
                return XF - x.max()
            hit.terminal, hit.direction = True, -1
            sol = solve_ivp(rhs, (0, tmax), x0, events=hit,
                            max_step=tmax / 3000, rtol=1e-11, atol=1e-13,
                            dense_output=True)
            if not len(sol.t_events[0]):
                return None
            tf = sol.t_events[0][0]
            ts = np.linspace(0, tf, 500)
            X = sol.sol(ts).T
            D = np.array([dh(x, pol(t, x)) for t, x in zip(ts, X)])
            return ts, D, tf

        out = run(lambda t, x: min(umax, U0 + r * t), 6 * tf0 + 200)
        if out is None:
            continue
        ts, D, tf = out
        b = np.arccos(np.clip(D @ D[0], -1, 1))
        worst_ang = max(worst_ang, float(np.degrees(b).max()))
        worst_ramp = max(worst_ramp, float(np.trapezoid(ell(b), ts) / tf))
        # the sentence compares against the BEST constant input at each d,
        # so the per-d minimum is what enters the range
        best_here = np.inf
        for u in np.linspace(umin, umax, 17):
            o = run(lambda t, x, u=u: u, 200 * tf0 + 8000)
            if o is None:
                continue
            t2, D2, tf2 = o
            bb = np.arccos(np.clip(D2 @ D2[len(D2) // 2], -1, 1))
            best_here = min(best_here, float(np.trapezoid(ell(bb), t2) / tf2))
        if np.isfinite(best_here):
            cbest, cworst = min(cbest, best_here), max(cworst, best_here)
    return worst_ang, worst_ramp, cbest, cworst


def family_mistuned(eps=0.01, d=4, nseed=5):
    """Supplementary Section S3: the same family with its shape coefficients
    perturbed by eps. Returns (median loss under the myopic policy, median
    loss under the best constant input)."""
    U0, XF = 0.4, CFG["xfail"]
    umin, umax = CFG["U"]
    US = np.linspace(umin, umax, 121)
    myo, con = [], []
    for seed in range(nseed):
        rng = np.random.default_rng(10 + seed)
        a = rng.uniform(-0.5, 0.5, d)
        E = rng.uniform(0.4, 1.4, d)
        x0 = np.full(d, 0.02)
        k = np.exp(a - E * U0)
        tf0 = (XF - x0[0]) / k.max()
        r = (umax - U0) / tf0 * 0.98
        g = (r * E / k) * (1.0 + eps *
                           np.random.default_rng(1000 + seed)
                           .choice([-1.0, 1.0], d))

        def dh(x, u):
            v = np.exp(a - E * u) * np.exp(g * (x - x0))
            return v / np.linalg.norm(v)

        def run(pol, tmax):
            def rhs(t, x):
                return np.exp(a - E * pol(t, x)) * np.exp(g * (x - x0))

            def hit(t, x):
                return XF - x.max()
            hit.terminal, hit.direction = True, -1
            sol = solve_ivp(rhs, (0, tmax), x0, events=hit,
                            max_step=tmax / 1500, rtol=1e-10, atol=1e-12,
                            dense_output=True)
            if not len(sol.t_events[0]):
                return None
            tf = sol.t_events[0][0]
            ts = np.linspace(0, tf, 300)
            X = sol.sol(ts).T
            return ts, np.array([dh(x, pol(t, x))
                                 for t, x in zip(ts, X)]), tf

        o = run(lambda t, x: min(umax, U0 + r * t), 10 * tf0 + 400)
        if o is None:
            continue
        ts, D, tf = o
        q = D[len(D) // 2]

        def pol(t, x, q=q):
            c = np.array([dh(x, u) @ q for u in US])
            return US[int(np.argmax(c))]
        o2 = run(pol, 60 * tf0 + 4000)
        if o2 is not None:
            t2, D2, tf2 = o2
            myo.append(float(np.trapezoid(
                ell(np.arccos(np.clip(D2 @ q, -1, 1))), t2) / tf2))
        best = np.inf
        for u in np.linspace(umin, umax, 17):
            o3 = run(lambda t, x, u=u: u, 80 * tf0 + 6000)
            if o3 is None:
                continue
            t3, D3, tf3 = o3
            qq = D3[len(D3) // 2]
            best = min(best, float(np.trapezoid(
                ell(np.arccos(np.clip(D3 @ qq, -1, 1))), t3) / tf3))
        con.append(best)
    return float(np.median(myo)), float(np.median(con))


def downstream_rul(nind=12, nunit=160, K=30, span=0.40, seed=5):
    """Supplementary Section S4: over 12 indicators at u = 1.5, (worst relative
    error of the identity e^{2l} = variance ratio, Spearman correlation
    between an indicator's average loss and the median absolute error of a
    template-matching remaining-life estimator, ratio of the largest to the
    smallest median error)."""
    X, Tf = traj(lambda x: 1.5, 4000)
    if X is None:
        return np.nan, np.nan, np.nan
    tg = np.linspace(0.0, Tf, 4000)

    def at(t):
        t = np.atleast_1d(t)
        return np.stack([np.interp(t, tg, X[:, k]) for k in range(3)], -1)

    def dv(x):
        return np.exp(A - 1.5 * E) * (1.0 + HC * x)

    dm = dv(at(0.5 * Tf)[0])
    dh = dm / np.linalg.norm(dm)
    o = np.ones(3) / np.sqrt(3)
    pp = o - (o @ dh) * dh
    pp /= np.linalg.norm(pp)

    # --- the exact identity, at several misalignments -------------------
    worst = 0.0
    gm = CFG["gain"] * float(dm @ dm)
    for pd in (0.5, 2.0, 5.0, 15.0, 40.0):
        p = np.radians(pd)
        v = np.cos(p) * dh + np.sin(p) * pp
        c2 = float(dm @ v) ** 2 / float(dm @ dm)
        ratio = (1.0 + gm) / (1.0 + gm * c2)
        lo = 0.5 * (np.log1p(gm) - np.log1p(gm * c2))
        worst = max(worst, abs(ratio - np.exp(2 * lo)) / ratio)

    # --- the practitioner's estimator ------------------------------------
    aoff = np.linspace(0.0, span * Tf, K)
    grid_t = np.linspace(0.0, 0.60 * Tf, 900)
    cand = grid_t[grid_t + aoff[-1] <= Tf]
    XC = np.stack([at(t + aoff) for t in cand])          # (nc, K, 3)
    rg = np.random.default_rng(seed)
    taus = rg.uniform(0.0, 0.45 * Tf, nunit)
    XU = np.stack([at(t + aoff) for t in taus])          # (nu, K, 3)

    ts = np.linspace(0.05 * Tf, 0.95 * Tf, 400)
    XS = at(ts)

    out = []
    for k in range(nind):
        p = np.radians(0.0 if k == 0 else 2.0 * k ** 1.5)
        v = np.abs(np.cos(p) * dh + np.sin(p) * pp)
        v /= np.linalg.norm(v)
        li = []
        for x in XS:
            dd = dv(x)
            gg = CFG["gain"] * float(dd @ dd)
            c2 = float(dd @ v) ** 2 / float(dd @ dd)
            li.append(0.5 * (np.log1p(gg) - np.log1p(gg * c2)))
        L = float(np.mean(li))
        r2 = np.random.default_rng(400 + k)
        Z = XU @ v + r2.normal(0.0, 1.0, (nunit, K))
        zc = XC @ v
        j = np.argmin(((zc[None] - Z[:, None]) ** 2).sum(axis=2), axis=1)
        e = np.abs((Tf - cand[j] - aoff[-1]) - (Tf - taus - aoff[-1]))
        out.append((L, float(np.median(e))))
    out.sort()
    Lv = np.array([a for a, _ in out])
    Mv = np.array([b for _, b in out])
    sp = float(np.corrcoef(np.argsort(np.argsort(Lv)),
                           np.argsort(np.argsort(Mv)))[0, 1])
    return worst, sp, float(Mv.max() / Mv.min())


def _own_indicator(u, restarts=4, seed=3):
    """Best fixed indicator for constant input u (Nelder-Mead from the mean
    direction and `restarts` random starts): (average loss, indicator)."""
    X, DH, d2, Tf = const_curve(u, n=2000)

    def L(v):
        v = np.abs(v) / np.linalg.norm(v)
        return float(np.mean(ell(np.arccos(np.clip(DH @ v, -1, 1)), d2)))
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    for s0 in [DH.mean(0)] + [np.abs(rng.normal(size=3)) + .05
                              for _ in range(restarts)]:
        r = minimize(L, s0, method="Nelder-Mead",
                     options=dict(maxiter=3000, xatol=1e-9, fatol=1e-14))
        if r.fun < best[0]:
            best = (float(r.fun), np.abs(r.x) / np.linalg.norm(r.x))
    return best


def arc_bound_plant(us=tuple(np.linspace(0.0, 3.0, 7))):
    """Section 5.2's check of the arc bound: per constant input u, (arc in
    degrees, bound 1/cos^2(arc/2) - 1 in %, excess e^{2l} - 1 in % of the
    best fixed indicator for that input)."""
    out = []
    for u in us:
        X, DH, d2, Tf = const_curve(u, n=2000)
        arc = float(np.sum(np.arccos(np.clip((DH[1:] * DH[:-1]).sum(1),
                                             -1, 1))))
        Lo, _ = _own_indicator(u)
        out.append((float(np.degrees(arc)),
                    100.0 * (1.0 / np.cos(arc / 2.0) ** 2 - 1.0),
                    100.0 * (np.exp(2.0 * Lo) - 1.0)))
    return np.array(out)


def _rul_error(X, Tf, v, nunit=2000, K=30, span=0.40, seed=5, noise=1.0):
    """Median |remaining-life error| / T_f of the template-matching estimator
    of Section 5.2: a time-shifted population of nunit units along X, each
    observed at K instants over the last `span` of a lifetime with channel
    noise `noise`, matched to the reference path. Reads indicator v, or all
    channels if v is None."""
    n = len(X)
    tg = np.linspace(0.0, Tf, n)

    def at(t):
        t = np.atleast_1d(t)
        return np.stack([np.interp(t, tg, X[:, k]) for k in range(3)], -1)
    aoff = np.linspace(0.0, span * Tf, K)
    grid_t = np.linspace(0.0, 0.60 * Tf, 900)
    cand = grid_t[grid_t + aoff[-1] <= Tf]
    XC = np.stack([at(t + aoff) for t in cand])
    rg = np.random.default_rng(seed)
    taus = rg.uniform(0.0, 0.45 * Tf, nunit)
    XU = np.stack([at(t + aoff) for t in taus])
    r2 = np.random.default_rng(seed + 1)
    if v is None:
        Z = XU + r2.normal(0.0, noise, XU.shape)
        dd = ((XC[None] - Z[:, None]) ** 2).sum(axis=(2, 3))
    else:
        v = np.abs(v) / np.linalg.norm(v)
        Z = XU @ v + r2.normal(0.0, noise, (nunit, K))
        dd = (((XC @ v)[None] - Z[:, None]) ** 2).sum(axis=2)
    j = np.argmin(dd, axis=1)
    return float(np.median(np.abs(cand[j] - taus)) / Tf)


def policy_table():
    """Table 2: rows (loss, lifetime, RUL error with the indicator, RUL error
    with all channels) for constant inputs u = 0, 1.5, 3 (60, 41.5 and 25 C)
    each with its own best indicator, then the myopic policy at q* and at
    the balanced indicator."""
    rows = []
    for u in (0.0, 1.5, 3.0):
        Lo, qo = _own_indicator(u)
        X, DH, d2, Tf = const_curve(u, n=2000)
        rows.append((Lo, Tf, _rul_error(X, Tf, qo), _rul_error(X, Tf, None)))
    for q in (QS, np.ones(3) / np.sqrt(3)):
        L, Tf, X = greedy(q)
        rows.append((L, Tf, _rul_error(X, Tf, q), _rul_error(X, Tf, None)))
    return rows


def threshold_sensitivity(xf=1.0):
    """Supplementary Section S4: the headline quantities recomputed at
    condemnation limit xf instead of 0.9: (peak floor in degrees, myopic
    gain over the best constant input, lifetime change in %). XFAIL is
    module state and is restored afterwards."""
    global XFAIL
    keep = XFAIL
    try:
        _, Tf_here = traj(lambda x: 1.5)
        XFAIL = float(xf)
        r = minimize(lambda p: greedy(p, fast=True)[0], QS.copy(),
                     method="Nelder-Mead",
                     options=dict(xatol=1e-5, fatol=1e-9, maxiter=800))
        q1 = np.abs(r.x)
        q1 = q1 / np.linalg.norm(q1)
        L_ad, _, X1 = greedy(q1)
        L_co, _ = best_constant(q1)
        _, peak = beta_min_range(q1, X1)
        _, Tf_there = traj(lambda x: 1.5)
    finally:
        XFAIL = keep
    assert XFAIL == keep, "threshold not restored"
    return peak, L_co / L_ad, 100.0 * (Tf_there - Tf_here) / Tf_here


def record_identity(ntraj=12, seed=5):
    """Supplementary Section S2: the loss over a whole record, -1/2 log(1 - W
    S), against the difference of the two mutual informations computed from
    the Fisher sums. Returns (largest discrepancy, largest relative gap to
    the time-averaged loss)."""
    rng = np.random.default_rng(seed)
    worst_id, biggest_gap = 0.0, 0.0
    for _ in range(ntraj):
        u = rng.uniform(UMIN, UMAX)
        X = _traj_E(E, lambda x, _u=u: _u, n=400)
        if X is None:
            continue
        D = np.array([np.exp(A - u * E) * (1.0 + HC * x) for x in X])
        nrm2 = (D ** 2).sum(axis=1)
        v = np.abs(rng.normal(size=3)) + 1e-3
        v /= np.linalg.norm(v)

        G = CFG["gain"] * nrm2.sum()
        W = G / (1.0 + G)
        cos2 = (D @ v) ** 2 / nrm2
        S = (nrm2 * (1.0 - cos2)).sum() / nrm2.sum()

        direct = 0.5 * (np.log1p(G)
                        - np.log1p(CFG["gain"] * ((D @ v) ** 2).sum()))
        closed = -0.5 * np.log(1.0 - W * S)
        worst_id = max(worst_id, abs(direct - closed))

        g = CFG["gain"] * nrm2
        tavg = (0.5 * (np.log1p(g) - np.log1p(g * cos2))).mean()
        biggest_gap = max(biggest_gap, abs(tavg - direct) / max(direct, 1e-30))
    return worst_id, biggest_gap


G2 = np.array([0.9, 0.3, 1.2])  # second input's sensitivity pattern


def _traj_E(Ev, law, n=800, rtol=1e-9):
    """States of the system with activation energies Ev under the feedback law
    (None if x_f is not reached)."""
    def rhs(t, x):
        return np.exp(A - law(x) * Ev) * (1.0 + HC * x)

    def ev(t, x):
        return x.max() - XFAIL
    ev.terminal, ev.direction = True, 1
    sol = solve_ivp(rhs, [0, TMAX], np.full(3, X0), events=ev,
                    rtol=rtol, atol=rtol * 1e-2, dense_output=True)
    if not sol.t_events[0].size:
        return None
    return sol.sol(np.linspace(0, sol.t[-1], n)).T


def two_input_floor(n=600, ngrid=90, uc=(1.5, 1.5)):
    """Supplementary Section S4: zero loss with two inputs, phi_i = exp(a_i -
    E_i u1 - G2_i u2). The aligning input is solved along the segment x0 + s
    v to which zero loss confines the state, and the system is integrated
    under it. Returns det, points inside U, margin, input ranges, largest
    distance from the segment, worst closed-loop misalignment (degrees) and
    worst grid-search floor (degrees)."""
    def phi(u1, u2):
        return np.exp(A - u1 * E - u2 * G2)

    def dh(x, u1, u2):
        d = phi(u1, u2) * (1.0 + HC * x)
        return d / np.linalg.norm(d)

    def ev(t, x):
        return x.max() - XFAIL
    ev.terminal, ev.direction = True, 1
    sol = solve_ivp(lambda t, x: phi(*uc) * (1.0 + HC * x), [0, TMAX],
                    np.full(3, X0), events=ev, rtol=1e-10, atol=1e-12,
                    dense_output=True)
    Xc = sol.sol(np.linspace(0, sol.t[-1], n)).T
    v = dh(Xc[n // 2], *uc)

    M = np.array([[-(E[0] - E[2]), -(G2[0] - G2[2])],
                  [-(E[1] - E[2]), -(G2[1] - G2[2])]])
    det = float(np.linalg.det(M))

    def align(x):
        lh, lq = np.log(1.0 + HC * x), np.log(v)
        r = np.array([(lq[0] - lq[2]) - (A[0] - A[2]) - (lh[0] - lh[2]),
                      (lq[1] - lq[2]) - (A[1] - A[2]) - (lh[1] - lh[2])])
        return np.linalg.solve(M, r)

    x0 = np.full(3, X0)
    seg = x0[None, :] + np.linspace(0.0, (XFAIL - X0) / v.max(),
                                    n)[:, None] * v[None, :]
    UU = np.array([align(x) for x in seg])
    ok = np.all((UU >= UMIN) & (UU <= UMAX), axis=1)

    sol = solve_ivp(lambda t, x: phi(*align(x)) * (1.0 + HC * x), [0, TMAX],
                    x0, events=ev, rtol=1e-11, atol=1e-13, dense_output=True)
    XL = sol.sol(np.linspace(0, sol.t[-1], n)).T
    off = XL - x0[None, :]
    dist = float(np.linalg.norm(off - (off @ v)[:, None] * v[None, :],
                                axis=1).max())
    res = max(float(np.degrees(np.arccos(np.clip(dh(x, *align(x)) @ v,
                                                 -1, 1)))) for x in XL)

    us = np.linspace(UMIN, UMAX, ngrid)
    worst_floor = 0.0
    for x in seg:
        best = np.inf
        for a1 in us:
            D = np.array([dh(x, a1, a2) for a2 in us])
            best = min(best, float(np.degrees(
                np.arccos(np.clip(D @ v, -1, 1))).min()))
        worst_floor = max(worst_floor, best)
    return dict(det=det, inside=int(ok.sum()), n=n,
                margin=float(np.minimum(UU - UMIN, UMAX - UU).min()),
                u1=(float(UU[:, 0].min()), float(UU[:, 0].max())),
                u2=(float(UU[:, 1].min()), float(UU[:, 1].max())),
                dist=dist, res=res, floor=worst_floor)


def segment_counterexample():
    """Supplementary Section S4: an indicator reachable at every state of one
    trajectory but unattainable on the segment that zero loss requires (d =
    2, one input). Returns (reachable along the trajectory, x2 at its
    failure, offset of the aligning input, x2 where the input leaves U, x1
    where the segment reaches it, whether that is before failure)."""
    th, Ec, umax, kappa, xf = (np.array([0.0, 0.5]), np.array([1.0, 2.0]),
                               1.3, -0.2, 0.9)

    def ev(t, x):
        return x.max() - xf
    ev.terminal, ev.direction = True, 1
    sol = solve_ivp(lambda t, x: np.array([np.exp(th[0] - Ec[0] * umax),
                                           np.exp(th[1] - Ec[1] * umax + x[1])]),
                    [0, 100], np.zeros(2), events=ev, rtol=1e-11, atol=1e-13,
                    dense_output=True)
    X = sol.sol(np.linspace(0, sol.t[-1], 4000)).T
    off = (th[1] - th[0] - kappa) / (Ec[1] - Ec[0])
    u_req = off + X[:, 1] / (Ec[1] - Ec[0])
    reach = float(np.all((u_req >= 0.0) & (u_req <= umax)))
    x2_exit = umax * (Ec[1] - Ec[0]) - (th[1] - th[0]) + kappa
    x1_exit = x2_exit / np.exp(kappa)        # along S_q, x2 = e^kappa x1
    return reach, float(X[-1, 1]), float(off), float(x2_exit), \
        float(x1_exit), float(x1_exit < xf)


def fair_constant(NU=None, restarts=4, seed=3):
    """The fair baseline of Section 5.2: the best constant input on the grid
    together with its OWN best indicator. Returns (loss, u, indicator).
    best_constant(QS) instead holds the indicator at q*, the one designed
    for steering, which inflates the value of steering from 47 to 68."""
    US = np.linspace(UMIN, UMAX, NU or CFG["NU"])
    rng = np.random.default_rng(seed)
    best = (np.inf, None, None)
    for u in US:
        X, _ = traj(lambda x, u=u: u)
        if X is None:
            continue
        D = np.exp(A - u * E)[None, :] * (1.0 + HC[None, :] * X)
        nr = np.linalg.norm(D, axis=1)
        DH, d2 = D / nr[:, None], nr ** 2

        def L(v, DH=DH, d2=d2):
            v = np.abs(v) / np.linalg.norm(v)
            return float(np.mean(ell(np.arccos(np.clip(DH @ v, -1, 1)), d2)))
        starts = [DH.mean(0)] + [np.abs(rng.normal(size=3)) + 0.05
                                 for _ in range(restarts)]
        for s0 in starts:
            r = minimize(L, s0, method="Nelder-Mead",
                         options=dict(maxiter=3000, xatol=1e-9, fatol=1e-14))
            if r.fun < best[0]:
                best = (float(r.fun), float(u),
                        np.abs(r.x) / np.linalg.norm(r.x))
    return best


def _indicator_from(Ev, u0=1.2):
    """Indicator designed from a model with activation energies Ev: the gamma-
    weighted Karcher mean of that model's path under constant input u0
    (Section 5.3)."""
    X = _traj_E(Ev, lambda x: u0)
    D = np.exp(A - u0 * Ev)[None, :] * (1.0 + HC[None, :] * X)
    nr = np.linalg.norm(D, axis=1)
    g = CFG["gain"] * nr ** 2
    w = g / (1.0 + g)
    return karcher(D / nr[:, None], w / w.sum())


def _loss_on_true(q):
    """Average loss on the TRUE system of indicator q under the myopic policy
    aimed at q, at the system's own gamma (Section 5.3)."""
    q = np.abs(q) / np.linalg.norm(q)
    US = np.linspace(UMIN, UMAX, CFG["NU"])
    PHI = np.exp(A[None, :] - US[:, None] * E[None, :])

    def best(x):
        D = PHI * (1.0 + HC[None, :] * x[None, :])
        nr = np.linalg.norm(D, axis=1, keepdims=True)
        b = np.arccos(np.clip((D / nr) @ q, -1, 1))
        j = int(np.argmin(b))
        return US[j], float(b[j]), float(nr[j, 0] ** 2)

    X = _traj_E(E, lambda x: best(x)[0])
    if X is None:
        return np.inf
    v = np.array([best(x)[1:] for x in X])
    return float(np.mean(ell(v[:, 0], v[:, 1])))


def misspec_table(levels=(0.02, 0.05, 0.10, 0.20, 0.35), draws=40, seed=11):
    """Table 3: {omega: (median, 90th percentile, worst)} over `draws`
    perturbed energy vectors E (1 + omega eta), of the loss ratio to the
    indicator designed from the true model. Seeded, so reproducible."""
    rng = np.random.default_rng(seed)
    ref = _loss_on_true(_indicator_from(E))
    out = {}
    for eps in levels:
        ratios = []
        for _ in range(draws):
            Eb = E * (1.0 + eps * rng.standard_normal(3))
            if (Eb <= 0).any():
                continue
            ratios.append(_loss_on_true(_indicator_from(Eb)) / ref)
        a = np.array(ratios)
        out[eps] = (float(np.median(a)), float(np.percentile(a, 90)),
                    float(a.max()))
    return out


REGIME = [np.array([0.60, 1.10, 0.80]),
          np.array([0.80, 0.82, 0.79]),
          np.array([0.80, 0.805, 0.798])]


def regime_row(Ev, restarts=8, seed=7):
    """Supplementary Table S3: for activation energies Ev, (arc of the
    reachable set in degrees, rho = drift / arc, best myopic loss over the
    indicator, its gain over the best constant input at that indicator, and
    over the best constant design with its own indicator). E is module state
    and is restored afterwards."""
    global E
    keep, E = E, Ev
    try:
        US, PHI = grid(CFG["NU"])

        def dirs(x):
            D = PHI * (1.0 + HC[None, :] * x[None, :])
            return D / np.linalg.norm(D, axis=1, keepdims=True)

        Xn, _ = traj(lambda x: 1.5, 400)
        arcs, mids = [], []
        for x in Xn[::20]:
            Dm = dirs(x)
            arcs.append(np.degrees(np.arccos(np.clip(Dm[0] @ Dm[-1], -1, 1))))
            m = Dm.mean(0)
            mids.append(m / np.linalg.norm(m))
        mids = np.array(mids)
        arc = float(np.mean(arcs))
        drift = float(np.degrees(np.arccos(np.clip(mids[0] @ mids[-1], -1, 1))))
        rng = np.random.default_rng(seed)
        best = (np.inf, None)
        for _ in range(restarts):
            r = minimize(lambda v: greedy(v, fast=True)[0],
                         np.abs(rng.normal(0, 1, 3)) + 0.05,
                         method="Nelder-Mead",
                         options=dict(maxiter=90, xatol=1e-4, fatol=1e-9))
            if np.isfinite(r.fun) and r.fun < best[0]:
                best = (float(r.fun), np.abs(r.x) / np.linalg.norm(r.x))
        q_b = best[1]
        L_ad = greedy(q_b)[0]          # winner, full accuracy
        L_co, _ = best_constant(q_b)
        # the second gain column: a constant design with its own indicator
        L_fair = fair_constant()[0]
        return arc, drift / arc, L_ad, L_co / L_ad, L_fair / L_ad
    finally:
        E = keep


# ------------------------------------------------------ descriptive names --
myopic = greedy
trajectory = traj
system_loss = ell
own_indicator = _own_indicator
rul_error = _rul_error
arc_bound_table = arc_bound_plant
celsius = _t_of_u
activation_energy_eV = _ea
trajectory_with_energies = _traj_E
indicator_from_model = _indicator_from
loss_on_true_system = _loss_on_true
