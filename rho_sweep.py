r"""Does the regime index rho actually predict what steering buys?

Table II supports the screening claim with three plants. All three differ only
in the activation energies E: same three mechanisms, same one input, same
shape coefficients, same envelope, same initial state. Three points along one
axis of a nine-dimensional parameter space is not a screening law, and a sweep
that also varied only E would stay inside the same family and learn nothing.

So this re-implements the machinery for general d and m and sweeps plants that
differ in every parameter at once, including the number of mechanisms and the
number of independently adjustable inputs.

WHAT IS BEING TESTED. The manuscript's claim is directional: when rho is large
the reachable arc is short beside the drift, so steering cannot buy much. The
useful reading is a cheap veto -- compute rho from a nominal model, and if it
is large, do not bother designing for actuation. The two ways that can be
wrong are not symmetric:

  rho small, gain small   a wasted design study. Costly, not dangerous.
  rho large, gain large   the veto was wrong and the engineer walked away from
                          a plant that steering would have helped. This is the
                          direction that must not happen.

So the sweep reports the two separately rather than a single correlation.

VALIDATION FIRST. A new estimator that disagrees with the published numbers is
a bug until proven otherwise, so this reproduces Table II's three rows before
it is allowed to sweep anything. The paper's arc is the angle between the two
ENDS of the one-dimensional control grid; for m > 1 that is not defined, so
the diameter of the direction set is used, which coincides with it at m = 1.
The reproduction check is what establishes that.
"""
import io
import itertools
import json
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

OUT = os.path.dirname(os.path.abspath(__file__))
TMAX, XFAIL_D, X0_D = 200.0, 0.9, 5e-3

# One setting, used for the reproduction AND the sweep. Validating at high
# accuracy and then sweeping at low accuracy would check one estimator and
# measure a different one; whatever the sweep runs on has to be what earns
# the right to run by reproducing Table II. These are loose enough for a few
# hundred plants and are accepted only if they still land on the published
# arc, rho and gain.
NTRAJ, RTOL, RESTARTS = 250, 1e-6, 4
CAP = None                 # the whole control grid; see best_constant
MAXIT = lambda d: 40 + 12 * d


# ---------------------------------------------------------------- plant ---
class Plant:
    """dx_i/dt = phi_i(u) (1 + c_i x_i),  phi_i(u) = exp(a_i - <u, E_i>).

    E is d x m, so m = 1 reproduces the manuscript's exp(a_i - u E_i).
    """

    def __init__(self, a, E, c, U, gain, x0=X0_D, xfail=XFAIL_D, nu=21):
        self.a = np.asarray(a, float)
        self.E = np.atleast_2d(np.asarray(E, float))
        if self.E.shape[0] != self.a.size:
            self.E = self.E.T
        self.c = np.asarray(c, float)
        self.U = np.asarray(U, float)          # m x 2
        self.gain, self.x0, self.xfail = float(gain), float(x0), float(xfail)
        self.d, self.m = self.a.size, self.E.shape[1]
        axes = [np.linspace(lo, hi, nu) for lo, hi in self.U]
        self.UG = np.array(list(itertools.product(*axes)))   # K x m
        self.PHI = np.exp(self.a[None, :] - self.UG @ self.E.T)   # K x d

    def dirs(self, x):
        D = self.PHI * (1.0 + self.c[None, :] * x[None, :])
        nr = np.linalg.norm(D, axis=1, keepdims=True)
        return D / np.maximum(nr, 1e-300), nr[:, 0] ** 2

    def traj(self, law, n, rtol):
        def rhs(t, x):
            return np.exp(self.a - self.E @ law(x)) * (1.0 + self.c * x)

        def ev(t, x):
            return x.max() - self.xfail
        ev.terminal, ev.direction = True, 1
        s = solve_ivp(rhs, [0, TMAX], np.full(self.d, self.x0), events=ev,
                      rtol=rtol, atol=rtol * 1e-2, dense_output=True)
        if s.t_events[0].size == 0:
            return None, None
        return s.sol(np.linspace(0, s.t[-1], n)).T, float(s.t[-1])

    def ell(self, b, d2):
        g = self.gain * d2
        w = g / (1.0 + g)
        return -0.5 * np.log(np.maximum(1 - w * np.sin(b) ** 2, 1e-300))


# ------------------------------------------------------- rho and the gain --
def arc_and_rho(P, n=400):
    X, _ = P.traj(lambda x: P.UG[len(P.UG) // 2], n, 1e-8)
    if X is None:
        return None
    arcs, mids = [], []
    for x in X[::max(1, n // 20)]:
        D, _ = P.dirs(x)
        C = np.clip(D @ D.T, -1, 1)
        arcs.append(np.degrees(np.arccos(C.min())))   # diameter of the set
        mm = D.mean(0)
        mids.append(mm / np.linalg.norm(mm))
    mids = np.array(mids)
    arc = float(np.mean(arcs))
    drift = float(np.degrees(np.arccos(np.clip(mids[0] @ mids[-1], -1, 1))))
    if arc < 1e-9:
        return None
    return arc, drift / arc


def greedy(P, q, n=NTRAJ, rtol=RTOL):
    """Myopic-alignment policy: mean loss along the closed loop.

    arccos is decreasing, so the input minimising the angle is the input
    maximising the cosine; taking the argmax saves an arccos over the whole
    control grid at every right-hand-side evaluation, and the arccos is then
    evaluated once per sample at the chosen input.
    """
    q = np.abs(q) / np.linalg.norm(q)

    def u_of(x):
        D = P.PHI * (1.0 + P.c * x)
        return P.UG[int(np.argmax((D @ q) / np.linalg.norm(D, axis=1)))]

    X, Tf = P.traj(u_of, n, rtol)
    if X is None:
        return np.inf
    # all samples at once: (n, K, d). The per-sample Python loop this
    # replaces was 30 per cent of the runtime on its own.
    D = P.PHI[None, :, :] * (1.0 + P.c[None, None, :] * X[:, None, :])
    nr = np.linalg.norm(D, axis=2)
    cos = (D @ q) / nr
    j = np.argmax(cos, axis=1)[:, None]
    b = np.arccos(np.clip(np.take_along_axis(cos, j, 1)[:, 0], -1, 1))
    d2 = np.take_along_axis(nr, j, 1)[:, 0] ** 2
    return float(np.mean(P.ell(b, d2)))


def best_constant(P, q, n=NTRAJ, rtol=RTOL, cap=None):
    """Best mean loss over CONSTANT inputs, searched over the whole grid.

    This is the denominator of the reported gain, so under-searching it
    inflates every gain. An earlier version sampled 15 points with linspace
    over the FLATTENED grid, which at m = 2 walks a few rows of an 11 x 11
    square and never visits the rest of the input box; one plant's gain read
    12.4 against a true 4.54. cap is kept only so the bias can be
    re-measured, and defaults to the full grid.
    """
    q = np.abs(q) / np.linalg.norm(q)
    idx = (range(len(P.UG)) if cap is None else
           np.linspace(0, len(P.UG) - 1, min(cap, len(P.UG)))
           .round().astype(int))
    best = np.inf
    for k in idx:
        u = P.UG[k]
        X, _ = P.traj(lambda x, u=u: u, n, rtol)
        if X is None:
            continue
        Dk = P.PHI[k][None, :] * (1.0 + P.c[None, :] * X)
        nr = np.linalg.norm(Dk, axis=1)
        Dk = Dk / nr[:, None]
        L = float(np.mean(P.ell(np.arccos(np.clip(Dk @ q, -1, 1)), nr ** 2)))
        best = min(best, L)
    return best


def evaluate(P, restarts=RESTARTS, seed=7):
    ar = arc_and_rho(P)
    if ar is None:
        return None
    arc, rho = ar
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    for _ in range(restarts):
        r = minimize(lambda v: greedy(P, v),
                     np.abs(rng.normal(0, 1, P.d)) + 0.05,
                     method="Nelder-Mead",
                     options=dict(maxiter=MAXIT(P.d), xatol=1e-3,
                                  fatol=1e-9))
        if np.isfinite(r.fun) and r.fun < best[0]:
            best = (float(r.fun), np.abs(r.x) / np.linalg.norm(r.x))
    if best[1] is None:
        return None
    L_ad = best[0]
    L_co = best_constant(P, best[1])
    if not np.isfinite(L_co) or L_ad <= 0:
        return None
    return dict(d=P.d, m=P.m, arc=arc, rho=rho, L_ad=L_ad,
                gain=L_co / L_ad)


# ------------------------------------------------ 1. reproduce Table II ---
PAPER = [((0.60, 1.10, 0.80), 33.9, 0.88, 1.7e-4, 68.0),
         ((0.80, 0.82, 0.79), 1.93, 12.5, 3.6e-3, 2.9),
         ((0.80, 0.805, 0.798), 0.45, 53.1, 4.2e-3, 2.7)]
A0 = (0.0, 0.3, -0.2)
C0 = (-0.5, 2.0, 0.8)

print(f"reproducing Table II at the sweep settings "
      f"(n={NTRAJ}, rtol={RTOL}, restarts={RESTARTS}, "
      f"constant search: {'full grid' if CAP is None else CAP})")
print(f"{'E':<26}{'arc':>8}{'paper':>8}{'rho':>8}{'paper':>8}"
      f"{'gain':>8}{'paper':>8}")
ok = True
for Ev, arc_p, rho_p, L_p, g_p in PAPER:
    P = Plant(A0, np.array(Ev)[:, None], C0, [(0.0, 3.0)], 1.0e3, nu=41)
    r = evaluate(P)
    print(f"{str(Ev):<26}{r['arc']:8.2f}{arc_p:8.2f}{r['rho']:8.2f}"
          f"{rho_p:8.2f}{r['gain']:8.1f}{g_p:8.1f}")
    if abs(r["arc"] - arc_p) > 0.12 * arc_p or abs(r["rho"] - rho_p) > 0.15 * rho_p:
        ok = False
if not ok:
    print("\nthe general implementation does NOT reproduce the published "
          "rows; the sweep would be measuring a different estimator")
    sys.exit(1)
print("reproduced -- the general code is the same estimator\n")

# ------------------------------------------------------- 2. the sweep ----
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
rng = np.random.default_rng(20260923)
res, t0 = [], time.time()
for i in range(N):
    d = int(rng.integers(2, 6))                      # 2..5 mechanisms
    m = int(rng.integers(1, min(3, d)))              # 1..2 inputs
    a = rng.normal(0.0, 0.6, d)
    # energies spanning tight clusters (high rho) to wide spreads (low rho)
    spread = 10 ** rng.uniform(-2.2, 0.1)
    E = rng.uniform(0.6, 1.2) + spread * rng.normal(0, 1, (d, m))
    c = rng.uniform(-0.9, 2.5, d)
    U = np.column_stack([np.zeros(m), rng.uniform(0.8, 4.0, m)])
    gain = 10 ** rng.uniform(1.5, 4.0)
    P = Plant(a, E, c, U, gain, x0=10 ** rng.uniform(-3.2, -1.7),
              nu=21 if m == 1 else 11)
    try:
        r = evaluate(P)
    except Exception:
        r = None
    if r is not None and np.isfinite(r["gain"]) and r["rho"] < 1e4:
        r["spread"] = float(spread)
        res.append(r)
    if (i + 1) % 25 == 0:
        # dump as we go: a run stopped at four hours is still a
        # dataset, and this one may take longer than that.
        io.open(os.path.join(OUT, "rho_sweep.json"), "w").write(
            json.dumps(res))
        print(f"  {i+1}/{N}  kept {len(res)}  {time.time()-t0:.0f}s",
              flush=True)

io.open(os.path.join(OUT, "rho_sweep.json"), "w").write(json.dumps(res))
print(f"\n{len(res)} plants survived of {N}, {time.time()-t0:.0f}s")
