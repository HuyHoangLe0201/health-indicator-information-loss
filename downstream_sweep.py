r"""Section 6.8's downstream test, repeated across randomly drawn plants.

Section 6.8 ranks twelve indicators by their loss and by the median error of
a remaining-life estimator that knows nothing of the construction -- it
matches an observed scalar history against the indicator's own reference
curve -- and finds the two orderings agree at Spearman 0.89. That is one
plant. A reviewer asked whether it survives on others, and the question is
fair: twelve indicators on one plant say nothing about whether the ordering
is a property of the loss or of that plant.

This repeats the identical protocol on plants drawn at random: two to five
mechanisms, and the pre-exponentials, activation energies, shape coefficients,
operating point, initial state and signal-to-noise ratio all varied. For each
it records the rank correlation, a bootstrap interval that resamples UNITS
(the uncertainty is in each indicator's median error over 160 units, not in
the twelve losses, which are exact), and the spread of the loss across the
twelve indicators, since a plant on which the indicators barely differ in
loss cannot say anything about ordering.

VALIDATION FIRST. On the paper's own plant this must reproduce the gate's
Spearman and error ratio to machine precision, or it is a different estimator
and the sweep would be measuring something else. Every plant's parameters are
stored beside its result, so any row can be re-evaluated from the file.

Run:  python downstream_sweep.py [n_plants]      writes downstream_sweep.json
"""
import io
import json
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
TMAX, XFAIL = 200.0, 0.9


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def trajectory(phi, c, x0, n=4000, rtol=1e-10):
    """Constant-input degradation, dx_i/dt = phi_i (1 + c_i x_i), to failure."""
    d = phi.size

    def rhs(t, x):
        return phi * (1.0 + c * x)

    def ev(t, x):
        return x.max() - XFAIL
    ev.terminal, ev.direction = True, 1
    s = solve_ivp(rhs, [0, TMAX], np.full(d, x0), events=ev, rtol=rtol,
                  atol=rtol * 1e-2, dense_output=True)
    if s.t_events[0].size == 0:
        return None, None
    return s.sol(np.linspace(0, s.t[-1], n)).T, float(s.t[-1])


def downstream(a, E, c, u0, x0, gain, nind=12, nunit=160, K=30, span=0.40,
               seed=5, tilts="paper"):
    """The protocol of verify_numbers.downstream_rul(), for any plant.

    Returns (losses (nind,), per-unit absolute errors (nind, nunit)), or
    None if the plant does not fail inside the horizon or its informative
    direction is parallel to equal weights (no second axis to tilt toward).
    """
    phi = np.exp(np.asarray(a) - np.asarray(E) * u0)
    c = np.asarray(c, float)
    d = phi.size
    X, Tf = trajectory(phi, c, x0)
    if X is None:
        return None
    tg = np.linspace(0.0, Tf, 4000)

    def at(t):
        t = np.atleast_1d(t)
        return np.stack([np.interp(t, tg, X[:, k]) for k in range(d)], -1)

    def dv(x):
        return phi * (1.0 + c * x)

    dm = dv(at(0.5 * Tf)[0])
    dh = dm / np.linalg.norm(dm)
    o = np.ones(d) / np.sqrt(d)
    pp = o - (o @ dh) * dh
    if np.linalg.norm(pp) < 1e-6:
        return None
    pp /= np.linalg.norm(pp)

    aoff = np.linspace(0.0, span * Tf, K)
    grid_t = np.linspace(0.0, 0.60 * Tf, 900)
    cand = grid_t[grid_t + aoff[-1] <= Tf]
    XC = np.stack([at(t + aoff) for t in cand])
    rg = np.random.default_rng(seed)
    taus = rg.uniform(0.0, 0.45 * Tf, nunit)
    XU = np.stack([at(t + aoff) for t in taus])
    XS = at(np.linspace(0.05 * Tf, 0.95 * Tf, 400))

    # "paper": the Section 6.8 design, tilting ever further toward equal
    #   weights. Loss and error then both grow with ONE parameter, so their
    #   agreement is close to structural -- over 120 plants the rank
    #   correlation never fell below 0.87.
    # "random": each indicator tilts by a random amount in a random
    #   direction orthogonal to the mid-life informative direction. Two
    #   indicators at the same mid-life angle then differ in loss according
    #   to whether they lean toward where the direction is heading or away
    #   from it, which the loss sees (it integrates over the whole life) and
    #   a snapshot does not. The mid-life angle is returned as the naive
    #   competitor the loss has to beat.
    trng = np.random.default_rng(900 + seed)
    Ls, Es, Bs = [], [], []
    for k in range(nind):
        if tilts == "paper":
            p = np.radians(0.0 if k == 0 else 2.0 * k ** 1.5)
            ax = pp
        else:
            p = np.radians(trng.uniform(0.0, 60.0))
            g = trng.normal(0.0, 1.0, d)
            g -= (g @ dh) * dh
            ax = g / np.linalg.norm(g)
        v = np.abs(np.cos(p) * dh + np.sin(p) * ax)
        v /= np.linalg.norm(v)
        Bs.append(float(np.degrees(np.arccos(np.clip(v @ dh, -1, 1)))))
        D = np.array([dv(x) for x in XS])
        gg = gain * (D ** 2).sum(1)
        c2 = (D @ v) ** 2 / (D ** 2).sum(1)
        Ls.append(float(np.mean(0.5 * (np.log1p(gg) - np.log1p(gg * c2)))))
        r2 = np.random.default_rng(400 + k)
        Z = XU @ v + r2.normal(0.0, 1.0, (nunit, K))
        zc = XC @ v
        j = np.argmin(((zc[None] - Z[:, None]) ** 2).sum(axis=2), axis=1)
        Es.append(np.abs((Tf - cand[j] - aoff[-1]) - (Tf - taus - aoff[-1])))
    if tilts == "paper":
        return np.array(Ls), np.array(Es)
    return np.array(Ls), np.array(Es), np.array(Bs)


def summarise(L, E, rng, B=1000):
    med = np.median(E, axis=1)
    sp = spearman(L, med)
    n = E.shape[1]
    bs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        bs[b] = spearman(L, np.median(E[:, idx], axis=1))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(sp=sp, lo=float(lo), hi=float(hi),
                ratio=float(med.max() / med.min()),
                loss_min=float(L.min()), loss_max=float(L.max()))


if __name__ == "__main__":
    # ---- 1. the paper's plant, which must reproduce the gate exactly ----
    PAPER = dict(a=[0.0, 0.3, -0.2], E=[0.6, 1.1, 0.8], c=[-0.5, 2.0, 0.8],
                 u0=1.5, x0=5e-3, gain=1.0e3)
    r = downstream(**PAPER)
    L, E = r
    med = np.median(E, axis=1)
    sp0, ra0 = spearman(L, med), med.max() / med.min()
    print(f"paper's plant: Spearman {sp0:.6f}, error ratio {ra0:.6f}")
    print("  gate values : Spearman 0.888112, error ratio 1.519")
    if abs(sp0 - 0.888112) > 5e-6 or abs(ra0 - 1.519) > 5e-4:
        print("does NOT reproduce Section 6.8 -- refusing to sweep")
        sys.exit(1)
    print("  reproduced: this is the same estimator")

    # uncertainty on the paper's own plant. Units are resampled, not the
    # twelve indicators: the losses are exact and the indicators are a
    # designed sweep, so the randomness is in each median over 160 units.
    brng = np.random.default_rng(11)
    B, n_u = 5000, E.shape[1]
    bs_sp, bs_ra = np.empty(B), np.empty(B)
    for b in range(B):
        idx = brng.integers(0, n_u, n_u)
        m = np.median(E[:, idx], axis=1)
        bs_sp[b], bs_ra[b] = spearman(L, m), m.max() / m.min()
    prng = np.random.default_rng(12)
    P = 200000
    null = np.array([spearman(L, prng.permutation(med)) for _ in range(P)])
    PAPER_STATS = dict(
        sp=sp0, ratio=float(ra0),
        sp_lo=float(np.percentile(bs_sp, 2.5)),
        sp_hi=float(np.percentile(bs_sp, 97.5)),
        ratio_lo=float(np.percentile(bs_ra, 2.5)),
        ratio_hi=float(np.percentile(bs_ra, 97.5)),
        perm_p=float((1 + np.sum(null >= sp0)) / (P + 1)))
    print(f"  Spearman 95% interval [{PAPER_STATS['sp_lo']:.2f}, "
          f"{PAPER_STATS['sp_hi']:.2f}], permutation p "
          f"{PAPER_STATS['perm_p']:.1e}, error ratio interval "
          f"[{PAPER_STATS['ratio_lo']:.2f}, {PAPER_STATS['ratio_hi']:.2f}]\n")

    # ---- 2. the sweep ---------------------------------------------------
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    rng = np.random.default_rng(20260924)
    boot = np.random.default_rng(7)
    out, t0, drawn = [], time.time(), 0
    for i in range(N):
        drawn += 1
        d = int(rng.integers(2, 6))
        prm = dict(
            a=rng.normal(0.0, 0.6, d).tolist(),
            E=(rng.uniform(0.6, 1.2)
               + 10 ** rng.uniform(-2.2, 0.1) * rng.normal(0, 1, d)).tolist(),
            c=rng.uniform(-0.9, 2.5, d).tolist(),
            u0=float(rng.uniform(0.3, 2.5)),
            x0=float(10 ** rng.uniform(-3.2, -1.7)),
            gain=float(10 ** rng.uniform(1.5, 4.0)))
        try:
            r = downstream(**prm)
            q = downstream(**prm, tilts="random")
        except Exception:
            r = q = None
        row = dict(i=i, d=d, params=prm, ok=r is not None and q is not None)
        if row["ok"]:
            row.update(summarise(*r, boot))
            L2, E2, B2 = q
            med2 = np.median(E2, axis=1)
            s2 = summarise(L2, E2, boot)
            row["random"] = dict(
                sp_loss=s2["sp"], lo=s2["lo"], hi=s2["hi"],
                sp_angle=spearman(B2, med2),       # the naive competitor
                sp_loss_angle=spearman(L2, B2))    # how different they are
        out.append(row)
        if (i + 1) % 10 == 0:
            kept = sum(o["ok"] for o in out)
            print(f"  {i+1}/{N}  usable {kept}  {time.time()-t0:.0f}s",
                  flush=True)
            io.open(os.path.join(HERE, "downstream_sweep.json"), "w",
                    encoding="utf-8").write(json.dumps(dict(paper=PAPER_STATS, plants=out)))

    io.open(os.path.join(HERE, "downstream_sweep.json"), "w",
            encoding="utf-8").write(json.dumps(dict(paper=PAPER_STATS, plants=out)))
    ok = [o for o in out if o["ok"]]
    sp = np.array([o["sp"] for o in ok])
    print(f"\n{len(ok)} usable plants of {drawn} drawn, "
          f"{time.time()-t0:.0f}s")
    print(f"  paper design: Spearman median {np.median(sp):.2f}, "
          f"min {sp.min():.2f}, interval excludes zero in "
          f"{np.mean([o['lo'] > 0 for o in ok]):.0%}")
    rl = np.array([o["random"]["sp_loss"] for o in ok])
    ra = np.array([o["random"]["sp_angle"] for o in ok])
    print(f"  random tilts: loss    Spearman median {np.median(rl):.2f}, "
          f"interval excludes zero in "
          f"{np.mean([o['random']['lo'] > 0 for o in ok]):.0%}")
    print(f"                angle   Spearman median {np.median(ra):.2f}")
    print(f"                loss beats the mid-life angle on "
          f"{np.mean(rl > ra):.0%} of plants")
