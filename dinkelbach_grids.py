#!/usr/bin/env python3
r"""The fractional optimum of Section 6.6 on a sequence of state and control
grids, recorded to dinkelbach_grids_NU*.json. verify_numbers.py reads the record
and recomputes two of its entries afresh.

The scheme is the semi-Lagrangian discretisation of the stationary
Hamilton--Jacobi--Bellman equation of the Dinkelbach problem: fixed spatial
step h = 1.6 dx along the normalised drift, multilinear interpolation, running
cost (ell - lambda) dt, zero on the condemnation surface.

Why one sweep is exact. The drift is positive, so every foot point
P + st*g has coordinates no smaller than those of P, and the corners of its
interpolation cell other than P itself have a larger index sum. Processing
nodes in decreasing index sum, every such corner is final when P is reached,
and P's own weight a_u enters linearly: W(P) = min_u (c_u + r_u)/(1 - a_u),
which is the fixed point that value iteration on the same scheme converges to.
Run at N = 25, 29, 35 this reproduces the values that value iteration gave
(4.1554, 3.8045, 3.4318e-4) to the bisection tolerance.

Dinkelbach's own iteration replaces bisection: at lambda_k the minimising
discrete policy is carried through the same sweep to give its cost-to-go L and
time-to-go T at x0, and lambda_{k+1} = L/T. The discrete policies are finite,
so the iteration terminates; at the fixed point W(x0) = 0 is checked.

    python dinkelbach_grids.py NU N1 N2 ...      # appends to the record

The whole sequence used by the paper takes several hours, N = 137 at
NU = 161 alone about two:
    NU = 41, 81 and 161:  N = 25 35 49 69 97 137
"""
import io
import json
import os
import sys
import time

import numpy as np

import verify_numbers as vn

A, E, HC, QS = vn.A, vn.E, vn.HC, vn.QS
X0, XF = vn.X0, vn.XFAIL
HERE = os.path.dirname(os.path.abspath(__file__))


def record_path(NU):
    """One record per control grid, so that the three sequences can run in
    parallel without two processes rewriting one file."""
    return os.path.join(HERE, f"dinkelbach_grids_NU{int(NU)}.json")


def solve(lam, N, NU, hfac=1.6, want_LT=True):
    """W, L, T at x0 for the discrete Dinkelbach problem at lambda."""
    ax = np.linspace(X0, XF, N)
    dx = ax[1] - ax[0]
    h = hfac * dx
    US = np.linspace(vn.UMIN, vn.UMAX, NU)
    PHI = np.exp(A[None, :] - US[:, None] * E[None, :])
    Wf = np.zeros(N ** 3)
    Lf = np.zeros(N ** 3)
    Tf = np.zeros(N ** 3)
    I, J, K = np.meshgrid(np.arange(N), np.arange(N), np.arange(N),
                          indexing="ij")
    I, J, K = I.ravel(), J.ravel(), K.ravel()
    live = (I < N - 1) & (J < N - 1) & (K < N - 1)
    lev = I + J + K
    order = np.argsort(-lev[live], kind="stable")
    nodes = np.flatnonzero(live)[order]
    cuts = np.flatnonzero(np.diff(lev[nodes])) + 1
    corners = np.array([[a, b, c] for a in (0, 1) for b in (0, 1)
                        for c in (0, 1)])
    for blk in np.split(nodes, cuts):
        ii = np.stack([I[blk], J[blk], K[blk]], 1)
        P = ax[ii]
        f = PHI[None, :, :] * (1.0 + HC[None, None, :] * P[:, None, :])
        nf = np.linalg.norm(f, axis=2)
        gg = f / nf[:, :, None]
        with np.errstate(divide="ignore"):
            si = (XF - P)[:, None, :] / gg
        st = np.minimum(h, si.min(axis=2))
        hit = st < h - 1e-15
        ti = st / nf
        ru = vn.ell(np.arccos(np.clip(gg @ QS, -1, 1)), nf ** 2)
        foot = P[:, None, :] + st[:, :, None] * gg
        pos = (foot - X0) / dx
        idx = np.clip(np.floor(pos).astype(int), 0, N - 2)
        fr = np.clip(pos - idx, 0.0, 1.0)
        own = np.all(idx == ii[:, None, :], axis=2)
        a_own = np.where(own, np.prod(1.0 - fr, axis=2), 0.0)
        rW = np.zeros_like(ti)
        rL = np.zeros_like(ti)
        rT = np.zeros_like(ti)
        for c in corners:
            ci = idx + c[None, None, :]
            wgt = np.prod(np.where(c[None, None, :] == 1, fr, 1.0 - fr),
                          axis=2)
            flat = (ci[..., 0] * N + ci[..., 1]) * N + ci[..., 2]
            rW += wgt * Wf[flat]
            if want_LT:
                rL += wgt * Lf[flat]
                rT += wgt * Tf[flat]
        # the nodes of this block are still 0 in Wf, so rW leaves out the
        # node's own weight, which enters through a_own
        cont = np.where(hit, 0.0, 1.0)
        den = 1.0 - cont * a_own
        cand = (ti * (ru - lam) + cont * rW) / den
        j = np.argmin(cand, axis=1)
        r = np.arange(len(blk))
        Wf[blk] = cand[r, j]
        if want_LT:
            Lf[blk] = (ti[r, j] * ru[r, j] + cont[r, j] * rL[r, j]) / den[r, j]
            Tf[blk] = (ti[r, j] + cont[r, j] * rT[r, j]) / den[r, j]
    return Wf[0], Lf[0], Tf[0]


def dinkelbach(N, NU, lam0=2e-4, tol=1e-13, maxit=40):
    lam = lam0
    for it in range(1, maxit + 1):
        _, L0, T0 = solve(lam, N, NU)
        new = L0 / T0
        if abs(new - lam) < tol:
            lam = new
            break
        lam = new
    W0 = solve(lam, N, NU, want_LT=False)[0]
    assert abs(W0) < 1e-15, (N, NU, W0)
    return lam, it


def load(NU):
    """{N: lambda} for one control grid, or {} if not yet run."""
    p = record_path(NU)
    if not os.path.exists(p):
        return {}
    return json.load(io.open(p, encoding="utf-8"))


# The extrapolation of the record lives in verify_numbers.dinkelbach_record,
# the one place the reported interval is computed.


if __name__ == "__main__":
    NU = int(sys.argv[1])
    for N in map(int, sys.argv[2:]):
        t = time.time()
        lam, it = dinkelbach(N, NU)
        rec = load(NU)
        rec[str(N)] = lam
        with io.open(record_path(NU), "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1, sort_keys=True)
        print(f"NU={NU} N={N}: {lam:.10e} ({it} it, {time.time()-t:.0f}s)",
              flush=True)
