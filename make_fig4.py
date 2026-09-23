r"""The reachability floor over life, and the schedule that removes it.

Sections 6.3 and 6.4 state this result in words and three numbers, and nothing
plots it. Panel (a) is Section 6.3: with one input the floor is largest at the
start of life at 4.52 degrees, falls through mid-life below the sampling
resolution, and rises again towards failure. Panel (b) is Section 6.4: with a
second input the residual is 8.5e-7 degrees, zero to precision.

Panel (b) does NOT plot that residual. On any axis it is a flat line at zero,
which shows the reader nothing about why it is zero. What it plots instead is
the operating point the alignment demands: u1 has to sweep most of the
envelope over life while u2 barely moves, and both stay inside [0,3]^2 at all
600 states. That containment is the content of Proposition 4.4 and is the
reason the floor vanishes; the residual itself is one annotated number.

The two panels use different targets, because their two sections do. Panel (a)
measures against q*, the optimal fixed indicator, along the closed loop at q*.
Panel (b) measures against the direction the plant occupies at mid-life under
the central operating point (1.5, 1.5), on that plant's own trajectory -- the
open circle in (b), where both inputs equal 1.5 by construction. The caption
says so. Putting both curves on one axis would have read as a single
controlled comparison, which it is not.

No \mathcal: matplotlib's default dejavusans mathtext has no calligraphic
alphabet, and figs 1-3 all use that default, so switching fontsets here would
make this figure's symbols disagree with theirs.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_DATA = _os.environ.get("RESS_DATA", _os.path.join(_HERE, "data"))


# IEEE Xplore and Elsevier both reject Type 3 fonts, and matplotlib's
# pdf.fonttype defaults to exactly that. Do not remove.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
# one type scale shared with make_fig1.py, make_fig2.py and make_fig3.py
plt.rcParams["font.size"] = 8.0

A = np.array([0.0, 0.3, -0.2])
E = np.array([0.6, 1.1, 0.8])
G = np.array([0.9, 0.3, 1.2])
HC = np.array([-0.5, 2.0, 0.8])
UMIN, UMAX, XFAIL, X0 = 0.0, 3.0, 0.9, 5e-3
Q = np.array([0.466109, 0.770508, 0.434809])
Q = Q / np.linalg.norm(Q)
NSTATE = 600


def dhat(x, u1, u2=0.0):
    d = np.exp(A - u1 * E - u2 * G) * (1.0 + HC * x)
    return d / np.linalg.norm(d)


def trajectory(law, n=NSTATE):
    def rhs(t, x):
        u1, u2 = law(x)
        return np.exp(A - u1 * E - u2 * G) * (1.0 + HC * x)

    def ev(t, x):
        return x.max() - XFAIL
    ev.terminal, ev.direction = True, 1
    s = solve_ivp(rhs, [0, 1e4], np.full(3, X0), events=ev,
                  rtol=1e-10, atol=1e-12, dense_output=True)
    tg = np.linspace(0, s.t[-1], n)
    return s.sol(tg).T, np.linspace(0, 1, n)


# ---- (a) one input: the floor against q*, along the closed loop at q* ----
US = np.linspace(UMIN, UMAX, 41)


def greedy(x):
    D = np.array([dhat(x, u) for u in US])
    return (US[int(np.argmax(D @ Q))], 0.0)


X1, tau = trajectory(greedy)
UF = np.linspace(UMIN, UMAX, 2001)
PHI = np.exp(A[None, :] - UF[:, None] * E[None, :])
floor1 = np.empty(len(X1))
for i, x in enumerate(X1):
    D = PHI * (1.0 + HC[None, :] * x[None, :])
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    floor1[i] = np.degrees(np.arccos(np.clip((D @ Q).max(), -1, 1)))

# ---- (b) two inputs: the schedule that holds the mid-life direction -----
X2, tau2 = trajectory(lambda x: (1.5, 1.5))
tgt = dhat(X2[len(X2) // 2], 1.5, 1.5)
lt = np.log(tgt)
M = np.array([[E[0] - E[1], G[0] - G[1]],
              [E[1] - E[2], G[1] - G[2]]])
uu = np.empty((len(X2), 2))
res = np.empty(len(X2))
for i, x in enumerate(X2):
    h = np.log(1.0 + HC * x)
    rhs = np.array([(lt[0] - lt[1]) - (h[0] - h[1]) - (A[0] - A[1]),
                    (lt[1] - lt[2]) - (h[1] - h[2]) - (A[1] - A[2])])
    uu[i] = np.linalg.solve(-M, rhs)
    uc = np.clip(uu[i], UMIN, UMAX)
    res[i] = np.degrees(np.arccos(np.clip(dhat(x, uc[0], uc[1]) @ tgt, -1, 1)))

inside = int(np.sum(np.all((uu >= UMIN - 1e-9) & (uu <= UMAX + 1e-9), axis=1)))

# the figure must not quietly disagree with the text it illustrates
assert abs(floor1.max() - 4.52) < 0.01, floor1.max()
assert floor1.min() < 0.01, floor1.min()
assert inside == NSTATE, inside
assert res.max() < 1e-5, res.max()

# ---- draw ---------------------------------------------------------------
fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.4, 2.25))

ax.semilogy(tau, floor1, color="0.15", lw=1.4)
i0 = int(floor1.argmax())
imin = int(floor1.argmin())
ax.plot([tau[i0], tau[imin]], [floor1[i0], floor1[imin]], "o", ms=3.5,
        mfc="w", mec="0.15", mew=1.0, ls="none")
ax.annotate(r"$4.52^{\circ}$ at birth", (tau[i0], floor1[i0]),
            xytext=(7, 0), textcoords="offset points", fontsize=7.0,
            color="0.2")
ax.annotate("below the $0.01^{\\circ}$\nsampling grid",
            (tau[imin], floor1[imin]), xytext=(0.28, 8e-3),
            textcoords="data", fontsize=6.5, color="0.35",
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="0.55",
                            shrinkA=2.0, shrinkB=3.0))
ax.set_xlabel(r"normalised age $\tau$", fontsize=8.0)
ax.set_ylabel(r"floor $\beta_{\min}$ against $q^{\star}$  (deg)", fontsize=8.0)
ax.set_title(r"(a)  $m=1<d-1$", fontsize=8.0, loc="left")
ax.set_xlim(0, 1)
ax.set_ylim(1.1e-3, 40)

bx.plot(tau2, uu[:, 0], color="0.15", lw=1.4)
bx.plot(tau2, uu[:, 1], color="0.55", lw=1.4, ls="--")
bx.axhline(UMIN, color="0.75", lw=0.7, ls=":")
bx.axhline(UMAX, color="0.75", lw=0.7, ls=":")
bx.annotate("envelope $[0,3]$", (0.02, UMAX), xytext=(0, -11),
            textcoords="offset points", fontsize=6.5, color="0.45")
# direct labels: a legend box has nowhere to sit that is not on a curve
j = int(0.80 * (NSTATE - 1))
bx.annotate(r"$u_1$", (tau2[j], uu[j, 0]), xytext=(1, 6),
            textcoords="offset points", fontsize=7.5, color="0.15")
bx.annotate(r"$u_2$", (tau2[j], uu[j, 1]), xytext=(1, -12),
            textcoords="offset points", fontsize=7.5, color="0.5")
# the mid-life state the target direction is read from; both curves pass
# through 1.5 there by construction
k = NSTATE // 2
bx.plot([tau2[k]], [1.5], "o", ms=3.5, mfc="w", mec="0.3", mew=1.0)
bx.annotate("target read here", (tau2[k], 1.5), xytext=(-5, 8),
            textcoords="offset points", fontsize=6.5, color="0.35",
            ha="right")
e = int(np.floor(np.log10(res.max())))
bx.annotate(rf"residual ${res.max() / 10.0 ** e:.1f}\times10^{{{e}}}$ deg",
            (0.5, 0.10), xycoords="axes fraction", fontsize=7.0, color="0.2",
            ha="center")
bx.set_xlabel(r"normalised age $\tau$", fontsize=8.0)
bx.set_ylabel(r"required operating point", fontsize=8.0)
bx.set_title(r"(b)  $m=2=d-1$", fontsize=8.0, loc="left")
bx.set_xlim(0, 1)
bx.set_ylim(-0.25, 3.25)

for a in (ax, bx):
    a.tick_params(labelsize=7.0)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)

fig.tight_layout(pad=0.3, w_pad=1.4)
out = _os.path.join(_HERE, "fig4.pdf")
fig.savefig(out)
print(f"wrote {out}")
print(f"  (a) peak    {floor1.max():.4f} deg at tau={tau[i0]:.3f}   "
      f"(paper: 4.52 at start of life)")
print(f"  (a) minimum {floor1.min():.4f} deg at tau={tau[imin]:.3f}   "
      f"(paper: below 0.01)")
print(f"  (a) failure {floor1[-1]:.4f} deg")
print(f"  (b) inside [0,3]^2 {inside} of {NSTATE}   (paper: all 600)")
print(f"  (b) residual {res.max():.3e} deg   (paper: 8.5e-7)")
print(f"  (b) u1 spans [{uu[:, 0].min():.3f}, {uu[:, 0].max():.3f}], "
      f"u2 spans [{uu[:, 1].min():.3f}, {uu[:, 1].max():.3f}]")
