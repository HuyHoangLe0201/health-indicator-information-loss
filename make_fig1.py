"""
fig1.pdf, which the manuscript prints as FIGURE 2 -- the file
number and the figure number differ, because fig2.pdf's float comes
first. Revised as follows.

(a) unchanged: the information-share curve, the reachable sets, the indicator.
(b) replaced: the regime map. The old panel showed the grid convergence of
    lambda* at an arbitrary indicator; that indicator turned out to be a poor
    one and the number it produced is no longer the headline. The headline is
    what the control buys and how it degrades with the regime index.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_DATA = _os.environ.get("RESS_DATA", _os.path.join(_HERE, "data"))


# IEEE Xplore rejects Type 3 fonts, and matplotlib's pdf.fonttype defaults to
# exactly that. 42 embeds TrueType instead, which the compliance check
# accepts. Do not remove.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
# one type scale shared with make_fig2.py and make_fig3.py
plt.rcParams["font.size"] = 8.0

A = np.array([0.0, 0.3, -0.2])
E = np.array([0.6, 1.1, 0.8])
HC = np.array([-0.5, 2.0, 0.8])
UMIN, UMAX, XFAIL, X0 = 0.0, 3.0, 0.9, 5e-3
Q = np.array([0.466109, 0.770508, 0.434809])
Q = Q / np.linalg.norm(Q)


def dhat(x, u):
    d = np.exp(A - u * E) * (1.0 + HC * x)
    return d / np.linalg.norm(d)


def tern(p):
    p = np.atleast_2d(p)
    return (0.5 * (2 * p[:, 1] + p[:, 2]) / p.sum(1),
            (np.sqrt(3) / 2) * p[:, 2] / p.sum(1))


def rhs(t, x):
    return np.exp(A - 1.2 * E) * (1.0 + HC * x)


def ev(t, x):
    return x.max() - XFAIL


ev.terminal, ev.direction = True, 1
sol = solve_ivp(rhs, [0, 500], np.full(3, X0), events=ev, rtol=1e-11,
                atol=1e-13, dense_output=True, max_step=0.01)
tg = np.linspace(0, sol.t[-1], 400)
X = sol.sol(tg).T
P = np.array([dhat(x, 1.2) ** 2 for x in X])

fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.35))

# ------------------------------------------------------------- panel (a)
a = ax[0]
V = np.eye(3)
vx, vy = tern(V)
a.plot(np.append(vx, vx[0]), np.append(vy, vy[0]), color="0.75", lw=0.8)
for lbl, i in zip([r"$e_1$", r"$e_2$", r"$e_3$"], range(3)):
    a.annotate(lbl, (vx[i], vy[i]), fontsize=6.5, color="0.4", ha="center",
               va="center", xytext=(0, -7 if i < 2 else 6),
               textcoords="offset points")
px, py = tern(P)
a.plot(px, py, color="#0072B2", lw=1.6, label=r"plant $p_\tau$", zorder=3)
a.plot(px[0], py[0], "o", color="#0072B2", ms=3.5, zorder=4)
a.plot(px[-1], py[-1], "s", color="#0072B2", ms=3.5, zorder=4)
US = np.linspace(UMIN, UMAX, 120)
# One colour for all three arcs made them one object; the panel's whole point
# is that the set MOVES, and that the mid-life arc passes through the
# indicator while the other two miss it. Shade by age and label each arc, so
# the reader can tell which is which and check the caption against the plot.
FRACS = [0.05, 0.45, 0.9]
SHADE = ["#F5B78A", "#E07B39", "#8C3A00"]
for j, frac in enumerate(FRACS):
    k = int(frac * (len(X) - 1))
    D = np.array([dhat(X[k], u) ** 2 for u in US])
    dx, dy = tern(D)
    # The three arcs are named in the legend, not on the plot. Inline labels
    # collided: at one end two arcs nearly coincide, at the other they run
    # into beta, the failure marker and the e2 vertex. The legend corner is
    # empty and can hold all three without overprinting anything.
    lab = (rf"reachable $\mathcal{{D}}(x_\tau)$, $\tau={frac:g}$" if j == 0
           else rf"$\tau={frac:g}$")
    a.plot(dx, dy, color=SHADE[j], lw=1.1, alpha=0.95, zorder=2, label=lab)
qx, qy = tern(Q ** 2)
a.plot(qx, qy, "*", color="k", ms=8, zorder=5, label=r"indicator $q^\star$")
kk = int(0.45 * (len(X) - 1))
a.plot([px[kk], qx[0]], [py[kk], qy[0]], ":", color="0.35", lw=0.9, zorder=1)
a.annotate(r"$\beta$", (0.5 * (px[kk] + qx[0]), 0.5 * (py[kk] + qy[0])),
           fontsize=8.0, color="0.2", xytext=(4, 3), textcoords="offset points")
a.set_xlim(-0.06, 1.06)
a.set_ylim(-0.09, 0.95)
a.set_aspect("equal")
a.axis("off")
a.legend(fontsize=6.5, loc="upper left", frameon=False, handlelength=1.4,
         borderpad=0.1, labelspacing=0.25)
a.set_title(r"(a) information shares on the simplex", fontsize=8.0, pad=2)

# ------------------------------------------------------------- panel (b)
b = ax[1]
# These three arrays must agree with Table II of the manuscript; they are
# checked against a fresh recomputation by verify_numbers.py --regime, so a
# drift here fails the gate instead of quietly mislabelling the figure.
rho = np.array([0.883361, 12.5318, 53.0530])
L_ad = np.array([1.51173e-4, 3.23489e-3, 3.76498e-3])
L_co = L_ad * np.array([68.3835, 2.9336, 2.68635])
b.plot(rho, L_co, "s--", color="#D55E00", ms=5, lw=1.2,
       label="best constant input")
b.plot(rho, L_ad, "o-", color="#0072B2", ms=5, lw=1.4,
       label="adaptive control")
for r, la, lc in zip(rho, L_ad, L_co):
    # one decimal below ten, else none: "3x" for both 2.93 and 2.69 would
    # contradict the 2.7 quoted in the text and hide the two regimes apart
    g = lc / la
    b.annotate(rf"${g:.0f}\times$" if g >= 10 else rf"${g:.1f}\times$",
               xy=(r, np.sqrt(la * lc)),
               fontsize=7.5, color="0.2", ha="left",
               xytext=(5, -3), textcoords="offset points")
    b.plot([r, r], [la, lc], color="0.6", lw=0.7, zorder=0)
b.set_xscale("log")
b.set_yscale("log")
b.set_xlabel(r"regime index $\rho$  (drift / reachable arc)", fontsize=8.0)
b.set_ylabel(r"$L^\star$  (nat)", fontsize=8.0)
b.tick_params(labelsize=7.0)
b.set_xlim(0.55, 130)
b.legend(fontsize=6.5, loc="lower right", frameon=False, handlelength=1.8,
         borderpad=0.15, labelspacing=0.3)
b.set_title(r"(b) what the control buys, by regime", fontsize=8.0, pad=2)
b.grid(alpha=0.25, lw=0.5, which="both")

fig.tight_layout(pad=0.35)
out = _os.path.join(_HERE, "fig1.pdf")
fig.savefig(out)
print("wrote", out)
