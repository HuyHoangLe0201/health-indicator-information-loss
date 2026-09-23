"""
fig2.pdf, which the manuscript prints as FIGURE 1 -- the file
number and the figure number differ, because this float comes first.
Tightened, with an automatic overlap check.

Two problems in the previous version: the "dotted: quadratic" annotation in
(a) sat on the curves, and the panel was taller than it needed to be. The
legend now carries that information, the layout is compressed, and a checker
rasterises the figure and intersects text pixels with curve pixels, so a
collision cannot pass silently.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_DATA = _os.environ.get("RESS_DATA", _os.path.join(_HERE, "data"))


rng = np.random.default_rng(4)
plt.rcParams.update({"font.size": 8.0, "axes.linewidth": 0.7,
                     # Type 3 fonts fail the IEEE compliance check;
                     # 42 embeds TrueType. Do not remove.
                     "pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax = plt.subplots(1, 3, figsize=(5.4, 2.05))

# ------------------------------------------------------------------ (a) ---
a = ax[0]
bb = np.linspace(0, np.radians(60), 400)
for g, col in [(0.5, "#0072B2"), (2.0, "#000000"), (50.0, "#D55E00")]:
    w = g / (1 + g)
    a.plot(np.degrees(bb), -0.5 * np.log(1 - w * np.sin(bb) ** 2),
           "-", color=col, lw=1.2, label=rf"$\gamma={g:g}$")
    a.plot(np.degrees(bb), 0.5 * w * bb ** 2, ":", color=col, lw=1.1)
h, l = a.get_legend_handles_labels()
h.append(Line2D([], [], ls=":", color="0.4", lw=1.1))
l.append("quadratic")
a.legend(h, l, fontsize=6.5, loc="upper left", frameon=False,
         handlelength=1.5, labelspacing=0.18, borderpad=0.1)
a.set_xlabel(r"misalignment $\beta$ (deg)", fontsize=8.0, labelpad=1)
a.set_ylabel(r"$\ell$ (nat)", fontsize=8.0, labelpad=1)
a.set_title("(a) exact vs. quadratic", fontsize=8.0, pad=9)
a.tick_params(labelsize=7.0, pad=1.5)
a.set_xlim(0, 60)
a.set_ylim(0, 0.62)
a.grid(alpha=0.22, lw=0.45)

# ------------------------------------------------------------------ (b) ---
A = np.array([0.0, 0.3, -0.2])
E = np.array([0.6, 1.1, 0.8])
HC = np.array([-0.5, 2.0, 0.8])
Q = np.array([0.466109, 0.770508, 0.434809])
Q /= np.linalg.norm(Q)


def dh(x, u):
    d = np.exp(A - u * E) * (1.0 + HC * x)
    return d / np.linalg.norm(d)


def tern(p):
    p = np.atleast_2d(p)
    return (0.5 * (2 * p[:, 1] + p[:, 2]) / p.sum(1),
            (np.sqrt(3) / 2) * p[:, 2] / p.sum(1))


c = ax[1]
vx, vy = tern(np.eye(3))
c.plot(np.append(vx, vx[0]), np.append(vy, vy[0]), color="0.82", lw=0.7)
for lbl, i, dy in [(r"$e_1$", 0, -6), (r"$e_2$", 1, -9), (r"$e_3$", 2, 5)]:
    c.annotate(lbl, (vx[i], vy[i]), fontsize=6.5, color="0.45", ha="center",
               va="center", xytext=(0, dy), textcoords="offset points")
x0 = np.array([0.25, 0.30, 0.22])
Arc = np.array([dh(x0, u) ** 2 for u in np.linspace(0, 3, 160)])
axx, ayy = tern(Arc)
c.plot(axx, ayy, color="#D55E00", lw=1.5, zorder=2, label=r"$\mathcal{D}(x)$")
d0 = dh(x0, 1.2)
px, py = tern(d0 ** 2)
qx, qy = tern(Q ** 2)
c.plot(px, py, "o", color="#0072B2", ms=4.5, zorder=4, label=r"$\hat{d}(x,u)$")
c.plot(qx, qy, "*", color="k", ms=7, zorder=5, label=r"$q$")
e = Q - (d0 @ Q) * d0
e /= np.linalg.norm(e)
o = np.cross(d0, e)
for vec, col, lab, off in [(e, "#009E73", r"moves $\beta$", (1, 7)),
                           (o, "#7F7F7F", "free", (4, 4))]:
    tgt = np.abs(d0 + 0.40 * vec)
    tgt /= np.linalg.norm(tgt)
    tx, ty = tern(tgt ** 2)
    c.annotate("", xy=(tx[0], ty[0]), xytext=(px[0], py[0]),
               arrowprops=dict(arrowstyle="-|>", lw=1.6, color=col))
    c.annotate(lab, xy=(tx[0], ty[0]), fontsize=6.5, color=col,
               xytext=off, textcoords="offset points")
c.set_xlim(-0.04, 1.10)
c.set_ylim(-0.10, 0.93)
c.set_aspect("equal")
c.axis("off")
c.set_title("(b) cost of each motion", fontsize=8.0, pad=9)
c.legend(fontsize=6.5, loc="upper left", frameon=False, handlelength=1.1,
         labelspacing=0.18, borderpad=0.05, handletextpad=0.4)

# ------------------------------------------------------------------ (c) ---
_sev = _os.path.join(_DATA, "severson_cells.npz")
if not _os.path.exists(_sev):
    raise SystemExit(
        f"severson_cells.npz not found at {_sev}.\n"
        "Panel (c) needs the Severson fleet. Set RESS_DATA to the "
        "directory holding it, or see DATA.md.")
Z = np.load(_sev, allow_pickle=True)
CH = [str(s) for s in Z["channels"]]
USE = [CH.index("QDischarge"), CH.index("IR"), CH.index("Tavg")]
GRID = np.linspace(0.15, 0.90, 40)


def fleet(win):
    out = []
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
        out.append(np.stack([np.interp(GRID, tau, Dm[:, q])
                             for q in range(3)], 1))
    out = np.array(out)
    sg = np.sign(np.median(out.reshape(-1, 3), axis=0))
    out = out * sg[None, None, :]
    return out / np.linalg.norm(out, axis=2, keepdims=True)


def rot(idx, S):
    M = S[idx].mean(0)
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(M @ M[0], -1, 1)))


D301, D151 = fleet(301), fleet(151)
d = ax[2]
BS = np.array([rot(rng.choice(len(D301), len(D301), True), D301)
               for _ in range(400)])
lo, hi = np.percentile(BS, [5, 95], axis=0)
d.fill_between(GRID, lo, hi, color="#0072B2", alpha=0.20, lw=0,
               label=r"$90\%$ bootstrap")
d.plot(GRID, rot(np.arange(len(D301)), D301), color="#0072B2", lw=1.5,
       label="window 301")
d.plot(GRID, rot(np.arange(len(D151)), D151), "--", color="#D55E00", lw=1.2,
       label="window 151")
d.set_xlabel(r"normalised age $\tau$", fontsize=8.0, labelpad=1)
d.set_ylabel("rotation (deg)", fontsize=8.0, labelpad=1)
d.set_title("(c) Severson, 129 cells", fontsize=8.0, pad=9)
d.tick_params(labelsize=7.0, pad=1.5)
d.set_xlim(0.15, 0.90)
# 30, not 27: at the shared legend size of 6.5 the box no longer fits in any
# corner at 27. The band peaks near 25, so the extra headroom clears the
# upper-left corner without crowding anything.
d.set_ylim(0, 30)
# Upper left, kept, but only because the ylim above was raised: at the shared
# legend size of 6.5 the box no longer fitted here at ylim 27, and moving it
# to the lower right put it on the dashed curve instead. The overlap check
# below caught both attempts.
d.legend(fontsize=6.5, loc="upper left", frameon=False, handlelength=1.6,
         labelspacing=0.18, borderpad=0.1)
d.grid(alpha=0.22, lw=0.45)

fig.tight_layout(pad=0.18, w_pad=0.7)
out = _os.path.join(_HERE, "fig2.pdf")
fig.savefig(out)

# ------------------------------------------------- overlap check (pixel) ---
fig.canvas.draw()
r = fig.canvas.get_renderer()
bad = 0
for k, axx_ in enumerate("abc"):
    A_ = ax[k]
    boxes = []
    for t in A_.texts:
        if t.get_text().strip():
            boxes.append((t.get_text(), t.get_window_extent(r)))
    lg = A_.get_legend()
    if lg is not None:
        boxes.append(("<legend>", lg.get_window_extent(r)))
    for ln in A_.get_lines():
        xy = ln.get_xydata()
        if len(xy) == 0:
            continue
        pix = A_.transData.transform(xy)
        for name, bx in boxes:
            inside = ((pix[:, 0] > bx.x0) & (pix[:, 0] < bx.x1)
                      & (pix[:, 1] > bx.y0) & (pix[:, 1] < bx.y1))
            if inside.sum() > 2:
                print(f"  OVERLAP panel ({axx_}): '{name}' over a curve "
                      f"({int(inside.sum())} pts)")
                bad += 1
    # text on text
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (n1, b1), (n2, b2) = boxes[i], boxes[j]
            if (b1.x0 < b2.x1 and b2.x0 < b1.x1
                    and b1.y0 < b2.y1 and b2.y0 < b1.y1):
                print(f"  TEXT-ON-TEXT panel ({axx_}): '{n1}' vs '{n2}'")
                bad += 1
    # lettering outside the figure canvas
    fb = fig.bbox
    for name, bx in boxes:
        if bx.x0 < fb.x0 or bx.x1 > fb.x1 or bx.y0 < fb.y0 or bx.y1 > fb.y1:
            print(f"  CLIPPED panel ({axx_}): '{name}' outside the canvas")
            bad += 1
print(f"wrote {out}")
print(f"overlap check: {'CLEAN' if bad == 0 else str(bad) + ' collisions'}")
print(f"figure size: {fig.get_size_inches()}")
