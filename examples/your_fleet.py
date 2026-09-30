#!/usr/bin/env python3
"""Apply the five-step procedure of the paper to YOUR run-to-failure data.

Input: a CSV in long format with a unit id, a cycle (or time) index and one
column per monitoring channel, e.g.

    unit,cycle,capacity,resistance,temperature
    0,0,1.0998,0.0201,31.02
    0,1,1.0997,0.0199,30.87
    ...

Every unit must be observed from the start of service to failure (its last
row is its end of life), at equally spaced cycles.

    python examples/your_fleet.py data.csv                     # all channels
    python examples/your_fleet.py data.csv --channels capacity resistance
    python examples/your_fleet.py --demo demo.csv              # write a demo CSV
    python examples/your_fleet.py data.csv --window 101 --policy

Steps 1-3 are fast. Step 4 (leave-one-out remaining-life test) takes about
a minute per hundred units; --policy adds step 5.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from hiloss import direction, policy, rul  # noqa: E402


def read_fleet(path, channels=None, unit_col="unit", cycle_col="cycle"):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    cols = channels or [c for c in rows[0] if c not in (unit_col, cycle_col)]
    by = defaultdict(list)
    for r in rows:
        by[r[unit_col]].append((float(r[cycle_col]), [float(r[c]) for c in cols]))
    units = []
    for u in sorted(by, key=lambda s: (len(s), s)):
        seq = sorted(by[u])
        units.append(np.array([v for _, v in seq]))
    return units, cols


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--channels", nargs="+")
    ap.add_argument("--window", type=int, default=301,
                    help="Savitzky-Golay window in cycles (capped per unit)")
    ap.add_argument("--grid", nargs=3, type=float, default=(0.15, 0.90, 40),
                    metavar=("START", "END", "POINTS"),
                    help="normalised-life grid of the observed direction")
    ap.add_argument("--no-rul", action="store_true", help="skip step 4")
    ap.add_argument("--policy", action="store_true", help="also run step 5")
    ap.add_argument("--demo", metavar="OUT.csv",
                    help="write a demo fleet in the expected format and exit")
    a = ap.parse_args()

    if a.demo:
        from demo_fleet import demo_fleet, write_csv
        write_csv(a.demo, demo_fleet())
        print(f"wrote {a.demo}; now run: python examples/your_fleet.py {a.demo}")
        return
    if not a.csv:
        ap.error("give a CSV file, or --demo OUT.csv")

    units, cols = read_fleet(a.csv, a.channels)
    lives = [len(u) for u in units]
    grid = np.linspace(a.grid[0], a.grid[1], int(a.grid[2]))
    print(f"{len(units)} units, channels {cols}, lives {min(lives)} to {max(lives)}")

    fd = direction.observed_direction(units, window=a.window, grid=grid,
                                      channels=cols)
    s = fd.summary()
    sigma = np.median([direction.unit_derivative(u, a.window, grid)[1]
                       for u in units], axis=0)
    print("\nStep 1-2  the observed direction")
    print(f"  arc over tau in [{grid[0]:g}, {grid[-1]:g}]      {s['arc_deg']:.1f} deg")
    print(f"  bound on error-variance inflation   {s['excess_variance_pct']:.1f} %"
          "   (any fixed indicator within half the arc)")
    print(f"  bound on safety-margin widening     {s['margin_pct']:.1f} %")
    st3 = fd.step3()
    print("\nStep 3  the fixed indicator")
    print(f"  barycentre of the path: at most {st3['barycentre_max_deg']:.1f} deg"
          f" from it (half the arc: {st3['half_arc_deg']:.1f} deg)")
    if not st3["passes"]:
        print("  -> the barycentre is NOT covered by the bound; using the arc"
              f" midpoint instead (at most {st3['midpoint_max_deg']:.1f} deg)")
    v = st3["indicator"]
    a_raw = fd.signs * v / sigma
    a_raw /= np.linalg.norm(a_raw)
    print(f"  {st3['choice']}, whitened weights  " + "  ".join(
        f"{c}={w:+.3f}" for c, w in zip(cols, v)))
    print(f"  {st3['choice']}, raw weights       " + "  ".join(
        f"{c}={w:+.3g}" for c, w in zip(cols, a_raw)))
    print(f"  its own bound on error-variance inflation"
          f" {100 * (st3['bound'] - 1):.1f} %")
    for j, c in enumerate(cols):
        e = np.zeros(len(cols))
        e[j] = 1.0
        print(f"  channel '{c}' alone: at most {fd.max_angle_deg(e):.0f} deg from the path")

    if a.no_rul:
        return
    print("\nStep 4  leave-one-out remaining-life test ...", flush=True)
    model = rul.FleetModel(units, window=a.window, grid=grid)
    names = ("all", "barycentre", "midpoint", "fusion") + tuple(
        f"channel:{j}" for j in range(len(cols)))
    res = rul.leave_one_out(model, names=names, lmax=int(1.5 * max(lives)))
    ratios, pooled = rul.mse_ratios(res)
    label = {f"channel:{j}": c for j, c in enumerate(cols)}
    label.update(barycentre="barycentre", midpoint="arc midpoint", fusion="fusion")
    print("  mean squared error relative to all channels, at 50 / 70 / 90 % of life")
    for m in names[1:]:
        r = ratios[m]
        print(f"  {label[m]:<16} {r[0.5]:7.3f} {r[0.7]:7.3f} {r[0.9]:7.3f}"
              f"   pooled {pooled[m]:.3f}")

    if a.policy:
        print("\nStep 5  predictive replacement policy (failure 10x a replacement) ...",
              flush=True)
        pred = policy.inspection_predictions(model, names=("all", st3["choice"]),
                                             horizon=int(max(lives)))
        out = policy.evaluate(pred, model.lives, cost_ratios=(10.0,))
        for m in ("all", st3["choice"]):
            o = out[m][10.0]
            print(f"  {label.get(m, m):<16} cost {1000 * o['rate']:.3f} per 1000 cycles,"
                  f" {100 * o['fail']:.1f} % failures, {o['unused']:.0f} cycles unused")


if __name__ == "__main__":
    main()
