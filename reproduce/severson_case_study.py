#!/usr/bin/env python3
"""Section 4 of the paper: the five-step procedure on the Severson cells.

Reproduces the rotation arc and the bound (steps 1-3, Fig. 4(b)), the
leave-one-out remaining-life test (step 4) and the replacement policy
(step 5), i.e. every entry of Table 1, and writes them to results/.

    python reproduce/severson_case_study.py [--data DIR] [--out results]
    python reproduce/severson_case_study.py --sensitivity   # + Supplement S1

Needs severson_cells.npz (see DATA.md; hiloss.datasets.extract_severson
rebuilds it from the public batch files). About 10 minutes.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hiloss import datasets, direction, policy, rul  # noqa: E402

AGES = (0.5, 0.7, 0.9)
LABELS = {"all": "all channels", "barycentre": "fixed indicator",
          "channel:0": "capacity", "fusion": "fusion", "channel:2": "temperature"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", help="directory holding severson_cells.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--sensitivity", action="store_true",
                    help="also rerun the RUL test at the four settings of S1")
    a = ap.parse_args()
    if a.data:
        datasets.set_data_dir(a.data)
    os.makedirs(a.out, exist_ok=True)

    units, _ = datasets.load_severson()
    lives = np.array([len(u) for u in units])
    print(f"{len(units)} cells, lives {lives.min()} to {lives.max()} cycles")

    # ---- steps 1-3 ---------------------------------------------------------
    fd = direction.observed_direction(units, window=301,
                                      channels=datasets.SEVERSON_CHANNELS)
    fd151 = direction.observed_direction(units, window=151)
    s = fd.summary()
    print("\nSteps 1-3 (window 301 cycles, tau in [0.15, 0.90])")
    print(f"  arc of the observed direction       {s['arc_deg']:.1f} deg"
          f"   (window 151: {fd151.arc_deg:.1f} deg)")
    print(f"  first-to-last angle                 {s['end_to_end_deg']:.0f} deg")
    print(f"  bound on error-variance inflation   {s['excess_variance_pct']:.1f} %")
    print(f"  bound on safety-margin widening     {s['margin_pct']:.1f} %")
    print(f"  fixed indicator (whitened weights)  "
          f"({', '.join(f'{w:.2f}' for w in s['indicator'])}),"
          f" at most {s['max_angle_deg']:.0f} deg from the path")
    cap = fd.max_angle_deg([1, 0, 0])
    print(f"  capacity alone                      at most {cap:.0f} deg,"
          f" bound {100 * (1 / np.cos(np.radians(cap)) ** 2 - 1):.0f} %")

    lo, hi = direction.bootstrap_rotation(fd)
    with open(os.path.join(a.out, "severson_rotation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau", "rotation_deg_w301", "boot5", "boot95",
                    "rotation_deg_w151"])
        for row in zip(fd.grid, fd.rotation_deg(), lo, hi, fd151.rotation_deg()):
            w.writerow([f"{x:.6g}" for x in row])

    # ---- step 4 --------------------------------------------------------------
    print("\nStep 4: leave-one-out remaining-life estimation ...", flush=True)
    model = rul.FleetModel(units)
    res = rul.leave_one_out(model, AGES)
    ratios, pooled = rul.mse_ratios(res)
    rms = [np.sqrt(np.mean(np.square(res["err_cycles"][t]))) for t in AGES]
    print(f"  all channels: RMS error {rms[0]:.0f}, {rms[1]:.0f}, {rms[2]:.0f}"
          f" cycles at 50, 70, 90 % of life")
    wf = res["fusion_weights"].mean(0)
    wf /= np.linalg.norm(wf)
    ang = {"barycentre": s["max_angle_deg"], "channel:0": cap,
           "fusion": fd.max_angle_deg(wf), "channel:2": fd.max_angle_deg([0, 0, 1])}

    # ---- step 5 --------------------------------------------------------------
    print("Step 5: replacement policy ...", flush=True)
    pred = policy.inspection_predictions(model)
    pol = policy.evaluate(pred, model.lives)

    rows = []
    print(f"\nTable 1 {'input':<16}{'psi_max':>8}{'MSE 50%':>9}{'70%':>8}{'90%':>8}"
          f"{'cost':>8}{'fail':>7}{'unused':>8}")
    for m in ("all", "barycentre", "channel:0", "fusion", "channel:2"):
        r = [1.0, 1.0, 1.0] if m == "all" else [ratios[m][t] for t in AGES]
        p = pol.get(m, {}).get(10.0)
        row = dict(input=LABELS[m], psi_max_deg=ang.get(m),
                   mse_50=r[0], mse_70=r[1], mse_90=r[2],
                   cost_per_1000=None if p is None else 1000 * p["rate"],
                   fail_pct=None if p is None else 100 * p["fail"],
                   unused_cycles=None if p is None else p["unused"])
        rows.append(row)
        f = (lambda v, fmt: "---" if v is None else fmt.format(v))
        print(f"        {row['input']:<16}{f(row['psi_max_deg'], '{:.0f}'):>8}"
              f"{r[0]:>9.3g}{r[1]:>8.3g}{r[2]:>8.3g}"
              f"{f(row['cost_per_1000'], '{:.3f}'):>8}"
              f"{f(row['fail_pct'], '{:.1f}'):>7}{f(row['unused_cycles'], '{:.0f}'):>8}")
    dev = max(abs(pol["barycentre"][r]["rate"] / pol["all"][r]["rate"] - 1)
              for r in (5.0, 10.0, 20.0, 50.0))
    print(f"\n  fixed indicator within {100 * dev:.2f} % of all channels for cost"
          " ratios 5 to 50;"
          f" fusion costs {100 * (pol['fusion'][10.0]['rate'] / pol['all'][10.0]['rate'] - 1):.1f} % more")
    with open(os.path.join(a.out, "table1_severson.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows([{k: (f"{v:.6g}" if isinstance(v, float) else v)
                      for k, v in r.items()} for r in rows])

    if a.sensitivity:
        print("\nSupplement S1: the RUL test at other settings (pooled MSE ratio)")
        for back, nbase in ((0.20, 50), (0.50, 50), (0.30, 20), (0.30, 100)):
            mdl = rul.FleetModel(units, nbase=nbase)
            _, pl = rul.mse_ratios(rul.leave_one_out(mdl, AGES, back=back))
            print(f"  back {back:.2f}, baseline {nbase:3d}: fixed indicator "
                  f"{pl['barycentre']:.3f}, capacity {pl['channel:0']:.3f}")
    print(f"\nwrote {a.out}/table1_severson.csv and {a.out}/severson_rotation.csv")


if __name__ == "__main__":
    main()
