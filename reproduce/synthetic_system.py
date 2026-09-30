#!/usr/bin/env python3
"""Section 5 of the paper: selecting the operating point, on the synthetic
system. Needs no data.

    python reproduce/synthetic_system.py             # 5.2 and Table 2 (~10 min)
    python reproduce/synthetic_system.py --misspec   # + Table 3 (slow)
    python reproduce/synthetic_system.py --grids     # + control grids 81, 161

Writes results/table2_synthetic.csv (and table3_misspec.csv).
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hiloss import synthetic as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="results")
    ap.add_argument("--misspec", action="store_true", help="Table 3 (slow)")
    ap.add_argument("--grids", action="store_true",
                    help="the two factors on 81- and 161-point control grids")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("The system: activation energies "
          + ", ".join(f"{S.activation_energy_eV(e):.2f}" for e in S.E)
          + " eV; u = 0, 1.5, 3 is "
          + ", ".join(f"{S.celsius(u):.1f}" for u in (0.0, 1.5, 3.0)) + " C")

    arc = S.arc_bound_table()
    print(f"\nArc bound under constant inputs: arc {arc[:, 0].min():.1f} to "
          f"{arc[:, 0].max():.1f} deg, bound {arc[:, 1].min():.1f} to "
          f"{arc[:, 1].max():.1f} %, actual excess {arc[:, 2].min():.1f} to "
          f"{arc[:, 2].max():.1f} %")

    L_my, Tf_my, _ = S.myopic(S.QS)
    L_co, u_co = S.best_constant(S.QS)
    L_fa, u_fa, v_fa = S.fair_constant()
    print(f"\nSteering indicator q* = ({', '.join(f'{x:.3f}' for x in S.QS)})")
    print(f"  myopic policy at q*                 loss {L_my:.3g}")
    print(f"  best constant input at q*           loss {L_co:.3g}  -> factor {L_co / L_my:.0f}")
    print(f"  best constant design, own indicator loss {L_fa:.3g} at u = {u_fa:g}"
          f"  -> factor {L_fa / L_my:.0f}")
    if a.grids:
        for nu in (81, 161):
            la = S.myopic(S.QS, NU=nu)[0]
            print(f"  control grid {nu}: factors {S.best_constant(S.QS, NU=nu)[0] / la:.0f}"
                  f" and {S.fair_constant(NU=nu)[0] / la:.0f}")

    print("\nTable 2 (RUL errors: median |error| / lifetime, 2000 units) ...", flush=True)
    names = ["constant, 60 C", "constant, 41.5 C", "constant, 25 C",
             "myopic, q*", "myopic, balanced"]
    rows = []
    print(f"  {'strategy':<18}{'loss':>10}{'excess':>9}{'RUL ind.':>10}"
          f"{'RUL all':>9}{'T_f':>8}")
    for nm, (L, Tf, ez, er) in zip(names, S.policy_table()):
        rows.append(dict(strategy=nm, loss=L, excess_variance_pct=100 * (np.exp(2 * L) - 1),
                         rul_error_indicator_pct=100 * ez, rul_error_all_pct=100 * er,
                         lifetime=Tf))
        print(f"  {nm:<18}{L:>10.3g}{100 * (np.exp(2 * L) - 1):>8.2f}%"
              f"{100 * ez:>9.1f}%{100 * er:>8.1f}%{Tf:>8.3f}")
    with open(os.path.join(a.out, "table2_synthetic.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows([{k: (f"{v:.6g}" if isinstance(v, float) else v)
                      for k, v in r.items()} for r in rows])

    if a.misspec:
        print("\nTable 3: loss ratio of an indicator designed from a misspecified"
              " model (40 draws per level) ...", flush=True)
        tab = S.misspec_table()
        with open(os.path.join(a.out, "table3_misspec.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["omega", "median", "p90", "worst"])
            for om, (med, p90, wst) in tab.items():
                w.writerow([om, f"{med:.4g}", f"{p90:.4g}", f"{wst:.4g}"])
                print(f"  omega {om:<5} median {med:6.3g}  90th {p90:6.3g}  worst {wst:6.3g}")
    print(f"\nwrote results to {a.out}/")


if __name__ == "__main__":
    main()
