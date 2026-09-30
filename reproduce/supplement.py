#!/usr/bin/env python3
"""The supplementary results on the synthetic system (Supplementary Sections
S2 to S4). Needs no data.

    python reproduce/supplement.py            # the quick ones (~10 min)
    python reproduce/supplement.py --slow     # + Table S3, the one-input
                                              #   families and the x_f = 1 rerun

The sweeps and the computed fractional optimum have their own scripts and
records in reproduce/supplement/ (rho_sweep.py, downstream_sweep.py,
dinkelbach_grids.py; hours each), whose recorded output is included.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hiloss import synthetic as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--slow", action="store_true")
    a = ap.parse_args()

    wid, gap = S.record_identity()
    print(f"S2  loss over a record: closed form vs Fisher sums, largest "
          f"discrepancy {wid:.1e}; largest gap to the time-averaged loss {gap:.2f}")

    print("\nS4  Table S2: loss at an equal record Fisher score")
    print(f"    {'u':>4}{'delta':>8}{'I_rec':>9}{'I_z':>9}{'rms beta (deg)':>16}{'loss':>10}")
    for u, dl, F, Iz, rms, L in S.levelset_table():
        print(f"    {u:>4.1f}{dl:>8.3f}{F:>9.4g}{Iz:>9.4g}{rms:>16.2f}{L:>10.3g}")

    rch, x2f, off, x2e, x1e, before = S.segment_counterexample()
    print(f"\nS4  segment counterexample: reachable along the trajectory {bool(rch)};"
          f" the aligning input leaves U at x2 = {x2e:.2f} (x1 = {x1e:.2f}),"
          f" before failure {bool(before)}")
    t = S.two_input_floor()
    print(f"S4  two inputs: {t['inside']} of {t['n']} points of the segment inside U;"
          f" worst closed-loop misalignment {t['res']:.1e} deg;"
          f" largest distance from the segment {t['dist']:.1e}")
    fmax, fmin, mid = S.floor_shape()
    print(f"S4  one input: the floor to q* peaks at tau = {fmax:.2f} and dips at"
          f" tau = {fmin:.2f}")
    w, sp, rr = S.downstream_rul()
    print(f"S4  downstream estimator: identity error {w:.1e}; Spearman(loss,"
          f" median error) = {sp:.2f}; worst/best median error {rr:.2f}")

    if a.slow:
        ang, ramp, cb, cw = S.one_input_family()
        print(f"\nS3  single-input family d = 2..8: misalignment <= {ang:.1e} deg,"
              f" loss <= {ramp:.1e}; best constant input {cb:.1e} to {cw:.1e}")
        my, co = S.family_mistuned()
        print(f"S3  mistuned by 1%: myopic {my:.2g} against best constant {co:.2g}")
        peak, gain, dlife = S.threshold_sensitivity(1.0)
        print(f"S4  at x_f = 1: peak floor {peak:.1f} deg, myopic gain {gain:.0f},"
              f" lifetime {dlife:+.1f} %")
        print("\nS4  Table S3: authority and drift in a one-parameter family")
        for Ev in S.REGIME:
            arc, rho, L, g1, g2 = S.regime_row(Ev)
            print(f"    E = {np.round(Ev, 3)}: arc {arc:.1f} deg, rho {rho:.3f},"
                  f" loss {L:.3g}, gains {g1:.1f} and {g2:.1f}")


if __name__ == "__main__":
    main()
