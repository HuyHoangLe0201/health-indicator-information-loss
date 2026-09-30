#!/usr/bin/env python3
"""Section 6 of the paper: what six public run-to-failure fleets establish
(Table 4). Each fleet whose data are missing is skipped, and the skip is
printed; see DATA.md for the files and where to obtain them.

    python reproduce/public_fleets.py [--data DIR]      # C-MAPSS and N-CMAPSS
                                                        # take tens of minutes
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hiloss import datasets, fleets as F  # noqa: E402


def attempt(label, fn):
    try:
        out = fn()
        if out is None:
            raise FileNotFoundError("data not found")
        return out
    except Exception as exc:                         # report, never pass silently
        print(f"  SKIPPED {label}: {exc}")
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", help="directory holding the public datasets")
    a = ap.parse_args()
    if a.data:
        datasets.set_data_dir(a.data)
    print(f"data directory: {datasets.data_dir()}")

    print("\nRotation, Severson cells (window 301)")
    st = attempt("Severson", lambda: F.severson_stats(301))
    if st:
        print(f"  {st['n']} cells: arc {st['travel']:.1f} deg, first-to-last "
              f"{st['e2e']:.0f} deg, all weights positive on "
              f"{100 * st['allpos']:.0f}% of the path")

    print("\nActuation, C-MAPSS (between- minus within-regime angle, window 7)")
    for fd in ("FD004", "FD002"):
        c = attempt(fd, lambda fd=fd: F.cmapss_stats(f"train_{fd}.txt"))
        if c:
            m, npos, n = c["real"][7]
            ctrl = max(v[0] for v in c["ctrl"].values())
            print(f"  {fd}: {c['nregime']} regimes, B - W = {m:.1f} deg, positive"
                  f" for {npos} of {n} engines; surrogate regimes |B - W| <= {ctrl:.2f} deg")

    print("\nSeparability with pairing, N-CMAPSS (interaction over the"
          " operating-point main effect)")
    r2 = attempt("N-CMAPSS DS02", lambda: F.ncmapss_sep())
    if r2:
        print(f"  DS02: {r2[3]:.2f} (interaction {r2[2]:.1f} times its noise floor,"
              f" {r2[4]} units)")
        sw = attempt("N-CMAPSS sweep", F.ncmapss_sweep)
        if sw:
            print(f"  over the clusterings and stagings tried: at least {sw[2]:.2f}")
    r1 = attempt("N-CMAPSS DS01",
                 lambda: F.ncmapss_sep(units=(1, 2, 3, 4, 5, 6), key="ncmapss01"))
    if r1:
        print(f"  DS01: {r1[3]:.2f} (interaction {r1[2]:.1f} times its noise floor)")
    for key in ("ncmapss", "ncmapss01"):
        h = attempt(f"{key} health parameters",
                    lambda key=key: F.ncmapss_health_constant(key))
        if h:
            print(f"  {key}: health parameters constant within every one of"
                  f" {h[1]} flights: {h[0]}")

    # Table 4 reports only the design of these three fleets: each operating
    # condition is applied to its own group of units, or there are too few
    # units, so no effect of the operating point can be separated. (Their
    # direction paths are too noisy for an arc to mean anything.)
    print("\nFleets that cannot separate the operating point from the unit")
    x = attempt("XJTU-SY", F.xjtu_stats)
    if x:
        print(f"  XJTU-SY: {x['n']} bearings in {x['ncond']} operating conditions,"
              f" {'/'.join(map(str, x['sizes']))} per condition: the condition is"
              " fixed per bearing")
    p = attempt("PRONOSTIA", F.pronostia_stats)
    if p:
        print(f"  PRONOSTIA: {p['n']} bearings, the condition fixed per bearing")
    n = attempt("NASA", F.nasa_stats)
    if n:
        print(f"  NASA batteries: {n['n']} cells")


if __name__ == "__main__":
    main()
