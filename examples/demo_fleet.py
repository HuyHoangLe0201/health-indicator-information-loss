"""A small artificial run-to-failure fleet, so the examples and tests need
no download. Three channels that degrade at different rates, like a battery:
a capacity that fades slowly and then faster, a resistance that grows late,
and a temperature that follows the resistance. Units differ in life length,
so they share one path in normalised life, as the procedure assumes.
"""
import csv

import numpy as np

CHANNELS = ("capacity", "resistance", "temperature")


def demo_fleet(n_units=30, seed=0):
    """List of (n, 3) arrays, one per unit, each from new to failure."""
    rng = np.random.default_rng(seed)
    units = []
    for _ in range(n_units):
        n = int(rng.uniform(450, 900))
        tau = np.arange(n) / (n - 1.0)
        cap = 1.10 - 0.10 * tau - 0.12 * tau ** 4 + rng.normal(0, 0.002, n)
        res = 0.020 + 0.004 * tau + 0.020 * tau ** 3 + rng.normal(0, 0.0004, n)
        tmp = 31.0 + 1.5 * tau ** 2 + rng.normal(0, 0.15, n)
        units.append(np.column_stack([cap, res, tmp]))
    return units


def write_csv(path, units, channels=CHANNELS):
    """Long format: one row per unit and cycle."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit", "cycle", *channels])
        for i, Y in enumerate(units):
            for k, row in enumerate(Y):
                w.writerow([i, k, *(f"{x:.6g}" for x in row)])
