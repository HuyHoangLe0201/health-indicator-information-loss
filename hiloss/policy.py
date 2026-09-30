"""Step 5: a predictive replacement policy driven by the remaining-life
estimator, and its long-run cost.

Each unit is inspected every `dn` cycles from `n_first` until its end of
life. At every inspection the leave-one-out estimator predicts the remaining
life, and the unit is replaced at the first inspection at which the
prediction is at most `theta` cycles; a unit not replaced before its end of
life counts as a failure. For a failure-to-preventive cost ratio r the
long-run cost per cycle is

    (r x failures + preventive replacements) / (total cycles in service),

and theta is chosen per signal to minimise it (the same in-sample choice for
every signal, so absolute costs are optimistic but comparisons are fair).
"""
import numpy as np


def inspection_predictions(model, names=("all", "barycentre", "channel:0", "fusion"),
                           n_first=100, dn=25, horizon=1500, back=0.30):
    """{name: [array (inspections, 2) of (cycle, predicted remaining life)]}
    for every unit, leave-one-out."""
    pred = {m: [] for m in names}
    for k, c in enumerate(model.cells):
        sig = model.signals(k)
        insp = np.arange(n_first, c["n"], dn)
        for m in names:
            pred[m].append(np.array(
                [[n0, model.predict_life(k, int(n0), sig, m, back,
                                         int(n0) + horizon) - n0]
                 for n0 in insp]).reshape(-1, 2))
    return pred


def outcomes(P_units, lives, theta):
    """(failed, cycles in service, unused life) per unit for threshold theta."""
    N = len(lives)
    fail = np.zeros(N, bool)
    used = np.zeros(N)
    for k in range(N):
        P = P_units[k]
        hit = np.nonzero(P[:, 1] <= theta)[0] if len(P) else []
        if len(hit):
            used[k] = P[hit[0], 0]
        else:
            fail[k] = True
            used[k] = lives[k]
    return fail, used, np.where(fail, 0.0, lives - used)


def evaluate(pred, lives, thetas=tuple(range(0, 1001, 5)),
             cost_ratios=(5.0, 10.0, 20.0, 50.0), max_fail=0.05):
    """Per signal and cost ratio: minimal cost rate, its threshold, the
    failure fraction and the mean unused life (cycles and fraction of life);
    per signal, the smallest threshold with at most max_fail failures."""
    lives = np.asarray(lives, dtype=float)
    out = {}
    for m, P_units in pred.items():
        table = [outcomes(P_units, lives, th) for th in thetas]
        out[m] = {}
        for r in cost_ratios:
            rates = [(f.sum() * r + (~f).sum()) / u.sum() for f, u, _ in table]
            i = int(np.argmin(rates))
            f, u, un = table[i]
            out[m][r] = dict(rate=float(rates[i]), theta=float(thetas[i]),
                             fail=float(f.mean()), unused=float(un.mean()),
                             unused_frac=float(np.mean(un / lives)))
        for th, (f, u, un) in zip(thetas, table):
            if f.mean() <= max_fail:
                out[m]["matched"] = dict(theta=float(th), fail=float(f.mean()),
                                         unused=float(un.mean()))
                break
    return out
