# hiloss — information loss of fixed linear health indicators

Code for

> H. H. Le and K.-A. Nguyen, *Information loss of fixed linear health
> indicators under competing degradation mechanisms and variable operating
> conditions*, submitted to Reliability Engineering & System Safety.

A health indicator that combines monitoring channels with **fixed weights**
loses information about remaining useful life when degradation mechanisms
progress at different rates, because the combination that carries that
information — the *informative direction* — turns as the component ages.
The paper shows that the loss depends on one angle, and that the arc
through which the informative direction turns over life bounds it before
any indicator is designed:

    error-variance inflation  e^{2l}  <=  1 / cos^2(arc / 2)
    safety-margin widening    e^{l}   <=  1 / cos(arc / 2)

This repository lets you

1. **apply the method to your own run-to-failure data** (`examples/your_fleet.py`):
   estimate the informative direction and its arc, get the bound, design the
   fixed indicator, and test it with a leave-one-out remaining-life estimator
   and a predictive replacement policy;
2. **use the pieces in your own code** (the `hiloss` package);
3. **reproduce every result of the paper** (`reproduce/`).

## Install

Python 3.10 or later.

```bash
git clone https://github.com/HuyHoangLe0201/health-indicator-information-loss
cd health-indicator-information-loss
pip install -e .            # numpy, scipy; add [data,figures] for h5py, pandas, matplotlib
python -m unittest discover -s tests      # about 30 s, no data needed
```

## Quick start on your own data

Put your fleet in a CSV, one row per unit and cycle, every unit observed from
the start of service to failure:

    unit,cycle,capacity,resistance,temperature
    0,0,1.0998,0.0201,31.02
    ...

```bash
python examples/your_fleet.py --demo demo.csv     # writes an example file in this format
python examples/your_fleet.py demo.csv            # steps 1-4
python examples/your_fleet.py mydata.csv --channels capacity resistance --policy
```

It prints the arc of the observed direction, the bound on the error-variance
inflation, the fixed indicator (in whitened and in raw units), the angle of
each single channel to the path, and the leave-one-out mean squared error of
the remaining-life estimate from the fixed indicator, each channel and a
fusion indicator, relative to using all channels.

**Reading the output.** If the bound is small compared with the accuracy your
maintenance decision needs, a fixed indicator is enough; indicators whose
weights change with age become worthwhile as the arc approaches a right
angle, where the bound allows the error variance to double. Estimate the arc
under the operating conditions the fleet will meet in service: the observed
direction can turn with the operating regime even when the damage does not
(Section 6 of the paper).

**Assumptions.** The bound needs units that differ mainly in age along a
common degradation path (the paper's assumption A1) and a record that can
be linearised about that path (A2). The leave-one-out test is how you check,
on your data, whether that approximation is good enough.

## Using the package

```python
import numpy as np
from hiloss import observed_direction, loss, arc_bound
from hiloss.rul import FleetModel, leave_one_out, mse_ratios
from hiloss.policy import inspection_predictions, evaluate

units = [...]                        # list of (n_cycles, n_channels) arrays, new -> failure
fd = observed_direction(units, window=301, grid=np.linspace(0.15, 0.90, 40))
fd.arc_deg, fd.bound(), fd.indicator, fd.max_angle_deg()
fd.step3()      # step 3 with its check: the barycentre if it lies within half
                # the arc of every point of the path, else the arc midpoint

model = FleetModel(units)            # step 4: leave-one-out RUL
per_age, pooled = mse_ratios(leave_one_out(model))
pred = inspection_predictions(model) # step 5: replacement policy
costs = evaluate(pred, model.lives)

loss(psi=np.radians(20), gamma=100)  # the closed forms of Sections 2-3
```

| module | what it holds |
|---|---|
| `hiloss.core` | loss (7), error-variance and safety-margin inflation (8), arc length, arc bound (12), barycentre, Karcher mean |
| `hiloss.direction` | steps 1–3: observed direction, arc, bound, fixed indicator, rotation curve and its bootstrap band |
| `hiloss.rul` | step 4: leave-one-out similarity-based remaining-life estimator on all channels or any fixed indicator |
| `hiloss.policy` | step 5: predictive replacement policy and its long-run cost |
| `hiloss.synthetic` | the synthetic system of Section 5: myopic steering, constant designs, Tables 2 and 3, supplementary analyses |
| `hiloss.fleets` | Section 6: the six public fleets (rotation, C-MAPSS contrast, N-CMAPSS separability) |
| `hiloss.datasets` | where the data live; the Severson reader and extractor |

## Reproducing the paper

| paper | command | data | time |
|---|---|---|---|
| Section 4, Table 1, Fig. 4(b) | `python reproduce/severson_case_study.py` | Severson cells | ~15 min |
| Supplementary S1 (sensitivity) | `python reproduce/severson_case_study.py --sensitivity` | Severson cells | +20 min |
| Section 5.2, Table 2 | `python reproduce/synthetic_system.py` | none | ~10 min |
| Table 3 | `python reproduce/synthetic_system.py --misspec` | none | ~1 h |
| Section 6, Table 4 | `python reproduce/public_fleets.py` | six public fleets | ~1 h |
| Supplementary S2–S4 | `python reproduce/supplement.py [--slow]` | none | 10 min / hours |
| S4 sweeps, fractional optimum | `reproduce/supplement/rho_sweep.py`, `downstream_sweep.py`, `dinkelbach_grids.py`; `records.py` summarises the recorded runs | none | hours each |
| Fig. 4 and Figs. S1–S2 | `reproduce/figures/make_fig2.py`, `make_fig4.py`, `make_fig1.py` | Severson (Fig. 4 only) | minutes |

Scripts write CSV files to `results/`; the files committed there are the
output of these scripts, for comparison with your run. The library was
checked against the code that produced the paper's numbers: on the same data
every quantity of the case study agrees to machine precision, and the
Severson file rebuilt from the public batch files is identical to the one
the paper used. Figures 1–3 of the paper are schematic diagrams drawn in
LaTeX and are not generated here.

### Differences from the printed numbers

The code prints unrounded values. A few printed values are not the
correctly rounded computed ones:

| where | printed | computed |
|---|---|---|
| Section 5.2: steering factor at q\* against the best constant input | 68 | 68.59 |
| Supplementary S4, sensitivity to x_f: that factor at x_f = 0.9 and at x_f = 1 | 68, 58 | 68.59, 58.63 |
| Supplementary S4, sensitivity to x_f: peak floor at x_f = 1 | 5.41° | 5.40° |
| Supplementary Table S3, row 1, gain (also quoted twice in S4) | 68 | 68.63 |

None changes a conclusion. The headline factor of the abstract, 47 (computed
47.46), is correct.

## Data

No dataset is redistributed here. The Severson battery cells, which the case
study uses, are rebuilt from the public batch files with one call; the other
fleets are needed only for Section 6. See [DATA.md](DATA.md).

## Citation

See [CITATION.cff](CITATION.cff). Licence: MIT.
