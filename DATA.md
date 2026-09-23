# Data

No dataset is included here. The six fleets are public, but they total about
8.5 GB, and redistributing features derived from someone else's dataset is the
authors' licensing decision rather than something to bundle by default. Every
fleet below is obtainable from its original source.

The code looks for data in `$RESS_DATA`, and falls back to `./data` beside the
scripts. Anything missing is **skipped, and the skip is printed** — the run
does not fail silently, but it also does not check what it could not read.

```bash
export RESS_DATA=/path/to/data          # Windows: set RESS_DATA=C:\path\to\data
python verify_numbers.py --full
```

## Expected layout

`$RESS_DATA` should contain these names exactly:

| name under `$RESS_DATA` | size | source |
|---|---|---|
| `severson_cells.npz` | 5 MB | derived from Severson et al. 2019, *Data-driven prediction of battery cycle life before capacity degradation* |
| `xjtu_features.npz` | 1 MB | derived from the XJTU-SY bearing fleet, Wang et al. 2020 |
| `pronostia_cells.npz` | 3 MB | derived from PRONOSTIA / FEMTO, Nectoux et al. 2012 |
| `cv_features_dataset.csv` | 0.1 MB | derived from the NASA battery set, Saha & Goebel 2007 |
| `CMAPSSData/` | 45 MB | C-MAPSS, Saxena et al. 2008 — the original archive, unpacked |
| `N-CMAPSS_DS02-006.h5` | 2.4 GB | N-CMAPSS, Arias Chao et al. 2021 |
| `N-CMAPSS_DS01-005.h5` | 2.9 GB | N-CMAPSS, Arias Chao et al. 2021 |

The four small files are feature tables the authors derived from the raw
fleets; the three large ones are the published archives unchanged. The `.npz`
and `.csv` derivations are not reconstructed by this repo — request them from
the corresponding author, or rebuild them from the raw fleets.

## What each unlocks

| flag | needs | checks |
|---|---|---|
| *(none)* | nothing | the manuscript-text checks and the closed-form identities |
| `--regime` | nothing | the regime table, the one-input family, the reachability floors |
| `--fleet` | the fleets above | rotation, actuation, the separability test |
| `--full` | all of it | all 111 |

`python verify_numbers.py` with no flags needs no data at all and is the
fastest way to confirm the repo is intact.

`make_fig2.py` needs `severson_cells.npz` for panel (c); `make_fig1.py` and
`make_fig4.py` need no data.

## You do not need PHM 2012

`verify_numbers.py` defines a `phm2012()` reader and a `DATA["phm2012"]`
path, and **nothing calls it**. The call site went when Section 7 was
compressed; the function was kept so the analysis can be restored. Do not
download the 3.2 GB PHM 2012 challenge archive to reproduce this paper — it
contributes no check.

## A note on what the fleets can and cannot settle

Section 7 of the manuscript is explicit that only the two turbofan fleets vary
the operating point within a unit, so only they can separate an operating-point
effect from a unit effect. The rotation premise is testable on all six; the
actuation premise is confirmed on simulation and remains open on measured
hardware. Reproducing the numbers here will not change that, and the paper does
not claim otherwise.
