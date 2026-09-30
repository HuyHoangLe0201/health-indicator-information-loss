# Data

No dataset is redistributed here: the fleets are public, and each belongs to
its original authors. The code looks for data in the directory named by
`$HI_DATA`, and otherwise in `./data`; `hiloss.datasets.set_data_dir()`
changes it at run time.

```bash
export HI_DATA=/path/to/data        # Windows: set HI_DATA=C:\path\to\data
```

## What each part needs

| to reproduce | needs |
|---|---|
| `examples/`, `tests/`, the synthetic system (Section 5), the supplement | nothing |
| the case study (Section 4, Table 1, Fig. 4) | `severson_cells.npz` |
| the public-fleet evidence (Section 6, Table 4) | all files below; each missing one is skipped, and the skip is printed |

## Files expected under `$HI_DATA`

| name | size | source |
|---|---|---|
| `severson_cells.npz` | 5 MB | built from Severson et al. (2019), *Data-driven prediction of battery cycle life before capacity degradation*, Nature Energy — see below |
| `CMAPSSData/` | 45 MB | C-MAPSS, Saxena et al. (2008): the original archive, unpacked |
| `N-CMAPSS_DS02-006.h5` | 2.4 GB | N-CMAPSS, Arias Chao et al. (2021), unchanged |
| `N-CMAPSS_DS01-005.h5` | 2.9 GB | N-CMAPSS, Arias Chao et al. (2021), unchanged |
| `xjtu_features.npz` | 1 MB | features derived from the XJTU-SY bearings, Wang et al. (2020) |
| `pronostia_cells.npz` | 3 MB | features derived from PRONOSTIA / FEMTO, Nectoux et al. (2012) |
| `cv_features_dataset.csv` | 0.1 MB | features derived from the NASA batteries, Saha & Goebel (2007) |

### The Severson cells

Download the three batch files that accompany Severson et al. (2019), from the
authors' data portal (data.matr.io):

    2017-05-12_batchdata_updated_struct_errorcorrect.mat
    2017-06-30_batchdata_updated_struct_errorcorrect.mat
    2018-04-12_batchdata_updated_struct_errorcorrect.mat

and build the per-cycle summaries (only the small `summary` groups of the
8 GB files are read):

```bash
python -c "from hiloss.datasets import extract_severson; extract_severson('path/to/batch/files')"
```

This writes `severson_cells.npz` into the data directory. The case study then
keeps the 129 cells with a recorded life, at least 400 cycles and finite
capacity, resistance and temperature (`hiloss.datasets.load_severson`).

### The three derived feature files

`xjtu_features.npz`, `pronostia_cells.npz` and `cv_features_dataset.csv` are
feature tables the authors derived from the raw fleets; they enter only the
qualitative rows of Table 4 (the operating condition is fixed per unit, or
there are too few units). Request them from the corresponding author.

## What the fleets can and cannot settle

Only the two turbofan simulations vary the operating point within a unit, and
neither lets it act on the damage, so the premise that the operating point
moves the informative direction by acting on the degradation remains
untested on every public fleet (Section 6 of the paper). Reproducing the
numbers here does not change that.
