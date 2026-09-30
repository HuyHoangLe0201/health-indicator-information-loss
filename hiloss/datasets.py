"""Where the public datasets live, and the readers the case study uses.

The directory is $HI_DATA if set, else ./data under the current directory;
set_data_dir() changes it at run time. DATA.md lists every file, its size
and where to obtain it.

    load_severson()        the 129 cells of the case study (Section 4)
    extract_severson(...)  rebuild severson_cells.npz from the three public
                           batch files of Severson et al. (2019)
"""
import os

import numpy as np

FILES = dict(
    severson="severson_cells.npz",
    pronostia="pronostia_cells.npz",
    xjtu="xjtu_features.npz",
    nasa="cv_features_dataset.csv",
    cmapss="CMAPSSData",
    ncmapss="N-CMAPSS_DS02-006.h5",
    ncmapss01="N-CMAPSS_DS01-005.h5",
)
DATA = {}


def set_data_dir(path):
    """Point every dataset path at directory `path`."""
    for k, f in FILES.items():
        DATA[k] = os.path.join(path, f)
    return path


def data_dir():
    return os.path.dirname(DATA["severson"])


set_data_dir(os.environ.get("HI_DATA", os.path.join(os.getcwd(), "data")))

SEVERSON_CHANNELS = ("QDischarge", "IR", "Tavg")


def load_severson(path=None, channels=SEVERSON_CHANNELS, min_cycles=400):
    """The cells of the case study: those with a recorded life, at least
    min_cycles cycles and finite readings on every channel used.

    Returns (units, names): units is a list of (n, len(channels)) arrays,
    one row per cycle from the start of service to end of life.
    """
    path = path or DATA["severson"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Build it with extract_severson() from the "
            "public Severson et al. (2019) batch files, or set HI_DATA; see "
            "DATA.md.")
    Z = np.load(path, allow_pickle=True)
    CH = [str(c) for c in Z["channels"]]
    use = [CH.index(c) for c in channels]
    units, names = [], []
    for k in range(len(Z["data"])):
        if not np.isfinite(Z["lives"][k]):
            continue
        aa = np.asarray(Z["data"][k], dtype=float)
        if len(aa) < min_cycles or not np.isfinite(aa[:, use]).all():
            continue
        units.append(aa[:, use])
        names.append(str(Z["names"][k]))
    return units, names


SEVERSON_BATCHES = ("2017-05-12_batchdata_updated_struct_errorcorrect.mat",
                    "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
                    "2018-04-12_batchdata_updated_struct_errorcorrect.mat")
SEVERSON_ALL_CHANNELS = ("QDischarge", "QCharge", "IR", "Tavg", "Tmax", "Tmin",
                         "chargetime")


def extract_severson(raw_dir, out=None, verbose=True):
    """Per-cycle summaries of the Severson et al. (2019) fast-charging cells,
    from the three public batch files in raw_dir (only their small 'summary'
    groups are read). Keeps cells with at least 100 cycles, finite readings,
    a plausible 1.1 Ah LFP capacity and genuine fade (end capacity below 95%
    of the early plateau); the first and last cycle of each cell are dropped
    as known artefacts. Writes severson_cells.npz (default: into the data
    directory) and returns its path.
    """
    import h5py
    out = out or DATA["severson"]
    cells, names, lives = [], [], []
    for bi, fn in enumerate(SEVERSON_BATCHES):
        with h5py.File(os.path.join(raw_dir, fn), "r") as f:
            b = f["batch"]
            n = b["summary"].shape[0]
            for i in range(n):
                s = f[b["summary"][i, 0]]
                try:
                    A = np.stack([np.array(s[c]).ravel().astype(float)
                                  for c in SEVERSON_ALL_CHANNELS], 1)
                    life = float(np.array(f[b["cycle_life"][i, 0]]).ravel()[0])
                except Exception:
                    continue
                A = A[1:-1]
                if len(A) < 100 or not np.isfinite(A).all():
                    continue
                q = A[:, 0]
                if q[0] <= 0 or q.max() > 1.5 or q.min() < 0.5:
                    continue
                if q[-1] > 0.95 * np.median(q[:20]):
                    continue
                cells.append(A)
                names.append(f"b{bi + 1}c{i}")
                lives.append(life)
        if verbose:
            print(f"{fn}: {n} cells read, {len(cells)} kept so far")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, data=np.array(cells, dtype=object),
                        names=np.array(names), lives=np.array(lives),
                        channels=np.array(SEVERSON_ALL_CHANNELS))
    return out
