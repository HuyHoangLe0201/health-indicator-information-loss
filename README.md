# Information loss of fixed linear health indicators

Reproducibility repository for the manuscript

> **Information loss of fixed linear health indicators under
> competing degradation mechanisms and variable operating conditions**
> Huy Hoang Le, Kim-Anh Nguyen
> Submitted to *Reliability Engineering & System Safety*

## Quick start

```bash
pip install -r requirements.txt
python verify_numbers.py
```

That needs no data and no LaTeX. It recomputes every number the manuscript
states that can be checked without a fleet, and compares each against the value
written in `theory.tex` — not against a stored expectation, but against the
text itself, so the paper and the code cannot drift apart quietly. It ends with
`ALL CHECKS PASS` or names the rows that failed.

For all 421 checks you need the fleets; see [DATA.md](DATA.md).

```bash
export HI_DATA=/path/to/data
python verify_numbers.py --full        # about an hour
```

## What is here

| | |
|---|---|
| `theory.tex`, `theory.pdf` | the manuscript, 28 pages |
| `supplement.tex`, `supplement.pdf` | Supplementary Sections S1-S4: the Section 6 protocols and those of the Section 4 case study, the loss over a record and the best fixed indicator, the numerical checks, and the conditions for zero loss with further results on the synthetic system |
| `sub.tex`, `sub.pdf` | the same source set in two columns, which shows what overflows a narrow measure |
| `Ref.bib`, `theory.bbl` | bibliography |
| `fig1.pdf`, `fig2.pdf`, `fig4.pdf` | the three plotted figures: Figure 4 of the paper, Figures S1 and S2 of the supplement |
| `make_fig1.py`, `make_fig2.py`, `make_fig4.py` | the scripts that draw them |
| `verify_numbers.py` | the 421 checks |
| `rho_sweep.py`, `rho_sweep.json` | the random-system sweep of Supplementary Section S4, and what it produced |
| `downstream_sweep.py`, `downstream_sweep.json` | the downstream-estimator test of Supplementary Section S4, repeated on 120 random systems, and what it produced |
| `dinkelbach_grids.py`, `dinkelbach_grids_NU41.json`, `dinkelbach_grids_NU81.json`, `dinkelbach_grids_NU161.json` | the fractional optimum of Section 5.2 on state grids of 25 to 137 points per axis for three control grids, and what it produced |
| `verified_run.log` | the output of the run that checked the shipped PDF |
| `make_mechfigs.py` | writes the TikZ code of Figures 2 and 3 into `theory.tex`, from computed geometry, and asserts what each panel claims |
| `make_sub.py` | renders `sub.tex` from `theory.tex` |
| `check_fonts.py` | Type 3 and embedding check on every PDF |
| `LICENSE`, `CITATION.cff` | licence, and how to cite this repository |

Figures 1 to 3 are TikZ and live inside `theory.tex`; `make_mechfigs.py`
writes Figures 2 and 3.

The bundle sent to the journal -- cover letter, highlights and the
declarations -- is deliberately not published here. The cover letter is
correspondence with an editor, and everything else in the bundle duplicates
the files listed above. Nothing needed to reproduce a number in the paper is
missing.

### The figure files are not numbered like the figures

This trips people up, so it is worth saying plainly:

| file | prints as |
|---|---|
| `fig2.pdf` | **Figure 4** |
| `fig1.pdf` | **Figure S2** |
| `fig4.pdf` | **Figure S1** |

The names are the order the scripts were written, not the order the floats
appear, and two of them now sit in the supplement. `theory.aux` and `supplement.aux` are the authority; do not infer a figure number from a
filename. There is no `fig3.pdf` here — it belongs to an earlier, longer
version of the paper and `theory.tex` does not include it.

## Rebuilding

```bash
python make_fig1.py && python make_fig2.py && python make_fig4.py
python make_mechfigs.py
python make_sub.py
pdflatex theory.tex && bibtex theory
pdflatex theory.tex && pdflatex theory.tex
pdflatex supplement.tex && bibtex supplement
pdflatex supplement.tex && pdflatex supplement.tex
python check_fonts.py
```

`make_fig2.py` needs the Severson archive for panel (b). The other two need no
data. `bibtex` must run between the first and second LaTeX pass, or every
citation resolves to `[?]`; skipping it once put an unresolved citation into
a shipped PDF, and the build reported clean because the warning comes from
natbib rather than from LaTeX itself. The build should end with **zero
overfull boxes, no warnings from any package, and no `[?]` in the PDF**.

The supplement is built after the manuscript because it reads
`theory.aux` through the `xr` package, so its references to equations
and sections of the paper cannot go stale. The manuscript does NOT
reference the supplement that way: the journal compiles `theory.tex`
without the supplement's `.aux` and would print `??`. It writes
"Supplementary Section S1" in plain text instead, and
`verify_numbers.py` checks each such pointer against the supplement's
section list and against the topic it should point at.

Publisher preflight rejects Type 3 fonts and matplotlib emits them by default,
so each figure script sets `pdf.fonttype = 42`. `check_fonts.py` verifies the
result rather than trusting the setting.

## What the checks actually check

`verify_numbers.py` reads `theory.tex`, pulls each claimed number out of the
prose with a pattern, recomputes it, and compares. It also checks things that
are not numbers: that `sub.tex` is current with respect to `theory.tex`, that
every figure panel is cited, that the discussion's promised readings match the
ones given, that no sentence is repeated, and that announced counts match
listed ones.

Two rows are worth pointing at, because they guard a distinction rather than a
value:

- **`S2 record-level identity`** checks the closed form for the information a
  fixed indicator forfeits over a whole record against a direct
  mutual-information computation. It agrees to 6.1e-16.
- **`record vs time-averaged loss differ`** asserts that the record loss and
  the time-averaged instantaneous loss are *far apart* (0.31 relative). The
  manuscript optimises the second and states the first; if this row ever
  collapsed towards zero the distinction the text draws would have quietly
  gone.
- **`Table S2: l-bar within 6% of 1/2 rms^2`** reads two columns of Supplementary Table S2 as printed and checks a bound that ties them, independently of the code that
  filled them. For many revisions the column headed "rms" held a standard
  deviation, and every row matched because the gate compared it with the same
  computation; a reviewer caught it with this bound, which the standard
  deviation misses by a factor of four.

## What a passing run looks like

`verified_run.log` is the real output of `--full` against the files in
this repository: 421 rows, each with the value the manuscript states
beside the value recomputed from the model, and `ALL CHECKS PASS` at the
end. Compare your own run against it line by line.

It opens and closes with the md5 of every file the checks read: the
manuscript, the supplement, `verify_numbers.py`, both sweep records, the
grid script with its three records, and `fig4.pdf` with its script. That
bracket is the point: it says the checks and
the manuscript were the same files throughout, so a run that passed against an edited copy cannot
be presented as a run that passed against this one. `theory.tex` here is
`f4d11b6c64da`, which is the hash the log records.

The row count depends on the data you have. Without the fleets a subset
runs and the total is smaller; 421 is the number with all six present.

## Reproducing on another machine

Every path resolves relative to the script or to `$HI_DATA`; nothing points
into the authors' working directory. The three figure scripts were re-run in a
clean copy of this repo and produce PDFs whose page content streams and text
spans are identical to the ones shipped here.

## Known and open, as of submission

These are in the manuscript's own words and are not defects in the code:

- The failure threshold is a named parameter `x_f` of the model, and the synthetic
  system of Section 5.2 uses `x_f = 0.9`. It is worth saying why the code calls
  it `xfail` while `w = 0.9` also appears: `w` is the signal-to-noise weight of
  Eq. (7), an unrelated parameter that happens to carry the same value. The
  computed constants do depend on the threshold — at `x_f = 1.0` the peak floor
  is 5.41° rather than 4.52° and the myopic policy's gain at the steering
  indicator is 58× rather than 68× — so
  do not compare numbers across thresholds. Supplementary Section S4 says this, and two gate
  rows check it via `threshold_sensitivity()`, which moves `XFAIL` and restores
  it in a `finally`: leaving it moved would send every later check against the
  wrong threshold, and they would all still pass.

  The lifetime change is deliberately not quoted, here or in the paper, because
  it depends on the policy: about +7 % at a constant input of 1.5 and about
  +13 % under the myopic policy. A single figure would be meaningless without
  naming which.
- The fractional optimum of Section 5.2 is bracketed, not certified: the
  myopic policy bounds it from above, and the extrapolation of
  `dinkelbach_grids_NU*.json` puts it within about 2 % below. The lower end
  rests on the extrapolation, which the gate fits to the finest four grids
  and, for its spread, to the four next-finest.
- The actuation premise is untested. The measured fleets never vary the
  operating point within a unit, and the two turbofan simulations impose
  their damage independently of it: C-MAPSS as a function of time, and in
  N-CMAPSS the health parameters shipped with the data are constant within
  every flight, which the gate reads from the files.
- The separable form of Supplementary Section S4, which makes the operator's authority
  computable offline, is rejected on the highest-fidelity turbofan data
  jointly with the pairing of channels with mechanisms (Section 6); the
  rest of the paper does not depend on it.

## Requirements

Python 3.10+, and `requirements.txt`. `h5py` is needed only for the N-CMAPSS
files, `PyMuPDF` only for `check_fonts.py`. LaTeX is needed only to rebuild the
PDFs; the `article` class and the `unsrtnat` style come with any TeX
distribution.

## Licence and reuse

The code is under the MIT licence; see `LICENSE`. That covers
`verify_numbers.py`, the three figure scripts, `make_sub.py` and
`check_fonts.py`.

It does not cover the manuscript. `theory.tex`, `theory.pdf`, `sub.tex`,
`sub.pdf` and the figure PDFs are included because the checks read the
paper's own text rather than a stored copy of its numbers, so the code is
not runnable without them. They remain the authors' copyright, and once the
paper is published the publisher's terms apply to the typeset version.

The six fleets are not redistributed here; each is obtained from its own
source under its own terms. See [DATA.md](DATA.md).

## Citing this repository

`CITATION.cff` carries the metadata, and GitHub renders it under *Cite this
repository*. If you are citing a specific state of the code rather than the
paper, cite the archived release rather than the branch, so that the version
you ran is the version a reader gets.

## Contact

Kim-Anh Nguyen, corresponding author — nkanh@dut.udn.vn
