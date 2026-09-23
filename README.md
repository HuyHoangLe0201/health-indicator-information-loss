# Information loss of fixed health indicators as a control problem

Reproducibility repository for the manuscript

> **Information loss of fixed health indicators as a control problem: simplex
> geometry and reachability limits**
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

For all 113 checks you need the fleets; see [DATA.md](DATA.md).

```bash
export RESS_DATA=/path/to/data
python verify_numbers.py --full        # about 30 minutes
```

## What is here

| | |
|---|---|
| `theory.tex`, `theory.pdf` | the manuscript, 40 pages |
| `sub.tex`, `sub.pdf` | the same source in the journal's two-column layout |
| `Ref.bib`, `theory.bbl` | bibliography |
| `fig1.pdf`, `fig2.pdf`, `fig4.pdf` | the three plotted figures |
| `make_fig1.py`, `make_fig2.py`, `make_fig4.py` | the scripts that draw them |
| `verify_numbers.py` | the 113 checks |
| `make_sub.py` | renders `sub.tex` from `theory.tex` |
| `check_fonts.py` | Type 3 and embedding check on every PDF |
| `LICENSE`, `CITATION.cff` | licence, and how to cite this repository |

The other three figures are TikZ and live inside `theory.tex`.

The bundle sent to the journal -- cover letter, highlights and the
declarations -- is deliberately not published here. The cover letter is
correspondence with an editor and names manuscripts still under review
elsewhere; everything else in it duplicates the files listed above. Nothing
needed to reproduce a number in the paper is missing.

### The figure files are not numbered like the figures

This trips people up, so it is worth saying plainly:

| file | prints as |
|---|---|
| `fig2.pdf` | **Figure 2** |
| `fig1.pdf` | **Figure 4** |
| `fig4.pdf` | **Figure 5** |

The names are the order the scripts were written, not the order the floats
appear. `theory.aux` is the authority; do not infer a figure number from a
filename. There is no `fig3.pdf` here — it belongs to an earlier, longer
version of the paper and `theory.tex` does not include it.

## Rebuilding

```bash
python make_fig1.py && python make_fig2.py && python make_fig4.py
python make_sub.py
pdflatex theory.tex && bibtex theory
pdflatex theory.tex && pdflatex theory.tex
python check_fonts.py
```

`make_fig2.py` needs the Severson archive for panel (c). The other two need no
data. `bibtex` must run between the first and second LaTeX pass, or every
citation resolves to `[?]`; skipping it once put an unresolved citation into
a shipped PDF, and the build reported clean because the warning comes from
natbib rather than from LaTeX itself. The build should end with **zero
overfull boxes, no warnings from any package, and no `[?]` in the PDF**.

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

- **`Prop 2.3 record identity`** checks the closed form for the information a
  fixed indicator forfeits over a whole record against a direct
  mutual-information computation. It agrees to 6.1e-16.
- **`record vs time-averaged loss differ`** asserts that the record loss and
  the time-averaged instantaneous loss are *far apart* (0.31 relative). The
  manuscript optimises the second and states the first; if this row ever
  collapsed towards zero the distinction the text draws would have quietly
  gone.

## Reproducing on another machine

Every path resolves relative to the script or to `$RESS_DATA`; nothing points
into the authors' working directory. The three figure scripts were re-run in a
clean copy of this repo and produce PDFs whose page content streams and text
spans are identical to the ones shipped here.

## Known and open, as of submission

These are in the manuscript's own words and are not defects in the code:

- The failure threshold is a named parameter `x_f` of the model, and the test
  plant of Section 6.1 uses `x_f = 0.9`. It is worth saying why the code calls
  it `xfail` while `w = 0.9` also appears: `w` is the signal-to-noise weight of
  Lemma 2.2, an unrelated parameter that happens to carry the same value. The
  computed constants do depend on the threshold — at `x_f = 1.0` the peak floor
  is 5.41° rather than 4.52° and the adaptive gain is 58× rather than 68× — so
  do not compare numbers across thresholds. Section 6.1 says this, and two gate
  rows check it via `threshold_sensitivity()`, which moves `XFAIL` and restores
  it in a `finally`: leaving it moved would send every later check against the
  wrong threshold, and they would all still pass.

  The lifetime change is deliberately not quoted, here or in the paper, because
  it depends on the policy: about +7 % at a constant input of 1.5 and about
  +13 % under the adaptive policy. A single figure would be meaningless without
  naming which.
- The fractional limit in Section 5 is computed, not certified. The manuscript
  says so: "we do not claim a certified converse here".
- The actuation premise is confirmed on simulation and remains open on measured
  hardware, for the reason Section 7 gives — only the turbofan fleets vary the
  operating point within a unit.
- Section 4.2 and Proposition 4.4 hold on a separable subclass that the
  highest-fidelity turbofan data reject. Figure 6 marks which branch of the
  development depends on it; nothing before that subsection does.

## Requirements

Python 3.10+, and `requirements.txt`. `h5py` is needed only for the N-CMAPSS
files, `PyMuPDF` only for `check_fonts.py`. LaTeX is needed only to rebuild the
PDFs; `elsarticle.cls` comes from your TeX distribution.

## Licence and reuse

The code is under the MIT licence; see `LICENSE`. That covers
`verify_numbers.py`, the three figure scripts, `make_sub.py` and
`check_fonts.py`.

It does not cover the manuscript. `theory.tex`, `theory.pdf`, `sub.tex`,
`sub.pdf` and the figure PDFs are included because the checks read the
paper's own text rather than a stored copy of its numbers, so the code is
not runnable without them. They remain the authors' copyright, and once the
paper is published the publisher's terms apply to the typeset version.
`elsarticle-num.bst` is Elsevier's, under the LaTeX Project Public Licence.

The six fleets are not redistributed here; each is obtained from its own
source under its own terms. See [DATA.md](DATA.md).

## Citing this repository

`CITATION.cff` carries the metadata, and GitHub renders it under *Cite this
repository*. If you are citing a specific state of the code rather than the
paper, cite the archived release rather than the branch, so that the version
you ran is the version a reader gets.

## Contact

Kim-Anh Nguyen, corresponding author — nkanh@dut.udn.vn
