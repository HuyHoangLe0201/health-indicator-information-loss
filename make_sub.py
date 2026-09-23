"""Generate sub.tex from theory.tex.

sub.tex was once a hand-maintained second copy of the whole manuscript,
differing from the source only in its \\documentclass line, and the two had
drifted apart in 72 places, so sub.pdf was being built from a manuscript
missing every correction made to the real one. It is now generated: this
writes sub.tex from theory.tex with the one line swapped, and
verify_numbers.py checks that the file on disk is what this would produce.

theory.tex sets the single-spaced preprint layout; sub.tex sets the journal's
final two-column one. Retargeted from main.tex when theory.tex replaced it as
the manuscript; main.tex remains in the directory as the extended version.
"""
import io
import sys

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_DATA = _os.environ.get("RESS_DATA", _os.path.join(_HERE, "data"))


D = _HERE + _os.sep

MAIN_CLASS = "\\documentclass[preprint,12pt]{elsarticle}"
SUB_CLASS = "\\documentclass[final,5p,times,twocolumn]{elsarticle}"


def render():
    """The exact text sub.tex should contain, given theory.tex."""
    t = io.open(D + "theory.tex", encoding="utf-8-sig").read()
    n = t.count(MAIN_CLASS)
    if n != 1:
        raise SystemExit(f"theory.tex: expected 1 documentclass line, found {n}")
    return t.replace(MAIN_CLASS, SUB_CLASS, 1)


if __name__ == "__main__":
    want = render()
    try:
        have = io.open(D + "sub.tex", encoding="utf-8-sig").read()
    except FileNotFoundError:
        have = None
    if "--check" in sys.argv:
        print("sub.tex is current" if have == want
              else "sub.tex DIFFERS from theory.tex")
        sys.exit(0 if have == want else 1)
    io.open(D + "sub.tex", "w", encoding="utf-8", newline="").write(want)
    print("wrote sub.tex from theory.tex"
          if have != want else "sub.tex already current")
