"""Type 3 and unembedded-font check, for the files this repo builds.

Publisher preflight rejects Type 3 fonts, and matplotlib emits them by
default, so every figure script sets pdf.fonttype = 42. This verifies the
result rather than trusting the setting.

Paths are relative to this file, so the repo runs from anywhere. The working
copy of this script hardcoded an absolute path and also checked main.pdf and
fig3.pdf, which belong to the archived extended version and are not here.
"""
import os
import sys

import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = ("theory.pdf", "sub.pdf", "fig1.pdf", "fig2.pdf", "fig4.pdf")

bad = 0
for name in TARGETS:
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        print(f"{name}: MISSING (build it first)")
        bad += 1
        continue
    doc = fitz.open(path)
    seen = {}
    for pno in range(doc.page_count):
        for f in doc.get_page_fonts(pno):
            seen[f[3]] = (f[1], f[2])
    t3 = sum(1 for v in seen.values() if v[0] == "n/a" or "Type3" in v[1])
    ne = sum(1 for v in seen.values() if v[0] == "n/a")
    print(f"{name}: {len(seen)} font(s), Type3={t3}  not-embedded={ne}")
    bad += t3 + ne

print("\nFAIL" if bad else "\nall clean")
sys.exit(1 if bad else 0)
