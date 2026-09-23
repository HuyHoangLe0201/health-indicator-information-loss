#!/usr/bin/env python3
"""
Single source of truth for every number quoted in Sections V-VI of the letter.

Motivation. Two audit rounds found the same failure mode: a quantity was
computed at one configuration, the configuration later changed, and the
manuscript was not re-derived. Round one: q* was replaced by the outer
minimisation but Table I, the reachability range and the lifetime factor kept
their old values. Round two: the headline ratio had been computed at a q that
was never the q* printed in the paper, and it also drifted with the control
grid.

This script fixes one CONFIG, recomputes everything from it, then reads
main.tex and compares. Any drift between code and manuscript is reported.

    python verify_numbers.py            # Sections V-VI  (fast)
    python verify_numbers.py --fleet    # + Section 7, five fleets (slow)
    python verify_numbers.py --regime   # + Table II   (re-optimises q, slow)
    python verify_numbers.py --full     # everything, incl. the Dinkelbach limit

Exit status is non-zero if any check fails, so it can gate a build.
"""
import argparse
import os
import io
import re
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

# ----------------------------------------------------------------- CONFIG --
CFG = dict(
    a=np.array([0.0, 0.3, -0.2]),          # log pre-exponentials
    E=np.array([0.6, 1.1, 0.8]),           # activation energies
    c=np.array([-0.5, 2.0, 0.8]),          # shape coefficients h_i = 1 + c_i x
    U=(0.0, 3.0),                          # input box
    w=0.9,                                 # SNR weight gamma/(1+gamma)
    xfail=0.9, x0=5e-3, tmax=200.0,
    NU=41,                                 # control grid used throughout
    Ngrid=(25, 29, 35),                    # HJB state grids
    qstar=np.array([0.466109, 0.770508, 0.434809]),
    gain=1.0e3,                            # Var[R]/sigma^2 for the SNR profile
)
TEX = "theory.tex"
A, E, HC = CFG["a"], CFG["E"], CFG["c"]
WS, XFAIL, X0, TMAX = CFG["w"], CFG["xfail"], CFG["x0"], CFG["tmax"]
UMIN, UMAX = CFG["U"]
QS = CFG["qstar"] / np.linalg.norm(CFG["qstar"])


def ell(b, d2=None):
    """Exact loss of Lemma 2.2.

    w := gamma/(1+gamma) with gamma = ||d||^2 Var[R]. Pass d2 = ||d||^2 to
    evaluate at the plant's own signal-to-noise ratio, which is what the
    definition asks for and what Table I does.

    d2=None falls back to the declared constant CFG["w"]. That branch is for
    the generic families of Section 5.1 only: those plants carry no Var[R],
    so there is no gamma to form, and their claims are bounds with orders of
    magnitude of headroom.
    """
    if d2 is None:
        w = WS
    else:
        g = CFG["gain"] * np.asarray(d2, dtype=float)
        w = g / (1.0 + g)
    return -0.5 * np.log(np.maximum(1 - w * np.sin(b) ** 2, 1e-300))


def grid(NU):
    US = np.linspace(UMIN, UMAX, NU)
    return US, np.exp(A[None, :] - US[:, None] * E[None, :])


def traj(law, n=2000, rtol=1e-10):
    def rhs(t, x):
        return np.exp(A - law(x) * E) * (1.0 + HC * x)

    def ev(t, x):
        return x.max() - XFAIL
    ev.terminal, ev.direction = True, 1
    s = solve_ivp(rhs, [0, TMAX], np.full(3, X0), events=ev, rtol=rtol,
                  atol=rtol * 1e-2, dense_output=True)
    if s.t_events[0].size == 0:
        return None, None
    tg = np.linspace(0, s.t[-1], n)
    return s.sol(tg).T, float(s.t[-1])


def greedy(q, NU=None, fast=False):
    """adaptive (myopic-alignment) policy: value and lifetime.

    fast=True loosens the quadrature and the integrator for use inside the
    Nelder-Mead search of Table II; the winner is always re-evaluated at full
    accuracy before anything is reported.
    """
    q = np.abs(q) / np.linalg.norm(q)
    US, PHI = grid(NU or CFG["NU"])
    n_s, rt = (400, 1e-7) if fast else (2000, 1e-10)

    def bb(x):
        D = PHI * (1.0 + HC[None, :] * x[None, :])
        nr = np.linalg.norm(D, axis=1, keepdims=True)
        D = D / nr
        b = np.arccos(np.clip(D @ q, -1, 1))
        j = int(np.argmin(b))
        return US[j], float(b[j]), float(nr[j, 0] ** 2)

    X, Tf = traj(lambda x: bb(x)[0], n_s, rt)
    if X is None:
        return np.inf, np.inf, None
    v = np.array([bb(x)[1:] for x in X])        # columns: beta, ||d||^2
    return float(np.mean(ell(v[:, 0], v[:, 1]))), Tf, X


def best_constant(q, NU=None):
    q = np.abs(q) / np.linalg.norm(q)
    US, _ = grid(NU or CFG["NU"])
    best = (np.inf, None)
    for u in US:
        X, _ = traj(lambda x, u=u: u)
        if X is None:
            continue
        D = np.exp(A - u * E)[None, :] * (1 + HC[None, :] * X)
        nr = np.linalg.norm(D, axis=1, keepdims=True)
        D = D / nr
        L = float(np.mean(ell(np.arccos(np.clip(D @ q, -1, 1)),
                              nr[:, 0] ** 2)))
        if L < best[0]:
            best = (L, float(u))
    return best


def const_curve(u, n=600):
    X, Tf = traj(lambda x, u=u: u, n)
    D = np.exp(A - u * E)[None, :] * (1.0 + HC[None, :] * X)
    nr = np.linalg.norm(D, axis=1)
    return X, D / nr[:, None], nr ** 2, Tf


def beta_min_range(q, X):
    US, PHI = grid(CFG["NU"])
    out = []
    for x in X:
        D = PHI * (1.0 + HC[None, :] * x[None, :])
        D = D / np.linalg.norm(D, axis=1, keepdims=True)
        out.append(np.degrees(np.arccos(np.clip((D @ q).max(), -1, 1))))
    return float(np.min(out)), float(np.max(out))


def levelset_table():
    rows = [const_curve(u) for u in (0.0, 1.0, 1.5, 2.0, 3.0)]
    F = float(np.mean(rows[2][2]))
    out = []
    for u, (X, DH, d2, Tf) in zip((0.0, 1.0, 1.5, 2.0, 3.0), rows):
        dl = float(np.sqrt(F / np.mean(d2)))
        g = CFG["gain"] * dl ** 2 * d2
        wv = g / (1 + g)
        b = np.arccos(np.clip(DH @ QS, -1, 1))
        L = float(np.mean(-0.5 * np.log(np.maximum(
            1 - wv * np.sin(b) ** 2, 1e-300))))
        out.append((u, dl, F, float(np.degrees(b.std())), L))
    return out


def karcher(DH, mu, iters=500):
    b = (mu[:, None] * DH).sum(0)
    b /= np.linalg.norm(b)
    for _ in range(iters):
        cth = np.clip(DH @ b, -1, 1)
        R = DH - np.outer(cth, b)
        nr = np.linalg.norm(R, axis=1, keepdims=True)
        V = np.where(nr > 1e-14, R / np.maximum(nr, 1e-300), 0.0) \
            * np.arccos(cth)[:, None]
        v = (mu[:, None] * V).sum(0)
        nv = np.linalg.norm(v)
        if nv < 1e-15:
            break
        b = np.cos(nv) * b + np.sin(nv) * v / nv
        b /= np.linalg.norm(b)
    return b


def barycentre_scan():
    """quality of the barycentre rule against the characterisation point."""
    best, worst = (np.inf, None), (0.0, None)
    for u0 in np.linspace(0.1, 3.0, 20):
        X, DH, d2, Tf = const_curve(u0)
        g = CFG["gain"] * d2
        wv = g / (1 + g)
        q = karcher(DH, wv / wv.sum())
        L, _, _ = greedy(q)
        if L < best[0]:
            best = (L, float(u0))
        if L > worst[0]:
            worst = (L, float(u0))
    return best, worst


# ------------------------------------------------------------ manuscript --
def claims(path):
    # Collapse whitespace runs to single spaces. Rewrapping a paragraph used
    # to move a line break into the middle of a quoted quantity and silently
    # turn its check into "NOT FOUND in tex"; after this, the patterns below
    # only ever see single-spaced text and reflowing cannot break them.
    s = io.open(path, encoding="utf-8").read()
    return re.sub(r"\s+", " ", s)


SKIPS = []


def skip(*a):
    """A skipped section used to print a note and still let the run
    report ALL CHECKS PASS, so a missing dataset looked exactly like a
    passing one. Record it: the verdict must not be clean while the
    coverage is not."""
    msg = " ".join(str(x) for x in a)
    SKIPS.append(msg.strip())
    print(msg)



def discussion_leads(path):
    """How many practical readings does the Discussion actually give?

    An earlier draft announced three and gave four. The announced count is a
    number in the manuscript like any other and is checked like any other.

    The readings used to be italicised, which gave the count something to
    match on. They are now ordinary paragraphs, so the count is the number of
    paragraphs between the opening sentence and the Limitations heading.
    """
    t = io.open(path, encoding="utf-8").read()
    body = t.split(r"\section{Discussion}")[1]
    body = body.split(r"\paragraph{Limitations}")[0]
    blocks = [b for b in body.split("\n\n") if b.strip()]
    # block 0 is \label, block 1 is the sentence announcing the count
    return len(blocks) - 2


def discussion_order(path):
    """Are the three plant-independent readings the first three?

    Returns 1.0 when the opening's promise about ordering holds. The three
    are identified by their opening words, which are also what a reader
    scanning the section sees.
    """
    t = io.open(path, encoding="utf-8").read()
    body = t.split(r"\section{Discussion}")[1]
    body = body.split(r"\paragraph{Limitations}")[0]
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    paras = [p for p in paras if not p.startswith(r"\label")]
    if len(paras) < 4:
        return 0.0
    # Two unconditional readings, not three: "Misalignment is
    # worth measuring" went with the remaining-life benchmark that
    # supported it when the paper was cut to its theory.
    # "A Fisher criterion cannot be used to design a fixed indicator" was
    # broader than Proposition 4.6, which rules out ranking by the loss
    # and not designing on other grounds. The reading was reworded; the
    # check still asks that the two plant-independent readings lead.
    lead = ("Test the separability", "No Fisher criterion")
    got = tuple(p[:22] for p in paras[1:3])
    return float(all(g.startswith(x[:22]) for g, x in zip(got, lead)))


def find(s, pat):
    m = re.search(pat, s)
    return float(m.group(1)) if m else None


def baseline_rows(path):
    """Table III as a dict from indicator name to its multiple of optimum."""
    t = io.open(path, encoding="utf-8").read()
    tab = t.split(r"\label{tab:baselines}")[1].split(r"\end{tabular}")[0]
    out = {}
    for line in tab.split(r"\\"):
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 3:
            continue
        m = re.search(r"([0-9.]+)", cells[2])
        name = re.sub(r"\s+", " ", cells[0]).strip()
        if m and name and not name.startswith("indicator"):
            out[name] = float(m.group(1))
    return out


def caption_matches_table(path):
    """Does Table III's caption say what Table III shows?

    The caption makes three claims: that aiming at a representative point
    comes within a factor of two of the optimum, that the two endpoint rules
    cost between the two figures it quotes, and that those two figures
    bracket equal weights. All three are read off the rows.
    """
    r = baseline_rows(path)
    need = ("aligned at start of life", "aligned at failure", "equal weights",
            "aligned at mid-life")
    if not all(k in r for k in need):
        return 0.0
    ends = (r["aligned at start of life"], r["aligned at failure"])
    eq = r["equal weights"]
    bary = [v for k, v in r.items() if k.startswith("barycentre")]
    rep = [r["aligned at mid-life"]] + bary
    t = io.open(path, encoding="utf-8").read()
    cap = t.split(r"\label{tab:baselines}")[0].split(r"\caption{")[-1]
    lo = find(cap, r"costs \$([0-9]+)\$ to")
    hi = find(cap, r"to \$([0-9]+)\$ times it")
    br = find(cap, r"brackets the \$([0-9]+)\$")
    return float(
        all(v <= 2.0 for v in rep)                 # representative point
        and lo == min(ends) and hi == max(ends)    # quoted range is the rows'
        and br == eq and min(ends) < eq < max(ends))  # and it brackets


def longest_echo(path):
    """Longest run of words the prose says twice.

    Displays, tables and captions are removed first: a repeated table row is
    not a repeated sentence. A short run is a keyword; a long one is a
    sentence the reader has already read.
    """
    import collections
    t = io.open(path, encoding="utf-8").read()
    t = t.split(r"\linenumbers", 1)[1].split(r"\section*{Data availability")[0]
    t = re.sub(r"\\begin\{(equation|align|aligned|table|figure|tabular)\*?\}"
               r".*?\\end\{\1\*?\}", " ", t, flags=re.S)
    t = re.sub(r"\\caption\{.*?\n\n", " ", t, flags=re.S)
    w = re.findall(r"[a-z]+", t.lower())
    n = 8
    while n < 60:
        c = collections.Counter(tuple(w[i:i + n])
                                for i in range(len(w) - n + 1))
        if not any(v > 1 for v in c.values()):
            return float(n - 1)
        n += 1
    return float(n)


def fleet_count(path):
    """The announced number of fleets against the number cited for them.

    The fleets used to be listed in the sentence that announces them, and the
    citations sat between the colon and the next full stop. They are rows of
    Table~\\ref{tab:fleets} now, so the citations moved with them and this
    reads the announcement from the sentence and the citations from the table.
    The check is unchanged in intent: a paper that says "six fleets" and cites
    five has an error in the one place a reader counts.

    Keys are counted rather than \\cite groups, because one row cites two
    fleets in two commands and another might cite two in one.
    """
    t = io.open(path, encoding="utf-8").read()
    # this helper reads the RAW file, not claims() output, so the
    # pattern has to survive an 78-column wrap falling anywhere
    m = re.search(r"what\s+(\w+)\s+public\s+run-to-failure\s+fleets", t)
    if not m:
        return -1.0
    words = dict(three=3, four=4, five=5, six=6, seven=7, eight=8)
    said = words.get(m.group(1).lower(), -1)
    i = t.find(r"\label{tab:fleets}")
    if i < 0:
        return -1.0
    j = t.find(r"\end{tabular}", i)
    keys = set()
    for grp in re.findall(r"\\cite\{([^}]*)\}", t[i:j]):
        keys.update(k.strip() for k in grp.split(",") if k.strip())
    return float(said - len(keys))


def roadmap_gap(path):
    """Sections the Introduction's roadmap fails to mention.

    Returns the count, so zero is the passing value. Ranges are expanded:
    "Sections~A to~B" covers everything between A and B in document order.
    """
    t = io.open(path, encoding="utf-8").read()
    order = re.findall(r"\\section\{[^}]*\}\s*\\label\{([^}]*)\}", t)
    exempt = {"sec:intro", "sec:discussion", "sec:conclusion"}
    want = [k for k in order if k not in exempt]

    # Related work was merged into the Introduction, so the section
    # that used to end it is gone; the model section now does, and
    # the roadmap paragraph is the one that names it.
    intro = t.split(r"\section{Problem formulation}")[0]
    para = [p for p in intro.split("\n\n")
            if r"\ref{sec:prelim}" in p and r"\item" not in p]
    if not para:
        return float(len(want))
    road = para[-1]
    named = re.findall(r"\\ref\{(sec:[^}]*)\}", road)
    have = set(named)
    # expand "A to~B" and "A--B" into the sections they span
    for a, b in re.findall(r"\\ref\{(sec:[^}]*)\}\s*(?:to|--)~?"
                           r"\\ref\{(sec:[^}]*)\}", road):
        if a in order and b in order:
            i, j = sorted((order.index(a), order.index(b)))
            have.update(order[i:j + 1])
    return float(len([k for k in want if k not in have]))


def floor_along(X):
    """The reachability floor beta_min(x) at every state of a trajectory."""
    US, PHI = grid(CFG["NU"])
    out = []
    for x in X:
        D = PHI * (1.0 + HC[None, :] * x[None, :])
        D = D / np.linalg.norm(D, axis=1, keepdims=True)
        out.append(np.degrees(np.arccos(np.clip((D @ QS).max(), -1, 1))))
    return np.array(out)


def floor_shape():
    """Where the floor peaks, where it dips, and which drawn arc is closest.

    Returns (tau at the maximum, tau at the minimum, both along the closed
    loop, and 1.0 if the mid-life arc of Fig. 4(a) is the closest of the
    three it draws). The arcs are taken at u0 = 1.2, the trajectory the
    figure plots, so the last flag is a statement about the figure.
    """
    b = floor_along(greedy(QS)[2])
    tau = np.linspace(0.0, 1.0, len(b))
    bn = floor_along(const_curve(1.2)[0])
    taun = np.linspace(0.0, 1.0, len(bn))
    three = [float(np.interp(f, taun, bn)) for f in (0.05, 0.45, 0.9)]
    return (float(tau[b.argmax()]), float(tau[b.argmin()]),
            float(three[1] < min(three[0], three[2])))


def sub_is_current():
    """1.0 when sub.tex equals what make_sub.py would write from theory.tex."""
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import make_sub
    except Exception:
        return -1.0
    try:
        want = make_sub.render()
        have = io.open(os.path.join(here, "sub.tex"),
                       encoding="utf-8-sig").read()
    except Exception:
        return -1.0
    return float(want == have)


def generic_qualifiers(path):
    """Unqualified consequences drawn from $m<d-1$. Zero is the passing value.

    CANNOT SEE: the $m=1<d-1$ form without a trailing "=2", used in Section
    6.3 and Fig. 4's caption for computed facts rather than inferences; any
    overclaim phrased without the inequality at all; and every other
    qualifier the paper's results need.

    Remark 5.5 is a counterexample to the unqualified form, so a sentence
    that draws a conclusion from $m<d-1$ must hedge it -- "generic",
    "generically", or an explicit pointer to the remark. Occurrences that
    merely state the arithmetic ("so $m=1<d-1=2$") assert nothing and are
    not required to hedge.
    """
    t = io.open(path, encoding="utf-8").read()
    t = re.sub(r"\s+", " ", t)
    bad = 0
    for m in re.finditer(r"\$m(?:=1)?\s*<\s*d-1(?:=2)?\$", t):
        # the $m=1<d-1$ form without the trailing "=2" is used for computed
        # facts about the test plant, which need no qualifier; only the
        # bare condition and 7.1's "$m=1<d-1=2$" state an inference
        if "=1" in m.group(0) and "=2" not in m.group(0):
            continue
        tail = t[m.end():m.end() + 240]
        head = tail.lstrip()
        # a bare restatement of the inequality asserts nothing
        if head.startswith("="):
            continue
        # "when it does not fail, $m<d-1$ suffices" is the exceptional case:
        # a sufficiency claim, the opposite of the overclaim hunted here
        if head.startswith("suffices"):
            continue
        if not re.search(r"generic|Remark~\\ref\{rem:family\}", tail):
            bad += 1
    return float(bad)


def unscoped_regime_claims(path):
    """Sentences giving C-MAPSS six regimes without naming FD002/FD004.

    Zero is the passing value. FD001 and FD003 carry one operating condition
    each -- a fact this script checks two tiers up -- so an unqualified claim
    is wrong for half the benchmark.
    """
    t = io.open(path, encoding="utf-8").read()
    t = re.sub(r"\s+", " ", t)
    bad = 0
    for m in re.finditer(r"six[^.]{0,40}regimes", t):
        a = t.rfind(". ", 0, m.start())
        b = t.find(". ", m.end())
        sent = t[(0 if a < 0 else a + 2):(len(t) if b < 0 else b + 1)]
        if not re.search(r"FD00[24]|these subsets", sent):
            bad += 1
    return float(bad)


def unreferenced_panels(path):
    """Captioned figure panels the running text never sends a reader to.

    Zero is the passing value. A caption describing a panel is not prose
    directing a reader to look at it, and in the journal's two-column layout
    each figure costs a column-width block.

    CANNOT SEE: whether a reference sits in the paragraph that needs it, or
    whether the panel shows what the citing sentence claims -- both were
    checked by hand for Fig. 4(a) and neither is mechanical. A whole-figure
    reference is deliberately counted as covering no panel.
    """
    t = re.sub(r"\s+", " ", io.open(path, encoding="utf-8").read())
    bad = 0
    for m in re.finditer(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", t):
        blk = m.group(1)
        lab = re.search(r"\\label\{(fig:[^}]*)\}", blk)
        cap = re.search(r"\\caption\{(.*)", blk)
        if not lab or not cap:
            continue
        c = cap.group(1)
        panels = (set(re.findall(r"\(([a-c])[,)~ ]", c))
                  | set(re.findall(r"[,~ ]([a-c])\)", c)))
        body = t[:m.start()] + t[m.end():]
        refs = set()
        for r in re.findall(r"\\ref\{" + re.escape(lab.group(1))
                            + r"\}\s*\(([^)]*)\)", body):
            refs.update(re.findall(r"[a-c]", r))
        bad += len(panels - refs)
    return float(bad)


def inverted_regime_ratio(path):
    """Descriptions of the regime index that state its reciprocal. Zero passes.

    rho is drift over arc length: Section 6.7 defines it that way, Fig. 4(b)
    labels its axis that way, and the Discussion reads it that way. The
    Conclusion said "their width relative to their drift", the reciprocal.
    A ratio and its reciprocal both decide the same question, so nothing else
    in the paper contradicts it -- it simply names a quantity the paper never
    defines.

    CANNOT SEE: the same inversion phrased with "arc" or "length" instead of
    "width"; an inversion inside the definition in Section 6.7; or a numeric
    value quoted as 1/rho.
    """
    t = re.sub(r"\s+", " ", io.open(path, encoding="utf-8").read())
    return float(len(re.findall(
        r"width[^.]{0,40}relative to (?:the |their )?drift", t)))


def repro_exceptions(path):
    """Announced count of reproducibility exceptions minus the items listed.

    Zero is the passing value. The statement said "Three groups" after a
    fourth had been found; the same shape of error as the fleet count, and
    invisible for the same reason -- every item in the list is correct, only
    the number in front of them is not.

    Items are counted by their Section references, since each exception names
    exactly one section.

    CANNOT SEE: whether the listed exceptions are the right ones, or whether
    a fifth exists that the sentence does not mention at all.
    """
    t = re.sub(r"\s+", " ", io.open(path, encoding="utf-8").read())
    m = re.search(r"(\w+) groups of numbers lie outside it:(.*?)\.\s", t)
    if not m:
        return -1.0
    words = dict(two=2, three=3, four=4, five=5, six=6)
    said = words.get(m.group(1).lower(), -1)
    listed = len(re.findall(r"Section~\\ref\{", m.group(2)))
    return float(said - listed)


def limitation_count(path):
    """Announced number of limitations minus the number actually given.

    Zero passes, matching repro_exceptions. The limitations are SENTENCES of a
    single paragraph, so they cannot be counted as paragraphs the way the
    Discussion readings are, and only three of the four carry a Section
    reference, so they cannot be counted by \\ref the way the reproducibility
    exceptions are. They are counted as sentences after the announcing one,
    with math and cross-references removed first so that neither a decimal
    point nor "Section~\\ref{...}" creates a false boundary.

    CANNOT SEE: whether each sentence is a distinct limitation rather than a
    continuation of the one before it.
    """
    t = io.open(path, encoding="utf-8").read()
    if r"\paragraph{Limitations}" not in t:
        return -1.0
    body = t.split(r"\paragraph{Limitations}")[1]
    body = re.split(r"\\(?:section|paragraph|subsection)\{", body)[0]
    body = re.sub(r"\$[^$]*\$", " X ", body)
    body = re.sub(r"\\(?:eqref|ref|cite[a-z]*)\{[^}]*\}", " X ", body)
    body = re.sub(r"\\[A-Za-z]+\*?", " ", body)
    body = re.sub(r"\s+", " ", body).strip()

    words = dict(two=2, three=3, four=4, five=5, six=6, seven=7)
    m = re.match(r"(\w+) limitations? ", body)
    if not m:
        return -1.0
    said = words.get(m.group(1).lower(), -1)
    rest = body[m.end():]
    rest = rest.split(".", 1)[1] if "." in rest else ""
    given = [x for x in re.split(r"(?<=[.])\s+(?=[A-Z])", rest)
             if len(x.split()) > 4]
    return float(said - len(given))


def rho_sweep_stats():
    """The recorded regime-index sweep, recomputed from what it produced.

    rho_sweep.py draws 300 plants from a fixed seed -- two to five mechanisms,
    one or two inputs, and every other parameter varied with them -- and for
    each records the regime index and the gain of adaptive steering over the
    best constant input. It runs for about three hours, so the gate reads the
    recorded result rather than repeating it.

    CANNOT SEE: whether the sweep itself is right. It checks that Section 6.7
    quotes this file correctly, not that this file deserves to be quoted. The
    script is shipped so that a reader can regenerate it.
    """
    # os is imported here rather than at the top because the source copy of
    # this script does not import it; the repository copy does, added by the
    # $RESS_DATA patch. Importing locally keeps the two copies differing by
    # that patch alone, which is what the packaging audit checks.
    import json
    import os
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "rho_sweep.json")
    if not os.path.exists(fn):
        return None
    rows = json.load(io.open(fn, encoding="utf-8"))
    rho = np.array([r["rho"] for r in rows])
    arc = np.array([r["arc"] for r in rows])
    gain = np.array([r["gain"] for r in rows])

    def sp(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean()
        rb -= rb.mean()
        return float((ra * rb).sum()
                     / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))

    big = gain >= 5.0
    low = rho < 1.0
    veto = rho >= 1.0
    near = np.abs(np.log(rho) - np.log(0.88)) < 0.35
    return dict(
        n=float(len(rows)),
        sp=sp(rho, gain),
        sp_arc=sp(arc, gain),
        # rho = drift / arc by construction, so the drift is recovered
        # rather than stored. Section 6.7 claims it ranks the plants at
        # below 0.05 in magnitude, which is the evidence that the ratio's
        # numerator is not doing the work.
        sp_drift=sp(rho * arc, gain),
        share_low=100.0 * big[low].mean(),
        base=100.0 * big.mean(),
        miss=100.0 * big[veto].mean(),
        med_near=float(np.median(gain[near])),
    )


def check(rows):
    w1 = max(len(r[0]) for r in rows) + 1
    print(f"\n{'quantity':<{w1}} {'manuscript':>12} {'recomputed':>12}  status")
    print("-" * (w1 + 40))
    bad = 0
    for name, claimed, got, tol in rows:
        if claimed is None:
            st, bad = "NOT FOUND in tex", bad + 1
        elif got is None:
            st, bad = "not computed", bad + 1
        elif tol == "<":
            ok = got < claimed
            st = "ok (bound)" if ok else "BOUND VIOLATED"
            bad += 0 if ok else 1
        elif tol == ">":
            ok = got > claimed
            st = "ok (bound)" if ok else "BOUND VIOLATED"
            bad += 0 if ok else 1
        else:
            # a claim quoted to k decimals is honoured if the recomputed
            # value rounds to it; the relative tolerance is a second chance
            # for quantities the manuscript states loosely.
            dec = len(str(claimed).split(".")[1]) if "." in str(claimed) else 0
            ok = (round(got, dec) == round(claimed, dec)
                  or abs(claimed - got) <= tol * max(abs(got), 1e-30))
            st = "ok" if ok else "MISMATCH"
            bad += 0 if ok else 1
        cs = "--" if claimed is None else f"{claimed:12.6g}"
        gs = "--" if got is None else f"{got:12.6g}"
        print(f"{name:<{w1}} {cs} {gs}  {st}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", action="store_true",
                    help="also check Table II (re-optimises q, slow)")
    ap.add_argument("--fleet", action="store_true",
                    help="also check Section 7 against the five fleets")
    ap.add_argument("--full", action="store_true",
                    help="everything, including the Dinkelbach limit")
    ap.add_argument("--tex", default=TEX)
    args = ap.parse_args()
    s = claims(args.tex)

    print("CONFIG:", {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                      for k, v in CFG.items()})

    L_ad, Tf_ad, X_ad = greedy(QS)
    L_co, u_co = best_constant(QS)
    ratio = L_co / L_ad
    L_ad_fine, _, _ = greedy(QS, NU=161)
    ratio_fine = best_constant(QS, NU=161)[0] / L_ad_fine
    bmin_lo, bmin_hi = beta_min_range(QS, X_ad)
    tab = levelset_table()
    span = max(r[4] for r in tab) / min(r[4] for r in tab)
    fisher_pref = tab[0][4] / tab[1][4]         # loss at u=0 over u=1
    L_bal, Tf_bal, _ = greedy(np.ones(3) / np.sqrt(3))
    (Lb_best, u_best), (Lb_worst, _) = barycentre_scan()

    rows = [
        ("adaptive loss at q*",
         find(s, r"adaptive policy attains\s*\n?\$([0-9.]+)\\times10\^\{-4\}"),
         L_ad * 1e4, 5e-3),
        ("best-constant loss at q*",
         find(s, r"\$([0-9.]+)\\times10\^\{-2\}\$ under the best"),
         L_co * 1e2, 5e-3),
        ("headline ratio",
         find(s, r"input, a factor of \$([0-9]+)\$"),
         ratio, 0.01),
        ("ratio, 161-point grid",
         find(s, r"rising to \$([0-9]+)\$ at \$161\$ points"), ratio_fine,
         0.01),
        ("Table I span",
         find(s, r"exposure factor \$\\delta\$\.\s+The loss still spans a "
                 r"factor(?:\s+of)?\s+\$([0-9.]+)\$"), span, 0.01),
        ("Table I span, restated in Sec 6.2",
         find(s, r"held equal to seven digits,\s+the loss still spans a "
                 r"factor(?:\s+of)?\s+\$([0-9.]+)\$"), span, 0.01),
        ("Fisher-preferred/u=1 loss",
         find(s, r"loss is \$([0-9.]+)\$ times its value"), fisher_pref, 0.02),
        ("beta_min upper (deg)",
         find(s, r"\\beta_\{\\min\}=([0-9.]+)\^\{\\circ\}"), bmin_hi, 0.01),
        # the same factor is quoted in three further places; a
        # correction that reaches only the gated copy would leave those
        # stale and every check here would still pass
        ("headline ratio, restated in Sec 6.7",
         find(s, r"falls from \$([0-9]+)\$ to"), ratio, 0.01),
        ("headline ratio, restated in the Discussion",
         find(s, r"collapsing from \$([0-9]+)\\times\$ to under"),
         ratio, 0.01),
        # a fifth copy, in the abstract, which no pattern above reaches
        ("headline ratio, restated in the abstract",
         find(s, r"objective defined here by a factor of \$([0-9]+)\$"), ratio, 0.01),
        ("beta_min upper, restated in Sec 6.4",
         find(s, r"peak floor moves from \$([0-9.]+)\^\{\\circ\}\$"),
         bmin_hi, 0.01),
        ("beta_min upper, restated in the contributions",
         find(s, r"peak reachability floor from \$([0-9.]+)\^\{\\circ\}\$"),
         bmin_hi, 0.01),
        ("beta_min dips below (deg)",
         find(s, r"below the \$([0-9.]+)\^\{\\circ\}\$ resolution"),
         bmin_lo, "<"),
        ("lifetime factor q* vs balanced",
         find(s, r"differ by a factor \$([0-9.]+)\$ in lifetime"),
         Tf_bal / Tf_ad, 0.02),
    ]

    WORDNUM = dict(Two=2, Three=3, Four=4, Five=5, Six=6)
    _m = re.search(r"(\w+) practical readings follow", s)
    rows.append(("regime index stated upside down",
                 0.0, inverted_regime_ratio(args.tex), 0.0))
    rows.append(("figure panels the text never points at",
                 0.0, unreferenced_panels(args.tex), 0.0))
    rows.append(("unscoped six-regime claims about C-MAPSS",
                 0.0, unscoped_regime_claims(args.tex), 0.0))
    rows.append(("unqualified consequences of $m<d-1$",
                 0.0, generic_qualifiers(args.tex), 0.0))
    rows.append(("sub.tex matches theory.tex", 1.0, sub_is_current(), 0.0))
    rows.append(("sections the roadmap forgets",
                 0.0, roadmap_gap(args.tex), 0.0))
    rows.append(("longest sentence the prose says twice (words)",
                 16.0, longest_echo(args.tex), "<"))
    rows.append(("reproducibility exceptions announced vs listed",
                 0.0, repro_exceptions(args.tex), 0.0))
    rows.append(("limitations announced vs listed",
                 0.0, limitation_count(args.tex), 0.0))
    rows.append(("fleets announced minus fleets cited",
                 0.0, fleet_count(args.tex), 0.0))
    rows.append(("Discussion: unconditional readings come first",
                 1.0, discussion_order(args.tex), 0.0))
    rows.append(("Discussion readings announced vs given",
                 float(WORDNUM[_m.group(1)]) if _m else None,
                 float(discussion_leads(args.tex)), 0.0))

    for i, (u, dl, F, rms, L) in enumerate(tab):
        pat = (r"\$" + f"{u:.1f}".replace(".", r"\.") +
               r"\$ & \$([0-9.]+)\$ & \$[0-9.]+\$ & \$([0-9.]+)\$ & "
               r"\$([0-9.]+)\$")
        m = re.search(pat, s)
        rows.append((f"Table I u={u:.1f}: delta",
                     float(m.group(1)) if m else None, dl, 0.01))
        rows.append((f"Table I u={u:.1f}: rms beta",
                     float(m.group(2)) if m else None, rms, 0.01))
        rows.append((f"Table I u={u:.1f}: loss",
                     float(m.group(3)) if m else None, L, 0.02))

    _rs = rho_sweep_stats()
    if _rs is None:
        skip("  SKIPPED Section 6.7 sweep: rho_sweep.json not found")
    else:
        rows += [
            ("6.7 sweep plants",
             find(s, r"tested on \$([0-9]+)\$ further plants"),
             _rs["n"], 0.0),
            ("6.7 Spearman rho vs gain",
             find(s, r"between \$\\rho\$ and the gain is \$(-?[0-9.]+)\$"),
             _rs["sp"], 0.02),
            ("6.7 share repaying at rho<1",
             find(s, r"\$([0-9]+)\\%\$ repay steering"),
             _rs["share_low"], 0.02),
            ("6.7 base rate repaying",
             find(s, r"against \$([0-9]+)\\%\$ over the"),
             _rs["base"], 0.02),
            ("6.7 veto miss rate",
             find(s, r"would also reject \$([0-9]+)\\%\$"),
             _rs["miss"], 0.02),
            # The manuscript quotes a MAGNITUDE here, because the arc
            # correlates positively with the gain (a wider reachable set is
            # more authority) while rho correlates negatively. Quoting the
            # arc as -0.47 to make the comparison look tidy would have been a
            # sign error in the paper, hidden by a negation in the gate.
            ("6.7 arc alone, magnitude",
             find(s, r"at \$([0-9.]+)\$ in magnitude"),
             abs(_rs["sp_arc"]), 0.02),
            ("6.7 median gain at Table II's rho",
             find(s, r"the median gain is \$([0-9.]+)\$ rather"),
             _rs["med_near"], 0.02),
            # stated as a bound, not a value: 0.045 against a quoted 0.04
            # would fail a 2 per cent relative tolerance while agreeing
            # perfectly with what the sentence says.
            ("6.7 drift alone, magnitude bound",
             find(s, r"at below \$([0-9.]+)\$ in magnitude"),
             abs(_rs["sp_drift"]), "<"),
        ]

    if args.regime or args.full:


        print("[Sec 4.1] one-input family, d = 2..8 ...", flush=True)
        _ang, _ramp, _cb, _cw = one_input_family()
        rows += [
            ("4.1 family misalignment (deg)",
             find(s, r"misalignment below \$([0-9.]+)\\times10\^\{-6\}\$"),
             _ang * 1e6, "<"),
            ("4.1 family mean loss",
             find(s, r"mean loss below \$10\^\{-([0-9]+)\}\$"),
             # the claim is a loss BELOW 1e-15, so the exponent must exceed
             # the quoted 15: this row wants the other bound direction
             -np.log10(_ramp), ">"),
            ("4.1 best constant, low",
             find(s, r"against \$([0-9.]+)\\times10\^\{-5\}\$ to"),
             _cb * 1e5, 0.03),
            ("4.1 best constant, high",
             find(s, r"to \$([0-9.]+)\\times10\^\{-4\}\$ for the best constant"),
             _cw * 1e4, 0.03),
        ]
        print("[Sec 4.1] the same family mistuned by 1% ...", flush=True)
        _m, _c = family_mistuned(0.01)
        rows += [
            ("4.1 mistuned closed loop",
             find(s, r"raises the mean loss to \$([0-9.]+)\\times10\^\{-4\}\$"),
             _m * 1e4, 0.05),
            ("4.1 best constant, perturbed",
             find(s, r"against \$([0-9.]+)\\times10\^\{-4\}\$ for the best "
                     r"constant input on"),
             _c * 1e4, 0.05),
            # the point of the remark: steering must be the worse of the two
            ("4.1 mistuned is worse than not steering", 1.0,
             float(_m > _c), 0.0),
        ]
        print("\n[Table II] re-optimising q for each system ...", flush=True)
        tabII = re.findall(
            r"\$\(([0-9.,]+)\)\$ & \$([0-9.]+)\^\{\\circ\}\$ & \$([0-9.]+)\$ "
            r"& \$([0-9.]+)\\times10\^\{(-[0-9])\}\$ & \$([0-9.]+)\\times\$",
            s)
        _Lst_all = []
        for k, Ev in enumerate(REGIME):
            arc, rho, Lst, gain = regime_row(Ev)
            _Lst_all.append(Lst)
            cl = tabII[k] if k < len(tabII) else None
            rows += [
                (f"Table II row {k+1}: arc",
                 float(cl[1]) if cl else None, arc, 0.02),
                (f"Table II row {k+1}: rho",
                 float(cl[2]) if cl else None, rho, 0.03),
                (f"Table II row {k+1}: L*",
                 float(cl[3]) * 10 ** float(cl[4]) if cl else None, Lst, 0.05),
                (f"Table II row {k+1}: gain",
                 float(cl[5]) if cl else None, gain, 0.05),
            ]
            # Fig. 4(b) replots these same three regimes from constants
            # hard-carried into make_fig1.py. Nothing used to tie them to the
            # table: the figure once rounded both small gains to "3x", against
            # the 2.7 quoted in the text. Check them here.
            fg = fig1_constants()
            if fg is None:
                skip("  SKIPPED Fig. 4(b): make_fig1.py not readable")
            else:
                rows += [
                    (f"Fig. 4(b) row {k+1}: rho", fg["rho"][k], rho, 0.03),
                    (f"Fig. 4(b) row {k+1}: L*", fg["L_ad"][k], Lst, 0.05),
                    (f"Fig. 4(b) row {k+1}: gain", fg["gain"][k], gain, 0.05),
                ]




        _fmax, _fmin, _fmid = floor_shape()
        rows += [
            # the floor is largest at birth: it does not "rise to" 4.52
            ("floor peaks at the start of life (tau)", 0.02, _fmax, "<"),
            # and the dip is at mid-life, not early
            ("floor dips at mid-life, not early", 1.0,
             float(0.35 < _fmin < 0.65), 0.0),
            # Fig 4(a): the arc the caption calls closest really is
            ("Fig 4(a) mid-life arc is the closest of three", 1.0,
             _fmid, 0.0),
        ]
        rows.append(
            ("Sec 6.7 loss rises by a factor over the regimes",
             find(s, r"rises by a factor \$([0-9]+)\$"),
             _Lst_all[-1] / _Lst_all[0], 0.05))
        print("\n[Sec 6.8] loss against remaining-life error ...",
              flush=True)
        _id, _sp, _rt = downstream_rul()
        rows += [
            # exact algebra, so this is a machine-precision claim
            ("6.8 exp(2l) identity (rel err)", 1e-12, _id, "<"),
            ("6.8 loss vs RUL error, Spearman",
             find(s, r"at Spearman correlation \$([0-9.]+)\$"), _sp, 0.05),
            ("6.8 best-aligned beats worst",
             find(s, r"beats the worst by a factor \$([0-9.]+)\$"),
             _rt, 0.10),
        ]

        print("\n[Sec 6.1] sensitivity to the condemnation limit ...",
              flush=True)
        _pk, _gn, _dl = threshold_sensitivity(1.0)
        rows += [
            # Section 6.1 tells the reader what x_f = 1 would give. Both are
            # computed by the same routines as their 0.9 counterparts, so the
            # pair is a comparison of thresholds and not of conventions.
            ("6.1 peak floor at x_f=1 (deg)",
             find(s, r"rises from \$4\.52\^\{\\circ\}\$ to "
                     r"\$([0-9.]+)\^\{\\circ\}\$"), _pk, 0.01),
            ("6.1 gain at x_f=1",
             find(s, r"falls from \$68\$ to \$([0-9]+)\$"), _gn, 0.02),
        ]

        print("\n[Sec 2.4] Proposition 2.3, the record-level identity ...",
              flush=True)
        _rid, _rgap = record_identity()
        rows += [
            # the identity is exact, so this is a machine-precision claim
            ("Prop 2.3 record identity (abs err)", 1e-12, _rid, "<"),
            # and the record loss really is a DIFFERENT number from the
            # time-averaged one: if this ever collapses to zero the paper's
            # distinction has quietly gone and the surrounding text is wrong
            ("record vs time-averaged loss differ (rel)", 0.05, _rgap, ">"),
        ]

        print("\n[Sec 6.4] two-input reachability floor ...", flush=True)
        det, inside, ntot, res, gfloor = two_input_floor()
        rows += [
            ("2-input system determinant",
             find(s, r"determinant \$([0-9.]+)\$ here"), det, 0.02),
            ("2-input states inside the box",
             find(s, r"at every one of \$([0-9]+)\$ sampled states"),
             float(inside), 0.0),
            ("2-input states sampled",
             find(s, r"at every one of \$([0-9]+)\$ sampled states"),
             float(ntot), 0.0),
            # the residual must be zero to numerical precision, not merely
            # small: the paper claims the floor vanishes, not that it shrinks
            ("2-input residual (deg)", 1e-5, res, "<"),
            ("2-input grid floor (deg)",
             find(s, r"floor of at most \$([0-9.]+)\^\{\\circ\}\$"),
             gfloor, 0.05),
        ]

    if args.fleet or args.full:
        print("\n[Sec 7] reading the Severson fleet ...", flush=True)
        st = severson_stats(301)
        if st is None:
            skip("  SKIPPED: severson_cells.npz not found at",
                  DATA["severson"])
        else:
            rows += [
                ("Severson cells used",
                 find(s, r"\$([0-9]+)\$ Severson lithium-ion cells"), st["n"], 0.0),
                ("Severson arc travel (deg)",
                 find(s, r"travels\s*\n?\$([0-9.]+)\^\{\\circ\}\$ of arc"),
                 st["travel"], 0.01),
                ("Severson end-to-end (deg)",
                 find(s, r"ends \$([0-9]+)\^\{\\circ\}\$ from"),
                 st["e2e"], 0.03),
                ("Severson positivity at win 301",
                 1.0, st["allpos"], 0.0),
            ]

        print("[Sec 7] XJTU-SY ...", flush=True)
        xj = xjtu_stats()
        if xj is None:
            skip("  SKIPPED: not found at", DATA["xjtu"])
        else:
            ps = [xjtu_stats(w)["p"] for w in (11, 15, 21, 31)]
            rows += [
                ("XJTU conditions", 3.0, float(xj["ncond"]), 0.0),
                ("XJTU cells per condition", 5.0, float(xj["sizes"][0]), 0.0),
            ]

        print("[Sec 7] PRONOSTIA ...", flush=True)
        pr = pronostia_stats()
        if pr is None:
            skip("  SKIPPED: not found at", DATA["pronostia"])
        else:
            rows += [
            ]

        print("[Sec 7] NASA ...", flush=True)
        na = nasa_stats()
        if na is None:
            skip("  SKIPPED: not found at", DATA["nasa"])
        else:
            tr = [nasa_stats(w)["travel"] for w in (11, 21, 31, 41)]
            rows += [
                ("NASA cells", 4.0, float(na["n"]), 0.0),
                ("NASA window spread (fold)",
                 7.0, max(tr) / min(tr), 0.05),
                ("NASA positivity beyond win 31",
                 1.0, nasa_stats(31)["allpos"], 0.0),
            ]


        print("[Sec 7] separability: two-way layout inside engines ...",
              flush=True)
        _ad = None
        try:
            _ad = additivity()
        except Exception as exc:
            skip("  SKIPPED separability test:", exc)
        if _ad is None:
            skip("  SKIPPED separability test: no complete layout")
        else:
            _rg, _st, _it, _ne, _ss_rg, _ss_it = _ad
            rows += [
                # the sentence "in absolute terms it exceeds the regime
                # main effect" is a claim about the sums of squares, not the
                # ratios, so it is checked on the sums of squares
                ("Sec 7 interaction exceeds regime effect", 1.0,
                 float(_ss_it > _ss_rg), 0.0),
            ]


        print("[Sec 7] the same comparison under a second predictor ...",
              flush=True)
        try:
            _c1, _c2, _bt, _nf = rul_two_predictors()
            rows += [
                # the sentence that stops the agreement being read as two
                # weak estimators failing alike
                ("Sec 7 fleets where regression is more accurate",
                 float(_nf), float(_bt), 0.0),
            ]
        except Exception as exc:
            skip("  SKIPPED Sec 7 second predictor:", exc)

        print("[Sec 7] N-CMAPSS separability (slow) ...", flush=True)
        try:
            _o, _l, _i, _ratio, _nu = ncmapss_sep()
            _lo, _hi, _minr = ncmapss_sweep()
            rows += [
                ("Sec 7 NC interaction / main",
                 # anchored on the following word: DS01 adds a second
                 # occurrence of this phrase further down
                 find(s, r"factor \$([0-9.]+)\$ over the operating-point effect"),
                 _ratio, 0.03),
                # the sentence that the sweep exists to support
                ("Sec 7 NC interaction always exceeds the main effect",
                 1.0, float(_minr > 1.0), 0.0),
            ]

            _o1, _l1, _i1, _r1, _n1 = ncmapss_sep(
                units=(1, 2, 3, 4, 5, 6), key="ncmapss01")
            rows += [
                ("Sec 7 NC DS01 interaction / main",
                 find(s, r"on DS02 and \$([0-9.]+)\$ on DS01"),
                 _r1, 0.03),
                # the sentence says DS01 agrees, which means the same
                # inequality must hold there
                ("Sec 7 NC DS01 interaction exceeds the main effect",
                 1.0, float(_r1 > 1.0), 0.0),
            ]
        except Exception as exc:
            skip("  SKIPPED N-CMAPSS:", exc)
        print("[Sec 7] C-MAPSS FD004/FD002 (slow) ...", flush=True)
        c4 = cmapss_stats("train_FD004.txt")
        c2 = cmapss_stats("train_FD002.txt")
        if c4 is None or c2 is None:
            skip("  SKIPPED: not found at", DATA["cmapss"])
        else:
            W0 = 7                        # the window the letter quotes
            npos = c4["real"][W0][1] + c2["real"][W0][1]
            ntot = c4["real"][W0][2] + c2["real"][W0][2]
            # every window must agree in sign, and the control must not
            worst_ctrl = max(c4["ctrl"][w][0] for w in c4["ctrl"])
            worst_ctrl = max(worst_ctrl,
                             max(c2["ctrl"][w][0] for w in c2["ctrl"]))
            weakest_p = min([c4["ctrl"][w][1] for w in c4["ctrl"]]
                            + [c2["ctrl"][w][1] for w in c2["ctrl"]])
            rows += [
                ("C-MAPSS regimes, FD004", 6.0, float(c4["nregime"]), 0.0),
                ("C-MAPSS regimes, FD002", 6.0, float(c2["nregime"]), 0.0),
                ("C-MAPSS B-W, FD004 (deg)",
                 find(s, r"by \$([0-9.]+)\^\{\\circ\}\$ \(FD004\)"),
                 c4["real"][W0][0], 0.02),
                ("C-MAPSS B-W, FD002 (deg)",
                 find(s, r"\(FD004\) and\s*\$([0-9.]+)\^\{\\circ\}\$"
                         r" \(FD002\)"),
                 c2["real"][W0][0], 0.02),
                ("C-MAPSS engines with B>W",
                 find(s, r"positive for \$([0-9]+)\$ of"), float(npos), 0.0),
                ("C-MAPSS engines total",
                 find(s, r"of \$([0-9]+)\$ engines"),
                 float(ntot), 0.0),
                ("C-MAPSS engines total, conclusion",
                 find(s, r"and on \$([0-9]+)\$ simulated turbofan records it is"),
                 float(ntot), 0.0),
                # "stable over w in {7,9,11}": the smallest effect over the
                # three windows must still clear the largest control bias
                ("C-MAPSS weakest B-W over w (deg)",
                 worst_ctrl,
                 min([c4["real"][w][0] for w in c4["real"]]
                     + [c2["real"][w][0] for w in c2["real"]]), ">"),
                ("C-MAPSS control bias (deg)",
                 find(s, r"\\lvert B-W\\rvert\\le\s*([0-9.]+)\^\{\\circ\}"),
                 worst_ctrl, "<"),
                ("C-MAPSS control p, at chance",
                 0.05, weakest_p, ">"),
            ]

    if args.full:
        print("\n[Sec 6.6] Dinkelbach limit at q* ...", flush=True)
        lam = dinkelbach_limit()
        rows += [
            ("lambda* extrapolated",
             find(s, r"\\lambda\^\{\\star\}_\\infty=([0-9.]+)"
                     r"\\times10\^\{-4\}"), lam * 1e4, 0.03),
            # the letter's own guard: the limit must lie below a policy that
            # is actually realisable at the same control grid
            ("lambda* below realisable", L_ad, lam, "<"),
            # Section 6.5's "2.7% above the optimum" is exactly these two
            # numbers, and nothing read it until now. Its companion, 9.4% at
            # "a poorly chosen indicator", stays unchecked because the paper
            # does not say which indicator that is.
            ("Sec 6.5 myopic gap at q* (%)",
             find(s, r"sits \$([0-9.]+)\\%\$ above the optimum"),
             (L_ad / lam - 1.0) * 100.0, 0.05),
        ]

    bad = check(rows)
    if SKIPS:
        print(f"\n{len(SKIPS)} SECTION(S) SKIPPED -- coverage is incomplete:")
        for m in SKIPS:
            print("   ", m)
    total = bad + len(SKIPS)
    verdict = "ALL CHECKS PASS" if total == 0 else f"{total} PROBLEM(S)"
    print(f"\n{verdict}")
    return 1 if total else 0


def fig1_constants(path="make_fig1.py"):
    """Pull rho, L_ad and the gain vector out of the figure script.

    Read from the syntax tree rather than executed, so this stays a check and
    never a second way to run the plotting code.
    """
    import ast
    import os
    if not os.path.exists(path):
        return None
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    got = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if name not in ("rho", "L_ad", "L_co"):
            continue
        # rho / L_ad are np.array([...]); L_co is L_ad * np.array([...])
        for call in ast.walk(node.value):
            if (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "array"):
                got[name] = [float(e.value) for e in call.args[0].elts]
    if not {"rho", "L_ad", "L_co"} <= set(got):
        return None
    # L_co is written as L_ad * gains, so the array literal IS the gain vector
    return dict(rho=got["rho"], L_ad=got["L_ad"], gain=got["L_co"])




def cmapss_intervals(fname, single=False, nboot=4000, seed=5):
    """Bootstrap over engines for the split-half statistics.

    Returns (mean, lo, hi) for the real contrast and for the fake-regime
    control. On a single-condition subset the real contrast does not exist,
    so only the control is returned and it measures the procedure's bias with
    no condition structure available to contaminate it.
    """
    import numpy as _np
    rng = _np.random.default_rng(seed)
    d, sens = _cm_load(fname, guard_constant=True)
    G = {k: g.sort_values("cycle") for k, g in d.groupby(["unit", "cond"])}
    w, real, ctrl = 7, [], []
    for u in sorted({uu for uu, _ in G}):
        cs = [c for (uu, c) in G if uu == u]
        if not single and len(cs) >= 2:
            m = min(len(G[(u, c)]) for c in cs)
            if m >= 2 * (w + 3):
                H = {}
                for c in cs:
                    g = G[(u, c)]
                    idx = _np.unique(_np.linspace(0, len(g) - 1, m)
                                     .round().astype(int))
                    g = g.iloc[idx]
                    tau = g["tau"].to_numpy(float)
                    Y = g[sens].to_numpy(float)
                    a = _cm_dhat(tau[0::2], Y[0::2], w)
                    b = _cm_dhat(tau[1::2], Y[1::2], w)
                    if a is not None and b is not None:
                        H[c] = (a, b)
                ks = sorted(H)
                if len(ks) >= 2:
                    ww = [_cm_ang(*H[c]) for c in ks]
                    bb = [_cm_ang(H[c][0], H[c2][1])
                          for i, c in enumerate(ks) for c2 in ks[i + 1:]]
                    real.append(_np.mean(bb) - _np.mean(ww))
        wf, bf = [], []
        for c in cs:
            g = G[(u, c)]
            if len(g) < 4 * (w + 3):
                continue
            tau = g["tau"].to_numpy(float)
            Y = g[sens].to_numpy(float)
            q = [_cm_dhat(tau[i::4], Y[i::4], w) for i in range(4)]
            if any(x is None for x in q):
                continue
            wf.append(_cm_ang(q[0], q[1]))
            bf.append(_cm_ang(q[0], q[3]))
        if wf:
            ctrl.append(_np.mean(bf) - _np.mean(wf))

    def ci(v):
        v = _np.asarray(v)
        if v.size == 0:
            return (float("nan"),) * 3
        b = _np.array([rng.choice(v, v.size, replace=True).mean()
                       for _ in range(nboot)])
        return (float(v.mean()), float(_np.percentile(b, 2.5)),
                float(_np.percentile(b, 97.5)))

    return ci(real), ci(ctrl), int(d["cond"].nunique())


def hi_criteria(ndraw=8, nunit=40, noise=0.03, u_nom=1.2, nrestart=25):
    """Indicators selected by the criteria in routine prognostic practice.

    A fleet of units with dispersed rate parameters is simulated at a
    constant nominal operating point and observed with noise, which is the
    record a designer holds before any steering is contemplated. The weight
    vector is then chosen by each criterion and scored by the paper's own
    closed-loop loss, so the comparison is like for like with Table III.

    Definitions follow \\cite{coble2009}: monotonicity is the normalised
    excess of rises over falls, trendability the correlation with time, and
    prognosability the spread of the failure values against the range
    travelled.
    """
    from scipy.optimize import minimize
    rng_master = np.random.default_rng(20260829)

    def one_fleet(seed):
        rng = np.random.default_rng(seed)
        recs = []
        for _ in range(nunit):
            aa = CFG["a"] + rng.normal(0, 0.08, 3)
            ee = CFG["E"] * (1.0 + rng.normal(0, 0.05, 3))
            cc = CFG["c"] * (1.0 + rng.normal(0, 0.05, 3))
            x0 = np.full(3, CFG["x0"]) * np.exp(rng.normal(0, 0.2, 3))

            def rhs(t, x, aa=aa, ee=ee, cc=cc):
                return np.exp(aa - ee * u_nom) * (1.0 + cc * x)

            def hit(t, x):
                return CFG["xfail"] - x.max()
            hit.terminal, hit.direction = True, -1
            sol = solve_ivp(rhs, (0, 4000), x0, events=hit, max_step=2.0,
                            rtol=1e-9, atol=1e-11, dense_output=True)
            if not len(sol.t_events[0]):
                continue
            tf = sol.t_events[0][0]
            ts = np.linspace(0, tf, 160)
            Y = sol.sol(ts).T + rng.normal(0, noise, (160, 3))
            recs.append((ts, Y, tf))
        return recs

    def crit(v, recs):
        v = np.abs(np.asarray(v, float))
        v = v / (np.linalg.norm(v) + 1e-15)
        mono, trend, zf, z0 = [], [], [], []
        for ts, Y, tf in recs:
            z = Y @ v
            dz = np.diff(z)
            mono.append(abs((dz > 0).sum() - (dz < 0).sum()) / len(dz))
            trend.append(abs(np.corrcoef(z, ts)[0, 1]))
            z0.append(z[0])
            zf.append(z[-1])
        zf, z0 = np.array(zf), np.array(z0)
        prog = np.exp(-zf.std() / (abs(zf.mean() - z0.mean()) + 1e-12))
        return np.mean(mono), np.mean(trend), float(prog)

    def best_v(obj, seed):
        rng = np.random.default_rng(seed)
        top, bv = -np.inf, None
        for _ in range(nrestart):
            v0 = np.abs(rng.normal(size=3))
            r = minimize(lambda v: -obj(v), v0 / np.linalg.norm(v0),
                         method="Nelder-Mead",
                         options=dict(xatol=1e-4, fatol=1e-8, maxiter=500))
            if -r.fun > top:
                top, bv = -r.fun, np.abs(r.x) / np.linalg.norm(r.x)
        return bv

    opt = greedy(CFG["qstar"])[0]
    out = {}
    for k in range(ndraw):
        recs = one_fleet(1000 + k)
        for name, f in (
                ("monotonicity", lambda v, r=recs: crit(v, r)[0]),
                ("trendability", lambda v, r=recs: crit(v, r)[1]),
                ("prognosability", lambda v, r=recs: crit(v, r)[2]),
                ("summed", lambda v, r=recs: sum(crit(v, r)))):
            out.setdefault(name, []).append(
                greedy(best_v(f, 7))[0] / opt)
        Y0 = np.vstack([Y[:5] for _, Y, _ in recs])
        S = np.cov(Y0.T) + 1e-9 * np.eye(3)
        mv = np.abs(np.linalg.inv(np.linalg.cholesky(S)).sum(0))
        out.setdefault("mahalanobis", []).append(greedy(mv)[0] / opt)
        Xs = np.vstack([Y for _, Y, _ in recs])
        rul = np.concatenate([tf - ts for ts, _, tf in recs])
        b = np.linalg.lstsq(np.c_[Xs, np.ones(len(Xs))], rul,
                            rcond=None)[0][:3]
        out.setdefault("supervised", []).append(greedy(b)[0] / opt)
    return {k: (float(np.median(v)), float(np.min(v)), float(np.max(v)))
            for k, v in out.items()}


def cmapss_rul(tags=("FD001", "FD003", "FD002", "FD004"),
               lwin=30, kbest=5, cap=125.0):
    """Remaining-life prediction on the official C-MAPSS splits.

    Returns {tag: {name: (acute angle to dhat, RMSE, score, alpha)}}
    plus the median-RUL control under the key "control".

    The direction estimate keeps the sign of the whitened derivative. An
    earlier version took absolute values, which discards the sign structure
    that distinguishes rising from falling sensors and moved the FD001 RMSE
    from 19.0 to 31.2 -- so signs are not cosmetic here.
    """
    import os
    import pandas as pd
    from scipy.optimize import minimize as _min
    from scipy.signal import savgol_filter

    cols = (["unit", "cycle", "op1", "op2", "op3"]
            + [f"s{i}" for i in range(1, 22)])

    def read(fn):
        return pd.read_csv(os.path.join(DATA["cmapss"], fn), sep=r"\s+",
                           header=None, names=cols)

    def prep(tag):
        tr, te = read(f"train_{tag}.txt"), read(f"test_{tag}.txt")
        truth = np.loadtxt(os.path.join(DATA["cmapss"], f"RUL_{tag}.txt"))
        both = pd.concat([tr.assign(split=0), te.assign(split=1)])
        X = both[["op1", "op2", "op3"]].to_numpy(float)
        if (X.max(0) - X.min(0) > 0.1).any():
            Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
            _, cond = np.unique(np.round(Xs, 1), axis=0, return_inverse=True)
        else:
            cond = np.zeros(len(both), int)
        both["cond"] = cond
        sens = [c for c in cols if c.startswith("s")]
        # a sensor constant within ANY one condition divides by zero when that
        # condition is standardised, so the median is not a safe filter
        keep = [c for c in sens
                if both.groupby("cond")[c].std().min() > 1e-6]
        g = both.groupby("cond")
        for c in keep:
            both[c] = ((both[c] - g[c].transform("mean"))
                       / g[c].transform("std"))
        both = both.replace([np.inf, -np.inf], np.nan).dropna(subset=keep)
        return both[both.split == 0], both[both.split == 1], truth, keep

    def dhat_fleet(tr, keep, w=11):
        acc = []
        for _, g in tr.groupby("unit"):
            Y = g.sort_values("cycle")[keep].to_numpy(float)
            if len(Y) < w + 5:
                continue
            D, S = [], []
            for j in range(Y.shape[1]):
                y = Y[:, j]
                D.append(savgol_filter(y, w, 2, deriv=1))
                S.append(np.std(y - savgol_filter(y, w, 2)) + 1e-9)
            M = np.stack(D, 1) / np.array(S)[None, :]
            nr = np.linalg.norm(M, axis=1, keepdims=True)
            if np.isfinite(M).all() and (nr > 1e-12).all():
                acc.append((M / nr).mean(0))
        d = np.mean(acc, 0)
        return d / np.linalg.norm(d)

    def mono_opt(tr, keep, seed=0):
        ser = [g.sort_values("cycle")[keep].to_numpy(float)
               for _, g in tr.groupby("unit")]
        ser = [x for x in ser if len(x) > 20][:60]

        def neg(v):
            v = np.abs(v) / (np.linalg.norm(v) + 1e-12)
            m = []
            for Y in ser:
                dz = np.diff(Y @ v)
                m.append(abs((dz > 0).sum() - (dz < 0).sum()) / len(dz))
            return -np.mean(m)

        rng = np.random.default_rng(seed)
        best, bv = np.inf, None
        for _ in range(12):
            v0 = np.abs(rng.normal(size=len(keep)))
            r = _min(neg, v0 / np.linalg.norm(v0), method="Nelder-Mead",
                     options=dict(maxiter=1500, fatol=1e-6))
            if r.fun < best:
                best, bv = r.fun, np.abs(r.x) / np.linalg.norm(r.x)
        return bv

    def phm(d):
        return np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)

    def sim(v, tr, te, truth, keep):
        v = np.asarray(v, float)
        v = v / np.linalg.norm(v)
        LZ, LR = [], []
        for _, g in tr.groupby("unit"):
            z = g.sort_values("cycle")[keep].to_numpy(float) @ v
            n = len(z)
            for j in range(lwin, n):
                w = z[j - lwin:j]
                LZ.append(w - w.mean())
                LR.append(min(n - j, cap))
        LZ, LR = np.array(LZ), np.array(LR, float)
        pred = []
        for _, g in te.groupby("unit"):
            z = g.sort_values("cycle")[keep].to_numpy(float) @ v
            if len(z) < lwin:
                pred.append(LR.mean())
                continue
            w = z[-lwin:]
            w = w - w.mean()
            d = ((LZ - w) ** 2).sum(1)
            k = np.argpartition(d, kbest)[:kbest]
            wt = 1.0 / (d[k] + 1e-9)
            pred.append(float((LR[k] * wt).sum() / wt.sum()))
        t = np.minimum(truth[:len(pred)], cap)
        err = np.array(pred) - t
        hit = (np.abs(err) <= 0.2 * np.maximum(t, 1.0)).astype(float)
        return (float(np.sqrt((err ** 2).mean())), float(phm(err).sum()),
                float(hit.mean()), hit)

    out = {}
    for tag in tags:
        tr, te, truth, keep = prep(tag)
        dh = dhat_fleet(tr, keep)
        Y = tr[keep].to_numpy(float)
        pc = np.linalg.svd(Y - Y.mean(0), full_matrices=False)[2][0]
        rul = np.concatenate([np.arange(len(g), 0, -1) for _, g
                              in tr.sort_values(["unit", "cycle"])
                              .groupby("unit")])
        beta = np.linalg.lstsq(np.c_[Y, np.ones(len(Y))], rul,
                               rcond=None)[0][:len(keep)]
        res = {}
        for name, v in (("informative", dh), ("supervised", beta),
                        ("pca", pc), ("monotonicity", mono_opt(tr, keep)),
                        ("equal", np.ones(len(keep)))):
            v = np.asarray(v, float) / np.linalg.norm(v)
            ang = float(np.degrees(np.arccos(np.clip(abs(v @ dh), 0, 1))))
            r, sc, al, hit = sim(v, tr, te, truth, keep)
            res[name] = (ang, r, sc, al, hit)
        n = te["unit"].nunique()
        t = np.minimum(truth[:n], cap)
        e = np.full(n, float(np.median(t))) - t
        _ch = (np.abs(e) <= 0.2 * np.maximum(t, 1.0)).astype(float)
        res["control"] = (float("nan"), float(np.sqrt((e ** 2).mean())),
                          float(phm(e).sum()), float(_ch.mean()), _ch)
        out[tag] = res
    return out


def one_input_family(dmax=8):
    """The m=1 construction of Remark 5.5, integrated for d = 2..dmax.

    Returns (worst misalignment in degrees, worst ramp loss, best and worst
    constant-input loss over the sweep). The misalignment must be zero to
    integrator precision: it is the claim that the counting rule is not
    necessary.
    """
    U0, XF = 0.4, CFG["xfail"]
    umin, umax = CFG["U"]
    worst_ang, worst_ramp, cbest, cworst = 0.0, 0.0, np.inf, 0.0
    for d in range(2, dmax + 1):
        rng = np.random.default_rng(100 + d)
        a = rng.uniform(-0.5, 0.5, d)
        E = rng.uniform(0.4, 1.4, d)
        x0 = np.full(d, 0.02)
        k = np.exp(a - E * U0)
        tf0 = (XF - x0[0]) / k.max()
        r = (umax - U0) / tf0 * 0.98
        g = r * E / k

        def dh(x, u):
            v = np.exp(a - E * u) * np.exp(g * (x - x0))
            return v / np.linalg.norm(v)

        def run(pol, tmax):
            def rhs(t, x):
                return np.exp(a - E * pol(t, x)) * np.exp(g * (x - x0))

            def hit(t, x):
                return XF - x.max()
            hit.terminal, hit.direction = True, -1
            sol = solve_ivp(rhs, (0, tmax), x0, events=hit,
                            max_step=tmax / 3000, rtol=1e-11, atol=1e-13,
                            dense_output=True)
            if not len(sol.t_events[0]):
                return None
            tf = sol.t_events[0][0]
            ts = np.linspace(0, tf, 500)
            X = sol.sol(ts).T
            D = np.array([dh(x, pol(t, x)) for t, x in zip(ts, X)])
            return ts, D, tf

        out = run(lambda t, x: min(umax, U0 + r * t), 6 * tf0 + 200)
        if out is None:
            continue
        ts, D, tf = out
        b = np.arccos(np.clip(D @ D[0], -1, 1))
        worst_ang = max(worst_ang, float(np.degrees(b).max()))
        worst_ramp = max(worst_ramp, float(np.trapezoid(ell(b), ts) / tf))
        # the sentence compares against the BEST constant input at each d,
        # so the per-d minimum is what enters the range
        best_here = np.inf
        for u in np.linspace(umin, umax, 17):
            o = run(lambda t, x, u=u: u, 200 * tf0 + 8000)
            if o is None:
                continue
            t2, D2, tf2 = o
            bb = np.arccos(np.clip(D2 @ D2[len(D2) // 2], -1, 1))
            best_here = min(best_here, float(np.trapezoid(ell(bb), t2) / tf2))
        if np.isfinite(best_here):
            cbest, cworst = min(cbest, best_here), max(cworst, best_here)
    return worst_ang, worst_ramp, cbest, cworst


def family_mistuned(eps=0.01, d=4, nseed=5):
    """Remark 5.5's knife edge: the closed loop under eps shape error,
    against the best constant input on the same perturbed plant."""
    U0, XF = 0.4, CFG["xfail"]
    umin, umax = CFG["U"]
    US = np.linspace(umin, umax, 121)
    myo, con = [], []
    for seed in range(nseed):
        rng = np.random.default_rng(10 + seed)
        a = rng.uniform(-0.5, 0.5, d)
        E = rng.uniform(0.4, 1.4, d)
        x0 = np.full(d, 0.02)
        k = np.exp(a - E * U0)
        tf0 = (XF - x0[0]) / k.max()
        r = (umax - U0) / tf0 * 0.98
        g = (r * E / k) * (1.0 + eps *
                           np.random.default_rng(1000 + seed)
                           .choice([-1.0, 1.0], d))

        def dh(x, u):
            v = np.exp(a - E * u) * np.exp(g * (x - x0))
            return v / np.linalg.norm(v)

        def run(pol, tmax):
            def rhs(t, x):
                return np.exp(a - E * pol(t, x)) * np.exp(g * (x - x0))

            def hit(t, x):
                return XF - x.max()
            hit.terminal, hit.direction = True, -1
            sol = solve_ivp(rhs, (0, tmax), x0, events=hit,
                            max_step=tmax / 1500, rtol=1e-10, atol=1e-12,
                            dense_output=True)
            if not len(sol.t_events[0]):
                return None
            tf = sol.t_events[0][0]
            ts = np.linspace(0, tf, 300)
            X = sol.sol(ts).T
            return ts, np.array([dh(x, pol(t, x))
                                 for t, x in zip(ts, X)]), tf

        o = run(lambda t, x: min(umax, U0 + r * t), 10 * tf0 + 400)
        if o is None:
            continue
        ts, D, tf = o
        q = D[len(D) // 2]

        def pol(t, x, q=q):
            c = np.array([dh(x, u) @ q for u in US])
            return US[int(np.argmax(c))]
        o2 = run(pol, 60 * tf0 + 4000)
        if o2 is not None:
            t2, D2, tf2 = o2
            myo.append(float(np.trapezoid(
                ell(np.arccos(np.clip(D2 @ q, -1, 1))), t2) / tf2))
        best = np.inf
        for u in np.linspace(umin, umax, 17):
            o3 = run(lambda t, x, u=u: u, 80 * tf0 + 6000)
            if o3 is None:
                continue
            t3, D3, tf3 = o3
            qq = D3[len(D3) // 2]
            best = min(best, float(np.trapezoid(
                ell(np.arccos(np.clip(D3 @ qq, -1, 1))), t3) / tf3))
        con.append(best)
    return float(np.median(myo)), float(np.median(con))


def additivity(fl="train_FD004.txt", nstage=2, minpts=24, w=7):
    """Two-way layout of operating regime against stage of life, inside
    single engines, against a split-half noise floor.

    Separability (Lemma 5.2) says the regime shift is the same at every
    stage, so the interaction should not clear its floor. Pooling engines
    would let unit-to-unit variation pose as interaction, so each engine
    carries its own layout and the sums of squares are added.

    Returns (regime ratio, stage ratio, interaction ratio, engines used,
    absolute regime SS, absolute interaction SS). The absolute pair is
    needed because the text also claims the interaction exceeds the
    regime effect in absolute terms, which the ratios cannot show.
    """
    from scipy.signal import savgol_filter
    d, sens = _cm_load(fl, guard_constant=True)

    def logdir(tau, Y):
        if len(tau) < w + 3:
            return None
        D, S = [], []
        for j in range(Y.shape[1]):
            y = Y[:, j]
            D.append(savgol_filter(y, w, 2, deriv=1))
            S.append(np.std(y - savgol_filter(y, w, 2)) + 1e-12)
        M = np.stack(D, 1) / np.array(S)[None, :]
        if not np.isfinite(M).all():
            return None
        v = np.abs(M).mean(0)
        if (v < 1e-12).any():
            return None
        z = np.log(v)
        return z - z.mean()

    def twoway(M, stages, regs):
        keys = list(M)
        Y = np.array([M[k] for k in keys])
        grand = Y.mean(0)
        a = {t: np.mean([M[k] for k in keys if k[0] == t], 0) - grand
             for t in stages}
        b = {r: np.mean([M[k] for k in keys if k[1] == r], 0) - grand
             for r in regs}
        inter = np.array([M[k] - grand - a[k[0]] - b[k[1]] for k in keys])
        return np.array([
            float(np.sum(np.array(list(b.values())) ** 2)),
            float(np.sum(np.array(list(a.values())) ** 2)),
            float(np.sum(inter ** 2))])

    tot_m, tot_d, used = np.zeros(3), np.zeros(3), 0
    for u, gu in d.groupby("unit"):
        regs = sorted(gu["cond"].unique())
        if len(regs) < 3:
            continue
        M, N, ok = {}, {}, True
        for c in regs:
            g = gu[gu["cond"] == c].sort_values("cycle")
            tau = g["tau"].to_numpy(float)
            Y = g[sens].to_numpy(float)
            edges = np.linspace(tau.min(), tau.max(), nstage + 1)
            for t in range(nstage):
                m = (tau >= edges[t]) & (tau <= edges[t + 1])
                if m.sum() < minpts:
                    ok = False
                    break
                z1 = logdir(tau[m][0::2], Y[m][0::2])
                z2 = logdir(tau[m][1::2], Y[m][1::2])
                if z1 is None or z2 is None:
                    ok = False
                    break
                M[(t, c)] = 0.5 * (z1 + z2)
                N[(t, c)] = 0.5 * (z1 - z2)
            if not ok:
                break
        if not ok or len(M) != nstage * len(regs):
            continue
        st = list(range(nstage))
        tot_m += twoway(M, st, regs)
        tot_d += twoway(N, st, regs)
        used += 1
    if used == 0:
        return None
    r = tot_m / np.maximum(tot_d, 1e-30)
    return (float(r[0]), float(r[1]), float(r[2]), used,
            float(tot_m[0]), float(tot_m[2]))


def rul_bootstrap(RB_hits, angles, nboot=4000, seed=7):
    """Resample test units, paired across indicators.

    A draw takes the same units for every indicator, so the difficulty of a
    fleet cancels in the comparison between indicators and the interval
    describes the quantity the text quotes.
    """
    rng = np.random.default_rng(seed)
    tags = list(RB_hits)
    names = list(RB_hits[tags[0]])
    out = np.empty(nboot)
    for b in range(nboot):
        A, AL = [], []
        for t in tags:
            n = len(RB_hits[t][names[0]])
            idx = rng.integers(0, n, n)
            a = np.array([angles[t][nm] for nm in names])
            al = np.array([RB_hits[t][nm][idx].mean() for nm in names])
            A.append(a - a.mean())
            AL.append(al - al.mean())
        out[b] = np.corrcoef(np.concatenate(A), np.concatenate(AL))[0, 1]
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)),
            int((out > 0).sum()), nboot)


def phm2012(rescale=False, lwin=40, kbest=5, dt=10.0):
    """The official IEEE PHM 2012 bearing challenge.

    Six Learning_set bearings train a similarity predictor; the eleven
    truncated Test_set bearings get one prediction each; the truth is the
    acquisition-count difference against Full_Test_Set, times dt.

    Returns (best RMSE over the indicators, control RMSE, best challenge
    score over the indicators, control challenge score, min truth, max
    truth). A control that reads nothing is included because it decides
    whether the benchmark can separate indicators at all.
    """
    import os
    from scipy.optimize import minimize as _min
    from scipy.signal import savgol_filter

    root = DATA["phm2012"]
    npz = np.load(DATA["pronostia"], allow_pickle=True)
    nm = list(npz["names"])
    rec = {n: np.asarray(npz["data"][nm.index(n)], float) for n in nm}
    learn = ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2",
             "Bearing3_1", "Bearing3_2"]

    def count(sub, b):
        p = os.path.join(root, sub, b)
        return len([f for f in os.listdir(p) if f.startswith("acc")])

    test = sorted(os.listdir(os.path.join(root, "Test_set")))
    cut, truth = {}, {}
    for b in test:
        nt, nf = count("Test_set", b), count("Full_Test_Set", b)
        # the extracted series must cover the FULL record, or every truth
        # below is shifted; fail rather than drift
        assert len(rec[b]) == nf, (b, len(rec[b]), nf)
        cut[b], truth[b] = nt, (nf - nt) * dt
    for b in learn:
        assert len(rec[b]) == count("Learning_set", b), b

    if rescale:
        for k, Y in rec.items():
            rec[k] = Y / (np.abs(Y[:max(10, len(Y) // 10)]).mean(0) + 1e-9)

    def logdir(Y, w=31):
        D, S = [], []
        for j in range(Y.shape[1]):
            y = Y[:, j]
            D.append(savgol_filter(y, w, 2, deriv=1))
            S.append(np.std(y - savgol_filter(y, w, 2)) + 1e-12)
        M = np.stack(D, 1) / np.array(S)[None, :]
        nr = np.linalg.norm(M, axis=1, keepdims=True)
        ok = np.isfinite(M).all(1) & (nr[:, 0] > 1e-12)
        v = (M[ok] / nr[ok]).mean(0)
        return v / np.linalg.norm(v)

    dh = np.mean([logdir(rec[n]) for n in learn], 0)
    dh = dh / np.linalg.norm(dh)
    nf = rec[learn[0]].shape[1]
    Y = np.vstack([rec[n] for n in learn])
    pc = np.linalg.svd(Y - Y.mean(0), full_matrices=False)[2][0]
    r_tr = np.concatenate([np.arange(len(rec[n]), 0, -1) * dt
                           for n in learn])
    beta = np.linalg.lstsq(np.c_[Y, np.ones(len(Y))], r_tr,
                           rcond=None)[0][:nf]

    def mono(seed=0):
        rng = np.random.default_rng(seed)
        best, bv = np.inf, None
        for _ in range(12):
            v0 = np.abs(rng.normal(size=nf))

            def neg(v):
                v = np.abs(v) / (np.linalg.norm(v) + 1e-12)
                return -np.mean([abs((np.diff(rec[n] @ v) > 0).sum()
                                     - (np.diff(rec[n] @ v) < 0).sum())
                                 / (len(rec[n]) - 1) for n in learn])
            r = _min(neg, v0 / np.linalg.norm(v0), method="Nelder-Mead",
                     options=dict(maxiter=2000, fatol=1e-7))
            if r.fun < best:
                best, bv = r.fun, np.abs(r.x) / np.linalg.norm(r.x)
        return bv

    def predict(v):
        v = np.asarray(v, float)
        v = v / np.linalg.norm(v)
        LZ, LR = [], []
        for n in learn:
            z = rec[n] @ v
            m = len(z)
            for j in range(lwin, m):
                w = z[j - lwin:j]
                LZ.append(w - w.mean())
                LR.append((m - j) * dt)
        LZ, LR = np.array(LZ), np.array(LR)
        out = []
        for b in test:
            z = rec[b][:cut[b]] @ v
            w = z[-lwin:]
            w = w - w.mean()
            dd = ((LZ - w) ** 2).sum(1)
            k = np.argpartition(dd, kbest)[:kbest]
            wt = 1.0 / (dd[k] + 1e-9)
            out.append(float((LR[k] * wt).sum() / wt.sum()))
        return np.array(out)

    act = np.array([truth[b] for b in test])

    def sc(pred):
        er = 100.0 * (act - pred) / act
        a = np.where(er <= 0, np.exp(-np.log(0.5) * er / 5.0),
                     np.exp(np.log(0.5) * er / 20.0))
        return (float(np.sqrt(((pred - act) ** 2).mean())), float(a.mean()))

    rm, ss = [], []
    for v in (dh, beta, pc, mono(), np.ones(nf)):
        r, a = sc(predict(v))
        rm.append(r)
        ss.append(a)
    cr, ca = sc(np.full(len(test), float(np.median(
        [(len(rec[n]) - lwin) * dt * 0.3 for n in learn]))))
    return (min(rm), cr, max(ss), ca, float(act.min()), float(act.max()))


def rul_two_predictors(lwin=30, kbest=5, cap=125.0):
    """The Section 7 comparison under two unrelated downstream predictors.

    similarity  nonparametric window matching, as in cmapss_rul();
    regression  ridge on the level, slope and curvature of the same window.

    Returns (corr under similarity, corr under regression, number of fleets
    on which the regression has the lower RMSE, number of fleets).
    """
    import os
    import pandas as pd
    from scipy.optimize import minimize as _min
    from scipy.signal import savgol_filter

    cols = (["unit", "cycle", "op1", "op2", "op3"]
            + [f"s{i}" for i in range(1, 22)])
    tags = ("FD001", "FD002", "FD003", "FD004")

    def prep(tag):
        tr = pd.read_csv(os.path.join(DATA["cmapss"], f"train_{tag}.txt"),
                         sep=r"\s+", header=None, names=cols)
        te = pd.read_csv(os.path.join(DATA["cmapss"], f"test_{tag}.txt"),
                         sep=r"\s+", header=None, names=cols)
        truth = np.loadtxt(os.path.join(DATA["cmapss"], f"RUL_{tag}.txt"))
        both = pd.concat([tr.assign(split=0), te.assign(split=1)])
        X = both[["op1", "op2", "op3"]].to_numpy(float)
        if (X.max(0) - X.min(0) > 0.1).any():
            Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
            _, cd = np.unique(np.round(Xs, 1), axis=0, return_inverse=True)
        else:
            cd = np.zeros(len(both), int)
        both["cond"] = cd
        sens = [c for c in cols if c.startswith("s")]
        keep = [c for c in sens
                if both.groupby("cond")[c].std().min() > 1e-6]
        g = both.groupby("cond")
        for c in keep:
            both[c] = ((both[c] - g[c].transform("mean"))
                       / g[c].transform("std"))
        both = both.replace([np.inf, -np.inf], np.nan).dropna(subset=keep)
        return both[both.split == 0], both[both.split == 1], truth, keep

    def dfleet(tr, keep, w=11):
        acc = []
        for _, g in tr.groupby("unit"):
            Y = g.sort_values("cycle")[keep].to_numpy(float)
            if len(Y) < w + 5:
                continue
            D, S = [], []
            for j in range(Y.shape[1]):
                y = Y[:, j]
                D.append(savgol_filter(y, w, 2, deriv=1))
                S.append(np.std(y - savgol_filter(y, w, 2)) + 1e-9)
            M = np.stack(D, 1) / np.array(S)[None, :]
            nr = np.linalg.norm(M, axis=1, keepdims=True)
            if np.isfinite(M).all() and (nr > 1e-12).all():
                acc.append((M / nr).mean(0))
        d = np.mean(acc, 0)
        return d / np.linalg.norm(d)

    def mono(tr, keep, seed=0):
        ser = [g.sort_values("cycle")[keep].to_numpy(float)
               for _, g in tr.groupby("unit")]
        ser = [x for x in ser if len(x) > 20][:60]

        def neg(v):
            v = np.abs(v) / (np.linalg.norm(v) + 1e-12)
            return -np.mean([abs((np.diff(Y @ v) > 0).sum()
                                 - (np.diff(Y @ v) < 0).sum()) / (len(Y) - 1)
                             for Y in ser])
        rng = np.random.default_rng(seed)
        best, bv = np.inf, None
        for _ in range(12):
            v0 = np.abs(rng.normal(size=len(keep)))
            r = _min(neg, v0 / np.linalg.norm(v0), method="Nelder-Mead",
                     options=dict(maxiter=1500, fatol=1e-6))
            if r.fun < best:
                best, bv = r.fun, np.abs(r.x) / np.linalg.norm(r.x)
        return bv

    def feats(w):
        t = np.arange(len(w), dtype=float)
        c = np.polyfit(t, w, 2)
        return np.array([w[-1], c[1], c[0]])

    def both_preds(v, tr, te, truth, keep):
        v = np.asarray(v, float)
        v = v / np.linalg.norm(v)
        lib = [(g.sort_values("cycle")[keep].to_numpy(float) @ v, len(g))
               for _, g in tr.groupby("unit")]
        tez = [g.sort_values("cycle")[keep].to_numpy(float) @ v
               for _, g in te.groupby("unit")]
        LZ, LR, F = [], [], []
        for z, n in lib:
            for j in range(lwin, n):
                w = z[j - lwin:j]
                LZ.append(w - w.mean())
                LR.append(min(n - j, cap))
                F.append(feats(w))
        LZ, LR, F = np.array(LZ), np.array(LR, float), np.array(F)
        p1 = []
        for z in tez:
            if len(z) < lwin:
                p1.append(LR.mean())
                continue
            w = z[-lwin:]
            w = w - w.mean()
            d = ((LZ - w) ** 2).sum(1)
            k = np.argpartition(d, kbest)[:kbest]
            wt = 1.0 / (d[k] + 1e-9)
            p1.append(float((LR[k] * wt).sum() / wt.sum()))
        mu, sd = F.mean(0), F.std(0) + 1e-12
        Fz = np.c_[(F - mu) / sd, np.ones(len(F))]
        bta = np.linalg.solve(Fz.T @ Fz + 1e-3 * np.eye(Fz.shape[1]),
                              Fz.T @ LR)
        p2 = []
        for z in tez:
            if len(z) < lwin:
                p2.append(LR.mean())
                continue
            f = (feats(z[-lwin:]) - mu) / sd
            p2.append(float(np.clip(np.r_[f, 1.0] @ bta, 0.0, cap)))
        t = np.minimum(truth[:len(tez)], cap)
        out = []
        for p in (np.array(p1), np.array(p2)):
            e = p - t
            out.append((float(np.sqrt((e ** 2).mean())),
                        float(np.mean(np.abs(e)
                                      <= 0.2 * np.maximum(t, 1.0)))))
        return out

    names = ["informative", "supervised", "pca", "monotonicity", "equal"]
    A, S1, S2, better = [], [], [], 0
    for tag in tags:
        tr, te, truth, keep = prep(tag)
        dh = dfleet(tr, keep)
        Y = tr[keep].to_numpy(float)
        pc = np.linalg.svd(Y - Y.mean(0), full_matrices=False)[2][0]
        rl = np.concatenate([np.arange(len(g), 0, -1) for _, g
                             in tr.sort_values(["unit", "cycle"])
                             .groupby("unit")])
        bt = np.linalg.lstsq(np.c_[Y, np.ones(len(Y))], rl,
                             rcond=None)[0][:len(keep)]
        vs = dict(informative=dh, supervised=bt, pca=pc,
                  monotonicity=mono(tr, keep), equal=np.ones(len(keep)))
        a, s1, s2, r1s, r2s = [], [], [], [], []
        for n in names:
            v = np.asarray(vs[n], float) / np.linalg.norm(vs[n])
            a.append(float(np.degrees(np.arccos(np.clip(abs(v @ dh), 0, 1)))))
            (r1, a1), (r2, a2) = both_preds(v, tr, te, truth, keep)
            s1.append(a1)
            s2.append(a2)
            r1s.append(r1)
            r2s.append(r2)
        better += int(np.mean(r2s) < np.mean(r1s))
        a = np.array(a)
        A.append(a - a.mean())
        S1.append(np.array(s1) - np.mean(s1))
        S2.append(np.array(s2) - np.mean(s2))
    A = np.concatenate(A)
    return (float(np.corrcoef(A, np.concatenate(S1))[0, 1]),
            float(np.corrcoef(A, np.concatenate(S2))[0, 1]),
            int(better), len(tags))


def ncmapss_sep(nclust=6, nstage=2, minpts=400, units=(2, 5, 10, 16, 18, 20),
                key="ncmapss"):
    """Two-way separability test on N-CMAPSS DS02.

    The operating point is held fixed by clustering flight points on
    (altitude, Mach, throttle); within a cluster each sensor is regressed on
    remaining life across cycles, so the slope is a derivative at a fixed
    operating point rather than a residual after standardisation. Each cell
    is estimated twice from interleaved halves, so the noise floor is
    measured.

    Returns (main-effect ratios and interaction ratio against their floors,
    interaction over operating main effect, number of units with a complete
    layout).
    """
    import h5py

    def load(unit):
        with h5py.File(DATA[key], "r") as f:
            A = np.array(f["A_dev"][:, :2])
            m = A[:, 0].astype(int) == unit
            idx = np.where(m)[0]
            lo, hi = idx.min(), idx.max() + 1
            keep = m[lo:hi]
            W = np.array(f["W_dev"][lo:hi])[keep]
            X = np.array(f["X_s_dev"][lo:hi])[keep]
            Y = np.array(f["Y_dev"][lo:hi])[keep].ravel().astype(float)
            cyc = A[lo:hi][keep][:, 1].astype(int)
        return W, X, Y, cyc

    def clusters(W, k, seed=0):
        Z = (W[:, :3] - W[:, :3].mean(0)) / (W[:, :3].std(0) + 1e-12)
        rng = np.random.default_rng(seed)
        C = Z[rng.choice(len(Z), k, replace=False)]
        for _ in range(25):
            lab = np.argmin(((Z[:, None, :] - C[None]) ** 2).sum(2), axis=1)
            for j in range(k):
                if (lab == j).any():
                    C[j] = Z[lab == j].mean(0)
        return lab

    def direction(X, Y):
        if len(Y) < 30 or Y.std() < 1e-9:
            return None
        y = (Y - Y.mean()) / (Y.std() + 1e-12)
        sl, rs = [], []
        for j in range(X.shape[1]):
            x = X[:, j]
            b = float(np.dot(y, x - x.mean()) / len(y))
            sl.append(b)
            rs.append(np.std(x - x.mean() - b * y) + 1e-12)
        v = np.array(sl) / np.array(rs)
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n < 1e-12:
            return None
        z = np.log(np.abs(v / n) + 1e-12)
        return z - z.mean()

    def twoway(M, ns, nc):
        keys = list(M)
        Y = np.array([M[k] for k in keys])
        g = Y.mean(0)
        a = {t: np.mean([M[k] for k in keys if k[0] == t], 0) - g
             for t in range(ns)}
        b = {c: np.mean([M[k] for k in keys if k[1] == c], 0) - g
             for c in range(nc)}
        it = np.array([M[k] - g - a[k[0]] - b[k[1]] for k in keys])
        return np.array([float(np.sum(np.array(list(b.values())) ** 2)),
                         float(np.sum(np.array(list(a.values())) ** 2)),
                         float(np.sum(it ** 2))])

    tm, td, used = np.zeros(3), np.zeros(3), 0
    for u in units:
        W, X, Y, cyc = load(u)
        lab = clusters(W, nclust)
        edges = np.linspace(cyc.min(), cyc.max() + 1, nstage + 1)
        M, N, ok = {}, {}, True
        for t in range(nstage):
            ins = (cyc >= edges[t]) & (cyc < edges[t + 1])
            for c in range(nclust):
                sel = ins & (lab == c)
                if sel.sum() < minpts:
                    ok = False
                    break
                z1 = direction(X[sel][0::2], Y[sel][0::2])
                z2 = direction(X[sel][1::2], Y[sel][1::2])
                if z1 is None or z2 is None:
                    ok = False
                    break
                M[(t, c)] = 0.5 * (z1 + z2)
                N[(t, c)] = 0.5 * (z1 - z2)
            if not ok:
                break
        if not ok or len(M) != nstage * nclust:
            continue
        tm += twoway(M, nstage, nclust)
        td += twoway(N, nstage, nclust)
        used += 1
    r = tm / np.maximum(td, 1e-12)
    return (float(r[0]), float(r[1]), float(r[2]),
            float(tm[2] / tm[0]) if tm[0] > 0 else float("nan"), int(used))


def ncmapss_sweep():
    """The interaction ratio over the clusterings the text quotes."""
    out = []
    for nc, ns in ((3, 2), (4, 2), (6, 2), (8, 2), (6, 3), (4, 4), (10, 2)):
        try:
            r = ncmapss_sep(nc, ns, minpts=300)
        except Exception:
            continue
        if r[4] > 0:
            out.append((r[2], r[3]))
    return (min(x[0] for x in out), max(x[0] for x in out),
            min(x[1] for x in out))


def cmapss_dose(fname):
    """Excess B-W regressed on the separation between regime operating points.

    A binary contrast can be blamed on anything co-varying with the regime
    label; a dose-response cannot. Returns (slope deg per unit separation,
    intercept at zero separation, pairs, quartile means of the excess).
    """
    import numpy as _np
    from scipy import stats as _st
    d, sens = _cm_load(fname)
    X = d[["op1", "op2", "op3"]].to_numpy(float)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    cen = {c: Xs[d["cond"].to_numpy() == c].mean(0)
           for c in sorted(d["cond"].unique())}
    G = {k: g.sort_values("cycle") for k, g in d.groupby(["unit", "cond"])}
    w = 7
    sep, exc = [], []
    for u in sorted({uu for uu, _ in G}):
        cs = [c for (uu, c) in G if uu == u]
        if len(cs) < 2:
            continue
        m = min(len(G[(u, c)]) for c in cs)
        if m < 2 * (w + 3):
            continue
        H = {}
        for c in cs:
            g = G[(u, c)]
            idx = _np.unique(_np.linspace(0, len(g) - 1, m)
                             .round().astype(int))
            g = g.iloc[idx]
            tau = g["tau"].to_numpy(float)
            Y = g[sens].to_numpy(float)
            a, b = _cm_dhat(tau[0::2], Y[0::2], w), \
                _cm_dhat(tau[1::2], Y[1::2], w)
            if a is not None and b is not None:
                H[c] = (a, b)
        ks = sorted(H)
        if len(ks) < 2:
            continue
        wbar = _np.mean([_cm_ang(*H[c]) for c in ks])
        for i, c in enumerate(ks):
            for c2 in ks[i + 1:]:
                sep.append(float(_np.linalg.norm(cen[c] - cen[c2])))
                exc.append(_cm_ang(H[c][0], H[c2][1]) - wbar)
    sep, exc = _np.array(sep), _np.array(exc)
    r = _st.linregress(sep, exc)
    q = _np.quantile(sep, [0, .25, .5, .75, 1.0])
    means = [float(exc[(sep >= lo) & (sep <= hi)].mean())
             for lo, hi in zip(q[:-1], q[1:])]
    return float(r.slope), float(r.intercept), len(sep), means


def downstream_rul(nind=12, nunit=160, K=30, span=0.40, seed=5):
    """Section 6.8: does a lower loss buy a lower remaining-life error?

    Returns (worst relative error of the exp(2 l) identity, Spearman
    correlation between mean loss and median absolute error, ratio of the
    worst indicator's median error to the best's).

    The estimator deliberately knows nothing about the construction: it
    matches an observed scalar history against the indicator's own reference
    curve and reads remaining life from the best-fitting offset.
    """
    X, Tf = traj(lambda x: 1.5, 4000)
    if X is None:
        return np.nan, np.nan, np.nan
    tg = np.linspace(0.0, Tf, 4000)

    def at(t):
        t = np.atleast_1d(t)
        return np.stack([np.interp(t, tg, X[:, k]) for k in range(3)], -1)

    def dv(x):
        return np.exp(A - 1.5 * E) * (1.0 + HC * x)

    dm = dv(at(0.5 * Tf)[0])
    dh = dm / np.linalg.norm(dm)
    o = np.ones(3) / np.sqrt(3)
    pp = o - (o @ dh) * dh
    pp /= np.linalg.norm(pp)

    # --- the exact identity, at several misalignments -------------------
    worst = 0.0
    gm = CFG["gain"] * float(dm @ dm)
    for pd in (0.5, 2.0, 5.0, 15.0, 40.0):
        p = np.radians(pd)
        v = np.cos(p) * dh + np.sin(p) * pp
        c2 = float(dm @ v) ** 2 / float(dm @ dm)
        ratio = (1.0 + gm) / (1.0 + gm * c2)
        lo = 0.5 * (np.log1p(gm) - np.log1p(gm * c2))
        worst = max(worst, abs(ratio - np.exp(2 * lo)) / ratio)

    # --- the practitioner's estimator ------------------------------------
    aoff = np.linspace(0.0, span * Tf, K)
    grid_t = np.linspace(0.0, 0.60 * Tf, 900)
    cand = grid_t[grid_t + aoff[-1] <= Tf]
    XC = np.stack([at(t + aoff) for t in cand])          # (nc, K, 3)
    rg = np.random.default_rng(seed)
    taus = rg.uniform(0.0, 0.45 * Tf, nunit)
    XU = np.stack([at(t + aoff) for t in taus])          # (nu, K, 3)

    ts = np.linspace(0.05 * Tf, 0.95 * Tf, 400)
    XS = at(ts)

    out = []
    for k in range(nind):
        p = np.radians(0.0 if k == 0 else 2.0 * k ** 1.5)
        v = np.abs(np.cos(p) * dh + np.sin(p) * pp)
        v /= np.linalg.norm(v)
        li = []
        for x in XS:
            dd = dv(x)
            gg = CFG["gain"] * float(dd @ dd)
            c2 = float(dd @ v) ** 2 / float(dd @ dd)
            li.append(0.5 * (np.log1p(gg) - np.log1p(gg * c2)))
        L = float(np.mean(li))
        r2 = np.random.default_rng(400 + k)
        Z = XU @ v + r2.normal(0.0, 1.0, (nunit, K))
        zc = XC @ v
        j = np.argmin(((zc[None] - Z[:, None]) ** 2).sum(axis=2), axis=1)
        e = np.abs((Tf - cand[j] - aoff[-1]) - (Tf - taus - aoff[-1]))
        out.append((L, float(np.median(e))))
    out.sort()
    Lv = np.array([a for a, _ in out])
    Mv = np.array([b for _, b in out])
    sp = float(np.corrcoef(np.argsort(np.argsort(Lv)),
                           np.argsort(np.argsort(Mv)))[0, 1])
    return worst, sp, float(Mv.max() / Mv.min())


def threshold_sensitivity(xf=1.0):
    """How much of Section 6 rides on the condemnation limit x_f = 0.9?

    Section 6.1 quotes what happens at x_f = 1, and those numbers need to be
    computed the way the 0.9 ones are or the comparison is between two
    conventions rather than two thresholds. So this reuses greedy(),
    best_constant() and beta_min_range() unchanged -- same 41-point control
    grid, same integrator tolerances, same loss -- and moves only XFAIL.

    XFAIL is module state that traj() closes over, so it is restored in a
    finally: an exception here would otherwise leave every later check
    running against the wrong threshold, and they would still all pass.

    Returns (peak floor in degrees, adaptive gain, lifetime change in per
    cent) at the alternative threshold.
    """
    global XFAIL
    keep = XFAIL
    try:
        _, Tf_here = traj(lambda x: 1.5)
        XFAIL = float(xf)
        r = minimize(lambda p: greedy(p, fast=True)[0], QS.copy(),
                     method="Nelder-Mead",
                     options=dict(xatol=1e-5, fatol=1e-9, maxiter=800))
        q1 = np.abs(r.x)
        q1 = q1 / np.linalg.norm(q1)
        L_ad, _, X1 = greedy(q1)
        L_co, _ = best_constant(q1)
        _, peak = beta_min_range(q1, X1)
        _, Tf_there = traj(lambda x: 1.5)
    finally:
        XFAIL = keep
    assert XFAIL == keep, "threshold not restored"
    return peak, L_co / L_ad, 100.0 * (Tf_there - Tf_here) / Tf_here


def record_identity(ntraj=12, seed=5):
    """Proposition 2.3, checked against a direct computation.

    The proposition claims that the information a fixed indicator forfeits
    over a WHOLE record is -1/2 log(1 - W S), with G the summed SNR, W =
    G/(1+G), and S the ||d||^2-weighted mean of sin^2(psi). That is a
    different quantity from the time-averaged instantaneous loss the rest of
    the paper optimises, so it needs its own check rather than inheriting
    one.

    Compares the closed form against the difference of the two mutual
    informations computed straight from the Fisher sums. Returns the worst
    absolute discrepancy over random indicators and operating points, and
    the worst gap between the record loss and the time-averaged one -- the
    second is not an error, it is the size of the distinction the
    proposition draws, and a row asserting it is LARGE keeps the two from
    being quietly conflated again.
    """
    rng = np.random.default_rng(seed)
    worst_id, biggest_gap = 0.0, 0.0
    for _ in range(ntraj):
        u = rng.uniform(UMIN, UMAX)
        X = _traj_E(E, lambda x, _u=u: _u, n=400)
        if X is None:
            continue
        D = np.array([np.exp(A - u * E) * (1.0 + HC * x) for x in X])
        nrm2 = (D ** 2).sum(axis=1)
        v = np.abs(rng.normal(size=3)) + 1e-3
        v /= np.linalg.norm(v)

        G = CFG["gain"] * nrm2.sum()
        W = G / (1.0 + G)
        cos2 = (D @ v) ** 2 / nrm2
        S = (nrm2 * (1.0 - cos2)).sum() / nrm2.sum()

        direct = 0.5 * (np.log1p(G)
                        - np.log1p(CFG["gain"] * ((D @ v) ** 2).sum()))
        closed = -0.5 * np.log(1.0 - W * S)
        worst_id = max(worst_id, abs(direct - closed))

        g = CFG["gain"] * nrm2
        tavg = (0.5 * (np.log1p(g) - np.log1p(g * cos2))).mean()
        biggest_gap = max(biggest_gap, abs(tavg - direct) / max(direct, 1e-30))
    return worst_id, biggest_gap


# ------------------------------------------- two inputs: m = d-1 ---------
G2 = np.array([0.9, 0.3, 1.2])          # second input's sensitivity pattern


def _traj_E(Ev, law, n=800, rtol=1e-9):
    """Trajectory of the plant whose activation energies are Ev."""
    def rhs(t, x):
        return np.exp(A - law(x) * Ev) * (1.0 + HC * x)

    def ev(t, x):
        return x.max() - XFAIL
    ev.terminal, ev.direction = True, 1
    sol = solve_ivp(rhs, [0, TMAX], np.full(3, X0), events=ev,
                    rtol=rtol, atol=rtol * 1e-2, dense_output=True)
    if not sol.t_events[0].size:
        return None
    return sol.sol(np.linspace(0, sol.t[-1], n)).T


def two_input_floor(n=600, ngrid=90):
    """The 'if' half of Theorem 5.1: at m = d-1 the floor should vanish.

    With phi_i(u) = exp(a_i - E_i u1 - G_i u2), log d_i is affine in (u1,u2),
    so matching a target direction is d-1 = 2 linear equations in 2 unknowns.
    Returns (det, states inside the box, states tested, worst residual
    misalignment in deg, worst grid-search floor in deg).
    """
    def phi(u1, u2):
        return np.exp(A - u1 * E - u2 * G2)

    def dh(x, u1, u2):
        d = phi(u1, u2) * (1.0 + HC * x)
        return d / np.linalg.norm(d)

    def rhs(t, x):
        return phi(1.2, 1.2) * (1.0 + HC * x)

    def ev(t, x):
        return x.max() - XFAIL
    ev.terminal, ev.direction = True, 1
    sol = solve_ivp(rhs, [0, TMAX], np.full(3, X0), events=ev,
                    rtol=1e-10, atol=1e-12, dense_output=True)
    X = sol.sol(np.linspace(0, sol.t[-1], n)).T
    q = dh(X[n // 2], 1.2, 1.2)

    M = np.array([[-(E[0] - E[2]), -(G2[0] - G2[2])],
                  [-(E[1] - E[2]), -(G2[1] - G2[2])]])
    det = float(np.linalg.det(M))

    us = np.linspace(UMIN, UMAX, ngrid)
    inside, worst_res, worst_floor = 0, 0.0, 0.0
    for x in X:
        lh, lq = np.log(1.0 + HC * x), np.log(q)
        r = np.array([(lq[0] - lq[2]) - (A[0] - A[2]) - (lh[0] - lh[2]),
                      (lq[1] - lq[2]) - (A[1] - A[2]) - (lh[1] - lh[2])])
        u1, u2 = np.linalg.solve(M, r)
        if UMIN <= u1 <= UMAX and UMIN <= u2 <= UMAX:
            inside += 1
            worst_res = max(worst_res, float(np.degrees(np.arccos(
                np.clip(dh(x, u1, u2) @ q, -1, 1)))))
        best = np.inf
        for a1 in us:
            D = np.array([dh(x, a1, a2) for a2 in us])
            best = min(best, float(np.degrees(
                np.arccos(np.clip(D @ q, -1, 1))).min()))
        worst_floor = max(worst_floor, best)
    return det, inside, n, worst_res, worst_floor


# ------------------------------------------- model misspecification ------
def _indicator_from(Ev, u0=1.2):
    """Barycentre indicator of Prop. 4.3, built on the plant believed to
    have activation energies Ev."""
    X = _traj_E(Ev, lambda x: u0)
    D = np.exp(A - u0 * Ev)[None, :] * (1.0 + HC[None, :] * X)
    nr = np.linalg.norm(D, axis=1)
    g = CFG["gain"] * nr ** 2
    w = g / (1.0 + g)
    return karcher(D / nr[:, None], w / w.sum())


def _loss_on_true(q):
    """Closed-loop loss of indicator q on the plant with the TRUE energies."""
    q = np.abs(q) / np.linalg.norm(q)
    US = np.linspace(UMIN, UMAX, CFG["NU"])
    PHI = np.exp(A[None, :] - US[:, None] * E[None, :])

    def best(x):
        D = PHI * (1.0 + HC[None, :] * x[None, :])
        D = D / np.linalg.norm(D, axis=1, keepdims=True)
        b = np.arccos(np.clip(D @ q, -1, 1))
        j = int(np.argmin(b))
        return US[j], float(b[j])

    X = _traj_E(E, lambda x: best(x)[0])
    if X is None:
        return np.inf
    return float(np.mean(ell(np.array([best(x)[1] for x in X]))))


def misspec_table(levels=(0.02, 0.05, 0.10, 0.20, 0.35), draws=40, seed=11):
    """{eps: (median, 90th percentile, worst)} loss ratio against a design
    built from the true model. Seeded, so the table is reproducible."""
    rng = np.random.default_rng(seed)
    ref = _loss_on_true(_indicator_from(E))
    out = {}
    for eps in levels:
        ratios = []
        for _ in range(draws):
            Eb = E * (1.0 + eps * rng.standard_normal(3))
            if (Eb <= 0).any():
                continue
            ratios.append(_loss_on_true(_indicator_from(Eb)) / ref)
        a = np.array(ratios)
        out[eps] = (float(np.median(a)), float(np.percentile(a, 90)),
                    float(a.max()))
    return out


# --------------------------------------------------------- Table II ------
REGIME = [np.array([0.60, 1.10, 0.80]),
          np.array([0.80, 0.82, 0.79]),
          np.array([0.80, 0.805, 0.798])]


def regime_row(Ev, restarts=8, seed=7):
    """arc, rho, best adaptive loss and the gain over the best constant."""
    global E
    keep, E = E, Ev
    try:
        US, PHI = grid(CFG["NU"])

        def dirs(x):
            D = PHI * (1.0 + HC[None, :] * x[None, :])
            return D / np.linalg.norm(D, axis=1, keepdims=True)

        Xn, _ = traj(lambda x: 1.5, 400)
        arcs, mids = [], []
        for x in Xn[::20]:
            Dm = dirs(x)
            arcs.append(np.degrees(np.arccos(np.clip(Dm[0] @ Dm[-1], -1, 1))))
            m = Dm.mean(0)
            mids.append(m / np.linalg.norm(m))
        mids = np.array(mids)
        arc = float(np.mean(arcs))
        drift = float(np.degrees(np.arccos(np.clip(mids[0] @ mids[-1], -1, 1))))
        rng = np.random.default_rng(seed)
        best = (np.inf, None)
        for _ in range(restarts):
            r = minimize(lambda v: greedy(v, fast=True)[0],
                         np.abs(rng.normal(0, 1, 3)) + 0.05,
                         method="Nelder-Mead",
                         options=dict(maxiter=90, xatol=1e-4, fatol=1e-9))
            if np.isfinite(r.fun) and r.fun < best[0]:
                best = (float(r.fun), np.abs(r.x) / np.linalg.norm(r.x))
        q_b = best[1]
        L_ad = greedy(q_b)[0]          # winner, full accuracy
        L_co, _ = best_constant(q_b)
        return arc, drift / arc, L_ad, L_co / L_ad
    finally:
        E = keep


# ----------------------------------------------------------- Section 7 ---
# Data lives outside the repo: the six fleets are public but total about
# 8.5 GB, and redistributing derived features from someone else's dataset
# is not ours to do. Point RESS_DATA at a directory laid out as in
# DATA.md, or edit the paths below. Anything missing is SKIPPED, loudly.
_RD = os.environ.get("RESS_DATA",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "data"))
DATA = dict(
    severson=os.path.join(_RD, "severson_cells.npz"),
    pronostia=os.path.join(_RD, "pronostia_cells.npz"),
    xjtu=os.path.join(_RD, "xjtu_features.npz"),
    nasa=os.path.join(_RD, "cv_features_dataset.csv"),
    cmapss=os.path.join(_RD, "CMAPSSData") + os.sep,
    # phm2012() is defined but never called -- the call site went when
    # Section 7 was compressed. Kept so the analysis can be restored;
    # you do NOT need the 3.2 GB PHM 2012 archive to reproduce this paper.
    phm2012=os.path.join(_RD, "phm-ieee-2012-data-challenge-dataset-master"),
    ncmapss=os.path.join(_RD, "N-CMAPSS_DS02-006.h5"),
    ncmapss01=os.path.join(_RD, "N-CMAPSS_DS01-005.h5"),
)

# ---------------------------------------------------------------- C-MAPSS --
# The actuation premise. C-MAPSS is the only public fleet whose operating
# point varies WITHIN a unit, so an engine can serve as its own control and
# the between-unit confound that defeats the other four fleets cannot arise.
#
#   W = angle between dhat from two interleaved halves of ONE regime
#   B = angle between dhat from halves of DIFFERENT regimes
#
# Both use disjoint half-samples of equal size, so they share an estimator
# noise floor; under no actuation effect E[B] = E[W]. Every regime of an
# engine is resampled to a common count, and the Savitzky-Golay window is
# fixed rather than derived from n, so neither arm can gain an advantage from
# sample size. The same machinery run on FAKE regimes cut from a single real
# one measures the residual bias of the procedure itself.
CM_COLS = (["unit", "cycle"] + [f"op{i}" for i in (1, 2, 3)]
           + [f"s{i}" for i in range(1, 22)])
CM_GRID = np.linspace(0.30, 0.85, 10)


def _cm_load(fname, guard_constant=False):
    import os
    import pandas as pd
    d = pd.read_csv(os.path.join(DATA["cmapss"], fname), sep=r"\s+",
                    header=None, names=CM_COLS)
    d["tau"] = d["cycle"] / d.groupby("unit")["cycle"].transform("max")
    X = d[["op1", "op2", "op3"]].to_numpy(float)
    if guard_constant and not ((X.max(0) - X.min(0)) > 0.1).any():
        # a subset with one operating condition: standardising here would
        # divide by a near-zero std and turn numerical jitter into hundreds
        # of spurious regimes
        d["cond"] = 0
    else:
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
        _, d["cond"] = np.unique(np.round(Xs, 1), axis=0,
                                 return_inverse=True)
    sens = [c for c in CM_COLS if c.startswith("s")]
    sd = d.groupby(["unit", "cond"])[sens].std().median()
    sens = [c for c in sens if sd[c] > 1e-9]          # live within a regime
    g = d.groupby("cond")
    for c in sens:                                     # z-score per regime
        d[c] = (d[c] - g[c].transform("mean")) / g[c].transform("std")
    return d.dropna(subset=sens), sens


def _cm_dhat(tau, Y, w):
    from scipy.signal import savgol_filter
    if len(tau) < w + 3:
        return None
    dt = np.gradient(tau).mean()
    D, S = [], []
    for j in range(Y.shape[1]):
        y = Y[:, j]
        D.append(savgol_filter(y, w, 2, deriv=1) / dt)
        S.append(np.std(y - savgol_filter(y, w, 2)) + 1e-12)
    M = np.stack(D, 1) / np.array(S)[None, :]
    if not np.isfinite(M).all():
        return None
    V = np.stack([np.interp(CM_GRID, tau, M[:, j])
                  for j in range(M.shape[1])], 1)
    nv = np.linalg.norm(V, axis=1, keepdims=True)
    if (nv < 1e-12).any():
        return None
    return V / nv


def _cm_ang(a, b):
    return float(np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1))).mean())


def cmapss_stats(fname, windows=(7, 9, 11)):
    """{'real': {w: (mean B-W, npos, n)}, 'ctrl': {w: (|B-W|, p)}}"""
    import os
    from scipy import stats as _st
    if not os.path.isdir(DATA["cmapss"]):
        return None
    d, sens = _cm_load(fname)
    G = {k: g.sort_values("cycle") for k, g in d.groupby(["unit", "cond"])}
    units = sorted({u for u, _ in G})
    out = dict(real={}, ctrl={}, nregime=int(d["cond"].nunique()))
    for w in windows:
        real, ctrl = [], []
        for u in units:
            cs = [c for (uu, c) in G if uu == u]

            # --- real test, every regime resampled to a common count -----
            if len(cs) >= 2:
                m = min(len(G[(u, c)]) for c in cs)
                if m >= 2 * (w + 3):
                    H = {}
                    for c in cs:
                        g = G[(u, c)]
                        idx = np.unique(np.linspace(0, len(g) - 1, m)
                                        .round().astype(int))
                        g = g.iloc[idx]
                        tau = g["tau"].to_numpy(float)
                        Y = g[sens].to_numpy(float)
                        a = _cm_dhat(tau[0::2], Y[0::2], w)
                        b = _cm_dhat(tau[1::2], Y[1::2], w)
                        if a is not None and b is not None:
                            H[c] = (a, b)
                    ks = sorted(H)
                    if len(ks) >= 2:
                        ww = [_cm_ang(*H[c]) for c in ks]
                        bb = [_cm_ang(H[c][0], H[c2][1])
                              for i, c in enumerate(ks) for c2 in ks[i + 1:]]
                        real.append((np.mean(ww), np.mean(bb)))

            # --- negative control: fake regimes cut from one real regime --
            wf, bf = [], []
            for c in cs:
                g = G[(u, c)]
                if len(g) < 4 * (w + 3):
                    continue
                tau = g["tau"].to_numpy(float)
                Y = g[sens].to_numpy(float)
                q = [_cm_dhat(tau[i::4], Y[i::4], w) for i in range(4)]
                if any(x is None for x in q):
                    continue
                wf.append(_cm_ang(q[0], q[1]))       # fake A against itself
                bf.append(_cm_ang(q[0], q[3]))       # fake A against fake B
            if wf:
                ctrl.append((np.mean(wf), np.mean(bf)))

        R = np.array(real)
        C = np.array(ctrl)
        dr = R[:, 1] - R[:, 0]
        dc = C[:, 1] - C[:, 0]
        out["real"][w] = (float(dr.mean()), int((dr > 0).sum()), len(dr))
        out["ctrl"][w] = (float(abs(dc.mean())),
                          float(_st.ttest_rel(C[:, 1], C[:, 0]).pvalue))
    return out


def _dirs_from(mat, win, grid_):
    """whitened direction on a tau grid from an (n, p) feature record."""
    from scipy.signal import savgol_filter
    a = np.asarray(mat, dtype=float)
    n, pdim = a.shape
    if n < 40 or not np.isfinite(a).all():
        return None
    w = min(win, (n // 3) * 2 + 1)
    if w < 7:
        return None
    tau = np.arange(n) / (n - 1.0)
    D, S = [], []
    for j in range(pdim):
        y = a[:, j]
        if np.std(y) < 1e-12:
            return None
        D.append(savgol_filter(y, w, 2, deriv=1) * (n - 1))
        S.append(np.std(y - savgol_filter(y, w, 2)) + 1e-12)
    D = np.stack(D, 1) / np.array(S)[None, :]
    return np.stack([np.interp(grid_, tau, D[:, k]) for k in range(pdim)], 1)


def _orient(R):
    sg = np.sign(np.median(R.reshape(-1, R.shape[2]), axis=0))
    sg[sg == 0] = 1.0
    D = R * sg[None, None, :]
    return D / np.linalg.norm(D, axis=2, keepdims=True)


def _travel(DH):
    M = DH.mean(0)
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    return float(np.degrees(np.arccos(np.clip((M[:-1] * M[1:]).sum(1),
                                              -1, 1))).sum())


def _perm_p(DH, lab, nperm=3000, seed=3):
    """permutation test: is the between-condition spread bigger than chance?"""
    rng = np.random.default_rng(seed)
    conds = sorted(set(lab))

    def stat(labels):
        M = []
        for c in conds:
            m = DH[labels == c].mean(0)
            M.append(m / np.linalg.norm(m, axis=1, keepdims=True))
        s = [np.degrees(np.arccos(np.clip((M[i] * M[j]).sum(1), -1, 1))).mean()
             for i in range(len(conds)) for j in range(i + 1, len(conds))]
        return float(np.mean(s))
    obs = stat(lab)
    null = np.array([stat(rng.permutation(lab)) for _ in range(nperm)])
    return float((null >= obs).mean()), obs


def xjtu_stats(win=21):
    import os
    if not os.path.exists(DATA["xjtu"]):
        return None
    Z = np.load(DATA["xjtu"], allow_pickle=True)
    names = sorted({k.split("__")[0] for k in Z.keys()
                    if k.startswith("Bearing")})
    G = np.linspace(0.20, 0.90, 30)
    D, lab = [], []
    for b in names:
        d = _dirs_from(Z[b + "__X"], win, G)
        if d is not None and np.isfinite(d).all():
            D.append(d)
            lab.append(str(Z[b + "__cond"]))
    DH = _orient(np.array(D))
    lab = np.array(lab)
    p, _ = _perm_p(DH, lab)
    sizes = sorted({int((lab == c).sum()) for c in set(lab)})
    return dict(p=p, travel=_travel(DH), n=len(DH),
                ncond=len(set(lab)), sizes=sizes)


def pronostia_stats(win=201):
    import os
    if not os.path.exists(DATA["pronostia"]):
        return None
    Z = np.load(DATA["pronostia"], allow_pickle=True)
    G = np.linspace(0.20, 0.90, 30)
    D, lab = [], []
    for k, nm in enumerate(Z["names"]):
        d = _dirs_from(np.asarray(Z["data"][k], dtype=float), win, G)
        if d is not None and np.isfinite(d).all():
            D.append(d)
            lab.append("c" + str(nm)[7])
    DH = _orient(np.array(D))
    p, _ = _perm_p(DH, np.array(lab))
    return dict(p=p, travel=_travel(DH), n=len(DH))


def nasa_stats(win=21):
    import os
    if not os.path.exists(DATA["nasa"]):
        return None
    import pandas as pd
    df = pd.read_csv(DATA["nasa"])
    cols = ["t_CV", "tau", "area_CV"]
    G = np.linspace(0.15, 0.90, 30)
    D = []
    for _, g in df.groupby("battery"):
        d = _dirs_from(g.sort_values("cycle_seq")[cols].to_numpy(float),
                       win, G)
        if d is not None and np.isfinite(d).all():
            D.append(d)
    DH = _orient(np.array(D))
    per = [float(np.degrees(np.arccos(np.clip((DH[i, :-1] * DH[i, 1:]).sum(1),
                                              -1, 1))).sum())
           for i in range(len(DH))]
    M = DH.mean(0)
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    return dict(n=len(DH), per_lo=min(per), per_hi=max(per),
                allpos=float((M > 0).all(axis=1).mean()),
                travel=_travel(DH))


def severson_stats(win):
    import os
    if not os.path.exists(DATA["severson"]):
        return None
    from scipy.signal import savgol_filter
    Z = np.load(DATA["severson"], allow_pickle=True)
    CH = [str(c) for c in Z["channels"]]
    USE = [CH.index("QDischarge"), CH.index("IR"), CH.index("Tavg")]
    G = np.linspace(0.15, 0.90, 40)
    rows = []
    for k in range(len(Z["data"])):
        if not np.isfinite(Z["lives"][k]):
            continue
        aa = np.asarray(Z["data"][k], dtype=float)
        n = len(aa)
        if n < 400 or not np.isfinite(aa[:, USE]).all():
            continue
        tau = np.arange(n) / (n - 1.0)
        w = min(win, (n // 4) * 2 + 1)
        Dl, Sl = [], []
        for j in USE:
            y = aa[:, j]
            Dl.append(savgol_filter(y, w, 3, deriv=1) * (n - 1))
            Sl.append(np.std(y - savgol_filter(y, w, 3)) + 1e-12)
        Dm = np.stack(Dl, 1) / np.array(Sl)[None, :]
        rows.append(np.stack([np.interp(G, tau, Dm[:, c])
                              for c in range(3)], 1))
    R = np.array(rows)
    sg = np.sign(np.median(R.reshape(-1, 3), axis=0))
    DH = R * sg[None, None, :]
    DH = DH / np.linalg.norm(DH, axis=2, keepdims=True)
    M = DH.mean(0)
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    travel = float(np.degrees(np.arccos(np.clip((M[:-1] * M[1:]).sum(1),
                                                -1, 1))).sum())
    e2e = float(np.degrees(np.arccos(np.clip(M[0] @ M[-1], -1, 1))))
    P = M ** 2
    mid = P[len(P) // 2]
    return dict(n=len(R), travel=travel, e2e=e2e,
                allpos=float((M > 0).all(axis=1).mean()),
                shares=mid)


def dinkelbach_limit():
    """Richardson-extrapolated fractional optimum at q*; slow."""
    def solve_F(lam, N):
        ax = np.linspace(X0, XFAIL, N)
        g = [ax] * 3
        P = np.stack([m.ravel() for m in np.meshgrid(*g, indexing="ij")], 1)
        h = 1.6 * (ax[1] - ax[0])
        US, _ = grid(CFG["NU"])
        term = P.max(1) >= XFAIL - 1e-12
        ST, TI, HI, RU = [], [], [], []
        for u in US:
            f = np.exp(A - u * E)[None, :] * (1.0 + HC[None, :] * P)
            nf = np.linalg.norm(f, axis=1, keepdims=True)
            gg = f / nf
            with np.errstate(divide="ignore", invalid="ignore"):
                si = (XFAIL - P) / np.where(gg > 1e-15, gg, np.inf)
            st = np.minimum(h, np.min(np.where(si > 0, si, np.inf), axis=1))
            ST.append(P + st[:, None] * gg)
            TI.append(st / nf[:, 0])
            HI.append(st < h - 1e-15)
            RU.append(ell(np.arccos(np.clip(gg @ QS, -1, 1)),
                          nf[:, 0] ** 2))
        RU = np.stack(RU)
        W = np.zeros(len(P))
        for _ in range(500):
            it = RegularGridInterpolator(g, W.reshape(N, N, N),
                                         bounds_error=False, fill_value=None)
            Wn = np.stack([TI[j] * (RU[j] - lam)
                           + np.where(HI[j], 0.0, it(ST[j]))
                           for j in range(len(US))]).min(axis=0)
            Wn[term] = 0.0
            if np.max(np.abs(Wn - W)) < 1e-13:
                W = Wn
                break
            W = Wn
        it = RegularGridInterpolator(g, W.reshape(N, N, N),
                                     bounds_error=False, fill_value=None)
        return float(it(np.full((1, 3), X0))[0])

    vals = []
    for N in CFG["Ngrid"]:
        lo, hi = 0.0, 0.01
        for _ in range(26):
            mid = 0.5 * (lo + hi)
            if solve_F(mid, N) > 0:
                lo = mid
            else:
                hi = mid
        vals.append(0.5 * (lo + hi))
        print(f"   Dinkelbach N={N}: {vals[-1]:.8f}", flush=True)
    v, Na = np.array(vals), np.array(CFG["Ngrid"], float)
    return v[-2] - (v[-2] - v[-1]) / (1 / Na[-2] - 1 / Na[-1]) / Na[-2]


if __name__ == "__main__":
    sys.exit(main())
