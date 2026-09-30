"""hiloss: information loss of fixed linear health indicators.

Code for "Information loss of fixed linear health indicators under competing
degradation mechanisms and variable operating conditions" (H. H. Le and
K.-A. Nguyen, Reliability Engineering & System Safety).

    core        closed forms: loss, variance and margin inflation, arc bound
    direction   steps 1-3 on run-to-failure data: observed direction, arc,
                bound, fixed indicator
    rul         step 4: leave-one-out similarity-based remaining-life estimator
    policy      step 5: predictive replacement policy and its long-run cost
    synthetic   the synthetic system of Section 5 (steering, Tables 2 and 3)
    fleets      Section 6: the public run-to-failure fleets (Table 4)
    datasets    where the data live; the Severson reader and extractor
"""
from . import core, direction, rul, policy, datasets
from .core import (loss, variance_inflation, margin_inflation, arc_length,
                   arc_bound, barycentre)
from .direction import observed_direction, FleetDirection

__version__ = "1.0.0"
