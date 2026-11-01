"""Measured, context-local ADP preference construction for V4b.

Pairs are created only from complete authenticated assignments in one
program/device/clock/budget context. Budget views reuse physical measurements.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def _area(row):
    return float(np.mean(np.asarray(row['physical'][1:], dtype=np.float64) /
                         np.asarray(row['capacities'], dtype=np.float64)))


def _adp(row):
    return float(row['physical'][0]) * _area(row)


def preference_pairs(records, budget_fraction, meaningful_gap=0.05):
    """Return ``(preferred_index, worse_index, reason)`` ADP pairs.

    One feasible candidate wins over an infeasible one. Both-infeasible and
    near-tie pairs are omitted. The known Viterbi/U200/5ns zero-area records
    are excluded from ADP orderings because the original CSV precision cannot
    establish whether those physical resource counts are truly zero.
    """
    if not 0.0 <= float(budget_fraction) <= 1.0:
        raise ValueError('budget_fraction must be in [0,1]')
    if not 0.0 <= float(meaningful_gap) < 1.0:
        raise ValueError('meaningful_gap must be in [0,1)')
    if not records:
        return [], {'feasible_infeasible': 0, 'adp': 0, 'near_tie': 0,
                    'both_infeasible': 0, 'zero_area_ambiguous': 0}
    capacities = np.asarray(records[0]['capacities'], dtype=np.float64)
    if capacities.shape != (4,) or np.any(capacities <= 0):
        raise ValueError('Expected positive [BRAM,DSP,FF,LUT] capacities')
    if any(not np.array_equal(np.asarray(row['capacities'], dtype=np.float64), capacities)
           for row in records):
        raise ValueError('Preference pool mixes physical device capacities')
    feasible = [bool(np.all(np.asarray(row['physical'][1:], dtype=np.float64)
                            <= capacities * budget_fraction)) for row in records]
    pairs = []
    counts = Counter()
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if any(_ambiguous_zero_area(records[k]) for k in (i, j)):
                # The source CSV's zero resource fields determine both area
                # ordering and feasibility at a requested budget; do not let
                # that ambiguity create either kind of preference label.
                counts['zero_area_ambiguous'] += 1
                continue
            if feasible[i] != feasible[j]:
                pairs.append((i, j, 'feasible_infeasible') if feasible[i]
                             else (j, i, 'feasible_infeasible'))
                counts['feasible_infeasible'] += 1
                continue
            if not feasible[i]:
                counts['both_infeasible'] += 1
                continue
            a, b = _adp(records[i]), _adp(records[j])
            relative_gap = abs(a - b) / max(abs(a), abs(b), 1e-12)
            if relative_gap <= meaningful_gap:
                counts['near_tie'] += 1
            else:
                pairs.append((i, j, 'adp') if a < b else (j, i, 'adp'))
                counts['adp'] += 1
    categories = ('feasible_infeasible', 'adp', 'near_tie',
                  'both_infeasible', 'zero_area_ambiguous')
    return pairs, {name: int(counts[name]) for name in categories}


def _ambiguous_zero_area(row):
    return (row.get('kernel') == 'machsuite-viterbi' and
            row.get('profile') == 'u200' and float(row.get('clock', -1)) == 5.0 and
            _area(row) == 0.0)

