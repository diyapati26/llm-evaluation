"""order.py — counterbalanced stateful attack order.

The old stateful chain used a FIXED order (confident_assertion always first), so
an attack could only be a "killing blow" on the residual of samples that
survived everything before it — confounding potency with position. Here each
item gets a seeded shuffle of the pressure attacks, so order is counterbalanced
across items and the killing-blow attribution is no longer position-biased.
Reproducible: same (seed, item_id) -> same order.
"""
from __future__ import annotations

import random

from latest.plan.arms import PRESSURE_ATTACKS


def stateful_order(item_id: str, seed: int) -> list[str]:
    order = list(PRESSURE_ATTACKS)
    random.Random(f"{seed}:order:{item_id}").shuffle(order)
    return order


def variant_order(item_id: str, attack: str, seed: int, n_variants: int) -> list[int]:
    """Mode 1 (repeat): counterbalanced order of variant indices for one attack
    type, so a fold isn't blamed on whichever phrasing happens to come last.
    Reproducible per (item, attack)."""
    order = list(range(n_variants))
    random.Random(f"{seed}:variant:{item_id}:{attack}").shuffle(order)
    return order


def gauntlet_order(item_id: str, seed: int, attacks_with_variants: list[tuple[str, int]]) -> list[list]:
    """Mode 2 (gauntlet): one fully counterbalanced sequence of [attack, variant]
    steps across every pressure attack x every variant. Reproducible per item.
    Returned as lists (not tuples) so it survives JSON/Parquet round-trips intact."""
    pairs = [[a, v] for a, n in attacks_with_variants for v in range(n)]
    random.Random(f"{seed}:gauntlet:{item_id}").shuffle(pairs)
    return pairs
