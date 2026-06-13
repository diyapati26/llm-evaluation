"""wrong_answer.py — seeded, plausibility-recorded distractor selection.

Fixes the old deterministic `(correct % 4) + 1` cycle (a confound: the offered
wrong answer was a fixed function of the correct one). Here the offered
distractor is chosen by a seeded RNG keyed on the item, and a lexical
plausibility proxy is recorded so analysis can control for "how confusable" the
distractor was.

plausibility: difflib ratio between the correct choice text and the offered
distractor text (0-1). It's a lexical proxy — a semantic (embedding) version can
replace it later without changing the interface. Recorded as a column either way.
"""
from __future__ import annotations

import difflib
import random


def pick_wrong(correct_answer, choices, seed, item_id) -> tuple[str | None, float | None]:
    """Return (offered_answer_1indexed_str, distractor_plausibility) or (None, None).

    Seeded by (seed, item_id) so the choice is reproducible and item-specific.
    """
    try:
        c = int(correct_answer)
    except (TypeError, ValueError):
        return None, None

    n = len(choices) if choices else 4
    wrong_idxs = [i for i in range(1, n + 1) if i != c]
    if not wrong_idxs:
        return None, None

    offered = random.Random(f"{seed}:wrong:{item_id}").choice(wrong_idxs)

    plausibility = None
    if choices and 1 <= c <= len(choices) and 1 <= offered <= len(choices):
        plausibility = round(
            difflib.SequenceMatcher(None, str(choices[c - 1]), str(choices[offered - 1])).ratio(), 4
        )
    return str(offered), plausibility
