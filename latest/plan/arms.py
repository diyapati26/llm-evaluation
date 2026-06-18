"""arms.py — attack taxonomy + experimental arms.

Arms (corrigibility dropped per project decision):
  - pressure_wrong : attacker asserts a WRONG answer (the manipulation test)
  - control        : neutral_control, no-pressure rephrasing (natural-drift baseline)

The arm is derived from the attack type (control == neutral_control) and stored
as a column so the analysis layer can group pressure vs. baseline cleanly.
"""
from __future__ import annotations

# The 7 social-pressure attacks (order here is the canonical list; the per-item
# stateful order is counterbalanced separately in order.py).
PRESSURE_ATTACKS = [
    "confident_assertion",
    "authority_pressure",
    "emotional_pressure",
    "false_consensus",
    "false_evidence",
    "identity_challenge",
    "concession_override",
]

CONTROL_ATTACK = "neutral_control"           # no-pressure baseline (natural drift)
DRIFT_SINGLE_ATTACK = "incremental_drift"    # single-turn variant in the attacks file
DRIFT_SEQUENCE_ATTACK = "incremental_drift_6_turn"  # the 6-turn drift sequence
STATEFUL_ATTACK = "stateful_chain"           # all attack TYPES chained (variant 0 each)
GAUNTLET_ATTACK = "gauntlet_chain"           # all attack types x ALL variants chained (Mode 2)

ARMS = ["pressure_wrong", "control"]


def arm_for_attack(attack: str) -> str:
    return "control" if attack == CONTROL_ATTACK else "pressure_wrong"
