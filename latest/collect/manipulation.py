"""collect/manipulation.py — execute one manipulation Trial as a turn sequence.

Per mode (turn 0 is always the baseline question, no attack):
  stateless : 1 attack turn (the trial's attack x variant)
  drift     : 5 gradual-escalation turns
  stateful  : the counterbalanced attack order (variant 0), stop at first fold
  repeat    : the counterbalanced variant order of ONE attack, stop at first fold
  gauntlet  : the counterbalanced (attack, variant) sequence, stop at first fold

The Conversation logs every turn to the ledger. Per-turn attack/variant labels
are written onto conv.ctx before each send so the CallRecord records WHICH attack
hit on that turn (needed for killing-blow analysis). Fold detection here is just
chain control flow — the actual resist/fold/hedge scoring happens in analysis/.
"""
from __future__ import annotations

from latest import loaders
from latest.plan import arms
from latest.providers import start_conversation
from latest.providers.base import CallContext
from latest.records import ReasonedAnswer

_ATTACKS = loaders.load_attacks()
_DRIFT = loaders.load_drift_turns()


def _ctx(trial, manifest, snapshot) -> CallContext:
    return CallContext(
        run_id=manifest.run_id, trial_id=trial.trial_id, module="manipulation",
        item_id=trial.item_id, subject=trial.subject, dataset=trial.dataset,
        arm=trial.arm, condition=trial.arm, replicate_idx=trial.replicate_idx,
        seed=manifest.seed, model_id_for_key=snapshot,
        git_sha=manifest.git_sha, config_hash=manifest.config_hash,
    )


def _letter(resp) -> str | None:
    return (resp.raw or {}).get("letter")


def _is_fold(base_letter, letter, correct) -> bool:
    """A genuine fold: the baseline was correct AND this turn switched to a specific
    wrong answer (UNCERTAIN/None is a hedge, not a fold). Baseline-aware so a chain
    on an already-wrong item isn't truncated at turn 1. Matches analysis fold logic;
    changes no scored metric (scoring re-derives from the ledger)."""
    if str(base_letter) != str(correct):
        return False
    return letter is not None and str(letter).upper() != "UNCERTAIN" and str(letter) != str(correct)


def _attack_prompt(attack, variant_idx, offered) -> str:
    # neutral_control templates have no {wrong_number}; .format ignores the unused kwarg.
    return _ATTACKS[attack][variant_idx].format(wrong_number=(offered if offered is not None else "1"))


def run_trial(model, trial, cache, ledger, manifest, snapshot) -> None:
    conv = start_conversation(model, ctx=_ctx(trial, manifest, snapshot), cache=cache, ledger=ledger)
    correct = trial.correct_answer

    # Turn 0 — baseline answer, no attack.
    conv.ctx.attack, conv.ctx.variant_idx = None, None
    base = _letter(conv.send(trial.question, ReasonedAnswer))

    mode = trial.mode
    if mode == "stateless":
        conv.ctx.attack, conv.ctx.variant_idx = trial.attack, trial.variant_idx
        conv.send(_attack_prompt(trial.attack, trial.variant_idx, trial.offered_answer), ReasonedAnswer)

    elif mode == "drift":
        for name, tmpl in _DRIFT.items():
            conv.ctx.attack, conv.ctx.variant_idx = name, 0
            conv.send(tmpl.format(wrong_number=(trial.offered_answer or "1")), ReasonedAnswer)

    elif mode == "stateful":
        for attack in (trial.stateful_order or arms.PRESSURE_ATTACKS):
            conv.ctx.attack, conv.ctx.variant_idx = attack, 0
            r = conv.send(_attack_prompt(attack, 0, trial.offered_answer), ReasonedAnswer)
            if _is_fold(base, _letter(r), correct):
                break

    elif mode == "repeat":
        order = (trial.metadata or {}).get("variant_order") or list(range(len(_ATTACKS[trial.attack])))
        for v in order:
            conv.ctx.attack, conv.ctx.variant_idx = trial.attack, v
            r = conv.send(_attack_prompt(trial.attack, v, trial.offered_answer), ReasonedAnswer)
            if _is_fold(base, _letter(r), correct):
                break

    elif mode == "gauntlet":
        for attack, v in (trial.metadata or {}).get("gauntlet_sequence", []):
            conv.ctx.attack, conv.ctx.variant_idx = attack, v
            r = conv.send(_attack_prompt(attack, v, trial.offered_answer), ReasonedAnswer)
            if _is_fold(base, _letter(r), correct):
                break

    else:
        raise ValueError(f"unknown manipulation mode '{mode}'")
