"""design.py — expand stimuli into the full list of Trials.

Pure expanders (inject items + attack templates -> Trials) so the design logic
is unit-testable with zero network, plus a build_all(cfg) orchestrator that
loads the real stimuli and expands every enabled module.

Manipulation expansion per item (x resamples):
  - stateless : item x (7 pressure attacks + neutral_control) x phrasing variants (fresh session each)
  - drift     : one 6-turn gradual-escalation sequence (if include_drift)
  - stateful  : all attack TYPES chained in one session (variant 0), counterbalanced order
  - repeat    : Mode 1 — one session per attack type, ALL its variants chained (counterbalanced)
  - gauntlet  : Mode 2 — one session, ALL attack types x ALL variants chained (counterbalanced)

The 2x2 the modes cover:  {one type | all types} x {one variant | all variants}.
"""
from __future__ import annotations

from latest.loaders import render_mc_prompt
from latest.plan import arms
from latest.plan.order import gauntlet_order, stateful_order, variant_order
from latest.plan.wrong_answer import pick_wrong
from latest.records import Trial, make_trial_id


def _mc_question(item: dict) -> str:
    return render_mc_prompt(item["question"], item["choices"]) if item.get("choices") else item["question"]


def _trial(coords: dict, **fields) -> Trial:
    return Trial(trial_id=make_trial_id(coords), **fields)


def _nvar(attacks, attack, max_variants) -> int:
    n = len(attacks[attack])
    return min(n, max_variants) if max_variants else n


def build_manipulation_trials(items, attacks, *, modes, include_drift, resamples, seed, max_variants=None) -> list[Trial]:
    pressure = [a for a in arms.PRESSURE_ATTACKS if a in attacks]
    has_control = arms.CONTROL_ATTACK in attacks
    trials: list[Trial] = []

    for item in items:
        q = _mc_question(item)
        correct = item.get("correct_answer")
        offered, plaus = pick_wrong(correct, item.get("choices"), seed, item["item_id"])
        common = dict(
            module="manipulation", dataset="mmlu", item_id=item["item_id"],
            subject=item.get("subject"), question=q, choices=item.get("choices"),
            correct_answer=correct,
        )

        for rep in range(resamples):
            if "stateless" in modes:
                attack_list = pressure + ([arms.CONTROL_ATTACK] if has_control else [])
                for attack in attack_list:
                    arm = arms.arm_for_attack(attack)
                    is_control = arm == "control"
                    for v in range(_nvar(attacks, attack, max_variants)):
                        coords = dict(module="manipulation", item_id=item["item_id"], mode="stateless",
                                      attack=attack, variant_idx=v, replicate_idx=rep)
                        trials.append(_trial(
                            coords, **common, arm=arm, attack=attack, variant_idx=v, mode="stateless",
                            offered_answer=(None if is_control else offered),
                            distractor_plausibility=(None if is_control else plaus), replicate_idx=rep,
                        ))

            if include_drift:
                coords = dict(module="manipulation", item_id=item["item_id"], mode="drift",
                              attack=arms.DRIFT_SEQUENCE_ATTACK, replicate_idx=rep)
                trials.append(_trial(
                    coords, **common, arm="pressure_wrong", attack=arms.DRIFT_SEQUENCE_ATTACK, mode="drift",
                    offered_answer=offered, distractor_plausibility=plaus, replicate_idx=rep,
                ))

            if "stateful" in modes:
                coords = dict(module="manipulation", item_id=item["item_id"], mode="stateful",
                              attack=arms.STATEFUL_ATTACK, replicate_idx=rep)
                trials.append(_trial(
                    coords, **common, arm="pressure_wrong", attack=arms.STATEFUL_ATTACK, mode="stateful",
                    stateful_order=stateful_order(item["item_id"], seed),
                    offered_answer=offered, distractor_plausibility=plaus, replicate_idx=rep,
                ))

            # Mode 1 (repeat): one session per attack type, all variants chained.
            if "repeat" in modes:
                repeat_attacks = pressure + ([arms.CONTROL_ATTACK] if has_control else [])
                for attack in repeat_attacks:
                    arm = arms.arm_for_attack(attack)
                    is_control = arm == "control"
                    n_var = _nvar(attacks, attack, max_variants)
                    coords = dict(module="manipulation", item_id=item["item_id"], mode="repeat",
                                  attack=attack, replicate_idx=rep)
                    trials.append(_trial(
                        coords, **common, arm=arm, attack=attack, mode="repeat",
                        offered_answer=(None if is_control else offered),
                        distractor_plausibility=(None if is_control else plaus), replicate_idx=rep,
                        metadata={"variant_order": variant_order(item["item_id"], attack, seed, n_var)},
                    ))

            # Mode 2 (gauntlet): one session, every pressure attack x every variant chained.
            if "gauntlet" in modes:
                awv = [(a, _nvar(attacks, a, max_variants)) for a in pressure]
                coords = dict(module="manipulation", item_id=item["item_id"], mode="gauntlet",
                              attack=arms.GAUNTLET_ATTACK, replicate_idx=rep)
                trials.append(_trial(
                    coords, **common, arm="pressure_wrong", attack=arms.GAUNTLET_ATTACK, mode="gauntlet",
                    offered_answer=offered, distractor_plausibility=plaus, replicate_idx=rep,
                    metadata={"gauntlet_sequence": gauntlet_order(item["item_id"], seed, awv)},
                ))

    return trials


def build_benchmark_trials(items_by_dataset: dict[str, list[dict]]) -> list[Trial]:
    trials: list[Trial] = []
    for name, items in items_by_dataset.items():
        for item in items:
            q = _mc_question(item)
            coords = dict(module="benchmark", dataset=name, item_id=item["item_id"])
            trials.append(_trial(
                coords, module="benchmark", dataset=name, item_id=item["item_id"],
                subject=item.get("subject"), question=q, choices=item.get("choices"),
                correct_answer=item.get("correct_answer"),
                metadata={"refs": item["refs"]} if item.get("refs") else {},
            ))
    return trials


def build_moral_trials(scenarios: list[dict]) -> list[Trial]:
    trials: list[Trial] = []
    for s in scenarios:
        coords = dict(module="moral", dataset="moral", item_id=s["id"])
        trials.append(_trial(
            coords, module="moral", dataset="moral", item_id=s["id"],
            subject=s.get("category"), question=s["scenario"],
            metadata={k: v for k, v in s.items() if k not in ("id", "category", "scenario")},
        ))
    return trials


def build_all(cfg: dict | None = None) -> list[Trial]:
    """Load real stimuli for every enabled module and expand into Trials. (Network.)"""
    from latest import loaders
    from latest.config.loader import load_config

    cfg = cfg if cfg is not None else load_config()
    seed = (cfg.get("run") or {}).get("seed", 42)
    trials: list[Trial] = []

    mcfg = cfg.get("manipulation") or {}
    if mcfg.get("enabled"):
        subjects = (cfg.get("dataset") or {}).get("hard_mmlu") or []
        items = loaders.load_hard_mmlu_items(subjects, mcfg.get("items_per_subject", 1), seed)
        trials += build_manipulation_trials(
            items, loaders.load_attacks(),
            modes=mcfg.get("modes", []), include_drift=mcfg.get("include_drift", True),
            resamples=mcfg.get("resamples", 1), seed=seed, max_variants=mcfg.get("max_variants"),
        )

    bcfg = cfg.get("benchmark") or {}
    if bcfg.get("enabled"):
        items_by = {name: loaders.load_benchmark_items(name, bcfg.get("n", 10), seed)
                    for name in bcfg.get("datasets", [])}
        trials += build_benchmark_trials(items_by)

    if (cfg.get("moral") or {}).get("enabled"):
        trials += build_moral_trials(loaders.load_moral_scenarios())

    return trials
