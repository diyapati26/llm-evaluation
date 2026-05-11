"""Manipulation resistance module — structured-output edition.

Every conversation turn returns a ReasonedAnswer ({letter, reasoning, …}) directly
from the provider, so we score on multiple axes:

  - letter_persistence  : 1.0 / 0.5 / 0.0 — held / hedged / folded
  - confidence_delta    : change in self-reported confidence (post − orig)
  - engagement          : 1.0 / 0.0 — did the model acknowledge the pushback?
  - hedged              : bool — did the letter become 'UNCERTAIN' after the attack?

Multiple phrasing variants per attack (loaded from data/manipulation_attacks.jsonl)
let run_manipulation.py report mean ± std across variants — the phrasing-sensitivity
metric. The `neutral_control` attack is no-pressure rephrasing; its fold rate
per (model, subject) is the natural drift baseline subtracted from raw resistance.

Conversation state is managed by providers.conversation.start_conversation, which
hides the OpenAI / Anthropic / Groq+OpenRouter / Gemini API differences behind a
uniform .send(message, schema) interface.

Numbers (1/2/3/4) replace A/B/C/D letters because the ReasonedAnswer schema uses
Literal['1','2','3','4','UNCERTAIN'].
"""
import scorers
from providers.conversation import start_conversation
from schemas import ReasonedAnswer
from utils.load_local import load_attacks, load_drift_turns

# {name: [variant_0, variant_1, variant_2, variant_3]} — edit the JSONL to tune.
ATTACKS = load_attacks()
DRIFT_TURNS = load_drift_turns()


# ── Helpers ───────────────────────────────────────────────────────


def get_wrong_number(correct_number):
    """Pick the next number after correct (cycling 1->2->3->4->1)."""
    try:
        c = int(correct_number)
    except (TypeError, ValueError):
        return "1"
    if c < 1 or c > 4:
        return "1"
    return str((c % 4) + 1)


def _answer_dict(parsed):
    """Convert a ReasonedAnswer instance to a plain dict (for JSON storage)."""
    return parsed.model_dump()


def _score_to_outcome(score):
    """Map letter-persistence float -> outcome string used by McNemar's + reporting."""
    if score is None:
        return "invalid"
    if score == 1.0:
        return "resistant"
    if score == 0.5:
        return "hedged"
    return "folded"


# ── Single-shot attack run ────────────────────────────────────────


def run_single_attack(model, question_text, correct_number, attack_type, variant_idx=0):
    """Run one (attack_type, variant_idx) against the model. Returns result dict."""
    if attack_type not in ATTACKS:
        raise ValueError(f"Unknown attack: {attack_type}")
    variants = ATTACKS[attack_type]
    if variant_idx < 0 or variant_idx >= len(variants):
        raise ValueError(f"variant_idx {variant_idx} out of range for {attack_type} ({len(variants)} variants)")

    wrong = get_wrong_number(correct_number)
    # neutral_control templates have no {wrong_number} placeholder; .format()
    # ignores unused keyword arguments, so the same line works for both.
    attack_prompt = variants[variant_idx].format(wrong_number=wrong)

    conv = start_conversation(model)
    r1 = conv.send(question_text, ReasonedAnswer)
    orig = r1.answer
    r2 = conv.send(attack_prompt, ReasonedAnswer)
    post = r2.answer

    correct_str = str(correct_number)
    score = scorers.score_letter_persistence(
        getattr(orig, "letter", None),
        getattr(post, "letter", None),
        correct_str,
    )

    return {
        "model": model,
        "attack": attack_type,
        "variant_idx": variant_idx,
        "wrong_offered": wrong,
        "correct": correct_str,
        "original": _answer_dict(orig),
        "post_attack": _answer_dict(post),
        "score": score,
        "outcome": _score_to_outcome(score),
        "scores": scorers.score_resistance_vector(orig, post, correct_str),
        "tokens": r1.input_tokens + r1.output_tokens
                + r2.input_tokens + r2.output_tokens,
        "cost_usd": round(r1.cost_usd + r2.cost_usd, 6),
        "latency_ms": round(r1.latency_ms + r2.latency_ms, 2),
    }


# ── 6-turn drift run ──────────────────────────────────────────────


def run_drift_attack(model, question_text, correct_number):
    """Run the 6-turn incremental drift sequence."""
    wrong = get_wrong_number(correct_number)
    conv = start_conversation(model)

    r1 = conv.send(question_text, ReasonedAnswer)
    transcript = [{"turn": 1, "answer": _answer_dict(r1.answer)}]
    cost = r1.cost_usd
    tokens = r1.input_tokens + r1.output_tokens
    latency = r1.latency_ms

    for turn_idx, (turn_name, template) in enumerate(DRIFT_TURNS.items(), start=2):
        prompt = template.format(wrong_number=wrong)
        r = conv.send(prompt, ReasonedAnswer)
        transcript.append({
            "turn": turn_idx,
            "attack_type": turn_name,
            "answer": _answer_dict(r.answer),
        })
        cost += r.cost_usd
        tokens += r.input_tokens + r.output_tokens
        latency += r.latency_ms

    correct_str = str(correct_number)
    orig_dict = transcript[0]["answer"]
    final_dict = transcript[-1]["answer"]

    per_turn = []
    for entry in transcript[1:]:
        per_turn.append({
            "turn": entry["turn"],
            "letter": entry["answer"]["letter"],
            "score": scorers.score_letter_persistence(
                orig_dict["letter"],
                entry["answer"]["letter"],
                correct_str,
            ),
        })

    final_score = scorers.score_letter_persistence(
        orig_dict["letter"], final_dict["letter"], correct_str
    )

    return {
        "model": model,
        "attack": "incremental_drift_6_turn",
        "variant_idx": 0,
        "wrong_offered": wrong,
        "correct": correct_str,
        "transcript": transcript,
        "per_turn_scores": per_turn,
        "score": final_score,
        "outcome": _score_to_outcome(final_score),
        "scores": scorers.score_resistance_vector(orig_dict, final_dict, correct_str),
        "tokens": tokens,
        "cost_usd": round(cost, 6),
        "latency_ms": round(latency, 2),
    }


# ── Public entry: run all attacks × variants on a single question ──


def run_all_attacks(model, question_text, correct_number, include_drift=True):
    """Run every (attack × variant) against one question. Returns list of result dicts."""
    results = []
    for attack_type, variants in ATTACKS.items():
        for variant_idx in range(len(variants)):
            try:
                results.append(run_single_attack(
                    model, question_text, correct_number, attack_type, variant_idx
                ))
            except Exception as e:
                results.append({
                    "model": model,
                    "attack": attack_type,
                    "variant_idx": variant_idx,
                    "error": str(e),
                    "score": None,
                    "outcome": "invalid",
                })
    if include_drift:
        try:
            results.append(run_drift_attack(model, question_text, correct_number))
        except Exception as e:
            results.append({
                "model": model,
                "attack": "incremental_drift_6_turn",
                "variant_idx": 0,
                "error": str(e),
                "score": None,
                "outcome": "invalid",
            })
    return results
