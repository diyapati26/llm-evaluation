"""Probe Groq's OpenAI-compatible endpoint to see which APIs are supported.

Run this BEFORE designing the new generic Groq provider — its output
determines whether we can use the Responses API + Conversations state
on Groq, or have to stick with chat.completions + client-side messages.

Usage (from repo root, with venv active):
    python "Simpler Arch/tests/probe_groq_responses_api.py"

Each test reports PASS / FAIL with the error so you can paste the
result back into the design conversation.
"""
import os
import sys
import traceback

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal

load_dotenv()

MODEL = "openai/gpt-oss-120b"  # the model we care about for the paper


def _client():
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )


# A minimal Pydantic schema for the structured-output tests
class Probe(BaseModel):
    letter: Literal["A", "B", "C", "D"]
    reasoning: str


def test(name, fn):
    """Run a probe, print PASS/FAIL with a short diagnostic."""
    print(f"\n--- {name} ---")
    try:
        result = fn()
        print(f"  PASS  -->{result}")
        return True
    except Exception as e:
        # Print just the first line of the traceback for brevity
        err = str(e).splitlines()[0] if str(e) else type(e).__name__
        print(f"  FAIL  -->{err}")
        return False


# ── 1. Does Groq accept /v1/responses at all? ────────────────────
def probe_responses_create():
    c = _client()
    r = c.responses.create(
        model=MODEL,
        input="Say the single word 'ok' and nothing else.",
        max_output_tokens=10,
    )
    return (r.output_text or "").strip()[:40]


# ── 2. Does Groq honor responses.parse(text_format=Pydantic)? ────
def probe_responses_parse():
    c = _client()
    r = c.responses.parse(
        model=MODEL,
        input=[{"role": "user", "content": (
            "Question: 2+2=?\n"
            "A. 3   B. 4   C. 5   D. 6\n"
            "Pick a letter and give one sentence reasoning."
        )}],
        text_format=Probe,
        max_output_tokens=120,
    )
    return f"letter={r.output_parsed.letter} reasoning={r.output_parsed.reasoning[:30]!r}"


# ── 3. Does Groq support conversations.create (server-side state)?
def probe_conversations_create():
    c = _client()
    conv = c.conversations.create()
    return f"conversation_id={conv.id}"


# ── 4. Can we do a real multi-turn via conversations + responses? ─
def probe_conversation_multi_turn():
    c = _client()
    conv = c.conversations.create()
    r1 = c.responses.create(
        model=MODEL, conversation=conv.id,
        input=[{"role": "user", "content": "My favorite number is 7. Remember it."}],
        max_output_tokens=40,
    )
    r2 = c.responses.create(
        model=MODEL, conversation=conv.id,
        input=[{"role": "user", "content": "What number did I just tell you?"}],
        max_output_tokens=20,
    )
    return f"turn2={r2.output_text.strip()[:60]!r}"


# ── 5. Sanity: the path we already know works ────────────────────
def _make_strict(node):
    """Groq strict mode requires additionalProperties=False on every object node."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "additionalProperties" not in node:
            node["additionalProperties"] = False
        for v in node.values():
            _make_strict(v)
    elif isinstance(node, list):
        for v in node:
            _make_strict(v)
    return node


def probe_chat_completions_json_schema():
    c = _client()
    r = c.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": (
            "Question: 2+2=?\n"
            "A. 3   B. 4   C. 5   D. 6\n"
            "Pick a letter and give one sentence reasoning."
        )}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "Probe",
                "schema": _make_strict(Probe.model_json_schema()),
                "strict": True,
            },
        },
        max_tokens=120,
    )
    return r.choices[0].message.content[:60]


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("Set GROQ_API_KEY in .env first.")
        sys.exit(1)

    print(f"Probing Groq @ api.groq.com/openai/v1 with model={MODEL}")

    results = {
        "responses.create":           test("1. responses.create",          probe_responses_create),
        "responses.parse(Pydantic)":  test("2. responses.parse",           probe_responses_parse),
        "conversations.create":       test("3. conversations.create",      probe_conversations_create),
        "multi-turn via conversations": test("4. multi-turn conversations", probe_conversation_multi_turn),
        "chat.completions+json_schema": test("5. chat.completions json_schema (baseline)",
                                             probe_chat_completions_json_schema),
    }

    print("\n" + "=" * 56)
    print("SUMMARY")
    print("=" * 56)
    for name, ok in results.items():
        flag = "PASS" if ok else "FAIL"
        print(f"  {flag:4}  {name}")

    print("\nWhat to do with this output:")
    print("  - 1 & 2 PASS --> new Groq provider can use Responses API")
    print("  - 3 & 4 PASS --> Groq supports server-side conversation state too")
    print("  - 5 always PASS --> chat.completions+json_schema is the fallback")


if __name__ == "__main__":
    main()
