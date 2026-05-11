"""Smoke test — send 'Hi' to each provider, print the response."""
import sys

from dotenv import load_dotenv

from providers import (
    anthropic_provider,
    gemini_provider,
    groq_provider,
    openai_provider,
    openrouter_provider,
)

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

msg = [{"role": "user", "content": "Hi"}]
print("openai     ", openai_provider.get_openai_chat(msg, "gpt-5.4-nano", max_tokens=50).text)
print("anthropic  ", anthropic_provider.get_anthropic_chat(msg, "claude-haiku-4-5", max_tokens=50).text)
print("groq       ", groq_provider.get_groq_chat(msg, "openai/gpt-oss-120b", max_tokens=50).text)
print("openrouter ", openrouter_provider.get_openrouter_chat(msg, "meta-llama/llama-3.3-70b-instruct", max_tokens=50).text)
print("gemini     ", gemini_provider.get_gemini_chat(msg, "gemini-2.5-flash", max_tokens=50).text)
