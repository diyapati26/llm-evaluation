"""Provider layer.

Importing this package registers all concrete providers with the router (each
provider module calls router.register at import time). The router exposes the
single resolve()/start_conversation()/chat()/estimate_cost() entry points used
by every collector — so there is exactly ONE place that maps a model string to
a provider (no dual route()/_resolve() drift).

Provider modules use lazy SDK imports, so importing this package is safe even if
a given provider's SDK isn't installed; the SDK is only needed at first call.
"""
from latest.providers import router  # noqa: F401
from latest.providers.router import (  # noqa: F401
    chat,
    estimate_cost,
    registered_providers,
    resolve,
    start_conversation,
)

# Trigger registration of the concrete providers.
from latest.providers import anthropic as _anthropic  # noqa: F401,E402
from latest.providers import chat_completions as _chat_completions  # noqa: F401,E402
from latest.providers import gemini as _gemini  # noqa: F401,E402
from latest.providers import openai as _openai  # noqa: F401,E402
