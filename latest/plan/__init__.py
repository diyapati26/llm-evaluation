"""plan/ — the frozen experimental-design matrix.

Builds trials.parquet (one Trial per experimental cell) from a seed, BEFORE any
API call. Every design-validity choice — which wrong answer is offered, how
plausible it is, the stateful attack order, the arm — is an auditable column,
not logic buried in a loop.
"""
