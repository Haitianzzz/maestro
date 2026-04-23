"""Maestro three-tier verifier package (spec 06).

Layered verification:

* Tier 1 ``deterministic`` — ruff + mypy (zero API cost) — landed module 11.
* Tier 2 ``test_based`` — pytest, optionally auto-generated tests — module 12.
* Tier 3 ``llm_judge`` — multi-sample LLM-as-Judge with disagreement
  detection — module 13.

This package exports only what module 11 ships. ``Verifier`` (the
tier-orchestrating class) is added when all three tiers land; for now the
scheduler can use :class:`DeterministicVerifier` directly as a
``VerifierProtocol`` satisfier.
"""

from .deterministic import DeterministicVerifier, SubprocessResult, run_subprocess

__all__ = [
    "DeterministicVerifier",
    "SubprocessResult",
    "run_subprocess",
]
