"""Coordinator SLM input formatting (IMPROVEMENTS.md quick win).

The frozen Qwen3-0.6B encoder is used as a *routing* feature extractor, not a
general-purpose LM. Prepending a short instruction before tokenization steers the
penultimate hidden state toward "which model/role should act next?" without adding
parameters. This is the Qwen3-Embedding recipe (arXiv 2506.05176): zero new params,
but old head checkpoints are invalid under a changed prefix — the knob is default-off.

The helpers here are pure string/config logic with **no torch dependency**, so they
unit-test on a box without a GPU. :class:`~trinity.coordinator.slm.CoordinatorEncoder`
and :func:`~trinity.coordinator.warmstart.encode_queries` call
:func:`apply_instruction_prefix` at encode time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_ROUTING_INSTRUCTION",
    "EncodingConfig",
    "apply_instruction_prefix",
    "effective_prefix",
    "prefix_would_change",
]

# Recommended prefix from docs/IMPROVEMENTS.md (instruction-prefix quick win).
# Ends with "Query: " so the following task text reads naturally when the
# transcript already starts with "QUERY:\n..." from the session loop.
DEFAULT_ROUTING_INSTRUCTION = (
    "Instruct: Select the best solver model and role for the following query.\n"
    "Query: "
)


@dataclass(frozen=True)
class EncodingConfig:
    """Optional instruction prefix applied before SLM tokenization.

    Attributes:
        enabled: When ``False`` (default), encoding is byte-identical to the
            legacy path — no prefix is prepended.
        prefix: The literal string prepended when ``enabled`` is ``True``. Empty
            or whitespace-only values are treated as disabled.
    """

    enabled: bool = False
    prefix: str = DEFAULT_ROUTING_INSTRUCTION

    @classmethod
    def from_coord_dict(cls, coord: dict[str, Any] | None) -> "EncodingConfig":
        """Parse ``coordinator.hidden_state.instruction_prefix`` from yaml."""
        if not coord:
            return cls()
        hs = coord.get("hidden_state") or {}
        block = hs.get("instruction_prefix")
        if block is None:
            return cls()
        if isinstance(block, str):
            # Shorthand: a bare string means "enabled with this prefix".
            if not block:
                return cls()
            return cls(enabled=True, prefix=block)
        if not isinstance(block, dict):
            return cls()
        prefix = str(block.get("text", block.get("prefix", DEFAULT_ROUTING_INSTRUCTION)))
        enabled = bool(block.get("enabled", False))
        # An explicit enabled=True with an empty text still falls back to default.
        if enabled and not prefix.strip():
            prefix = DEFAULT_ROUTING_INSTRUCTION
        return cls(enabled=enabled, prefix=prefix)

    @property
    def active(self) -> bool:
        """True iff a non-empty prefix would be applied."""
        return self.enabled and bool(self.prefix.strip())


def effective_prefix(cfg: EncodingConfig | None) -> str | None:
    """Return the prefix string to apply, or ``None`` when encoding is unchanged."""
    if cfg is None or not cfg.active:
        return None
    return cfg.prefix


def prefix_would_change(text: str, cfg: EncodingConfig | None) -> bool:
    """True iff ``apply_instruction_prefix`` would alter ``text``."""
    p = effective_prefix(cfg)
    if p is None:
        return False
    if not text:
        return True
    return not text.startswith(p)


def apply_instruction_prefix(text: str, cfg: EncodingConfig | None = None) -> str:
    """Prepend the routing instruction when configured.

    Idempotent: if ``text`` already starts with the configured prefix, it is
    returned unchanged (so callers may safely run transcript builders and warm-
    start encoders through the same helper).

    Args:
        text: Raw transcript or query text about to be tokenized.
        cfg: Parsed :class:`EncodingConfig`; ``None`` -> legacy (no prefix).

    Returns:
        ``text`` unchanged when disabled, else ``prefix + text`` (unless already
        prefixed).
    """
    p = effective_prefix(cfg)
    if p is None:
        return text
    if text.startswith(p):
        return text
    return f"{p}{text}"
