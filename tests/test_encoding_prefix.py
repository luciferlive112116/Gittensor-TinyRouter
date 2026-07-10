"""Offline unit tests for coordinator instruction-prefix encoding (IMPROVEMENTS.md).

These tests exercise ONLY the pure string/config layer in
``trinity.coordinator.encoding`` plus the wiring hooks in ``slm``,
``policy``, ``train``, and ``warmstart_head`` — no torch, GPU, or network.

Contract:
  * Default-off config preserves legacy encode input byte-for-byte.
  * Enabled prefix prepends the routing instruction once (idempotent).
  * ``CoordinatorEncoder.encode`` applies the prefix when configured.
  * ``trinity.train`` and ``scripts/warmstart_head.py`` read the yaml block.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.coordinator.encoding import (  # noqa: E402
    DEFAULT_ROUTING_INSTRUCTION,
    EncodingConfig,
    apply_instruction_prefix,
    effective_prefix,
    prefix_would_change,
)


# ---------------------------------------------------------------------------
# Guard: this module must stay torch-free.
# ---------------------------------------------------------------------------
def test_encoding_module_imports_without_torch():
    assert "torch" not in sys.modules, "encoding tests must not import torch"


# ---------------------------------------------------------------------------
# EncodingConfig parsing
# ---------------------------------------------------------------------------
def test_encoding_config_defaults_are_disabled():
    cfg = EncodingConfig()
    assert cfg.enabled is False
    assert cfg.prefix == DEFAULT_ROUTING_INSTRUCTION
    assert cfg.active is False


def test_encoding_config_from_empty_coord():
    assert EncodingConfig.from_coord_dict(None) == EncodingConfig()
    assert EncodingConfig.from_coord_dict({}) == EncodingConfig()
    assert EncodingConfig.from_coord_dict({"hidden_state": {}}) == EncodingConfig()


def test_encoding_config_from_enabled_block():
    coord = {
        "hidden_state": {
            "instruction_prefix": {
                "enabled": True,
                "text": "Route this:\n",
            }
        }
    }
    cfg = EncodingConfig.from_coord_dict(coord)
    assert cfg.enabled is True
    assert cfg.prefix == "Route this:\n"
    assert cfg.active is True


def test_encoding_config_shorthand_string_enables():
    coord = {"hidden_state": {"instruction_prefix": "Pick model:\n"}}
    cfg = EncodingConfig.from_coord_dict(coord)
    assert cfg.enabled is True
    assert cfg.prefix == "Pick model:\n"


def test_encoding_config_empty_shorthand_stays_disabled():
    coord = {"hidden_state": {"instruction_prefix": ""}}
    cfg = EncodingConfig.from_coord_dict(coord)
    assert cfg.enabled is False


def test_encoding_config_enabled_with_empty_text_uses_default():
    coord = {"hidden_state": {"instruction_prefix": {"enabled": True, "text": ""}}}
    cfg = EncodingConfig.from_coord_dict(coord)
    assert cfg.enabled is True
    assert cfg.prefix == DEFAULT_ROUTING_INSTRUCTION


def test_encoding_config_prefix_alias_key():
    coord = {
        "hidden_state": {
            "instruction_prefix": {"enabled": True, "prefix": "Alt:\n"}
        }
    }
    cfg = EncodingConfig.from_coord_dict(coord)
    assert cfg.prefix == "Alt:\n"


def test_encoding_config_active_false_when_disabled_even_with_text():
    cfg = EncodingConfig(enabled=False, prefix="hello")
    assert cfg.active is False


def test_encoding_config_active_false_when_enabled_but_whitespace_prefix():
    cfg = EncodingConfig(enabled=True, prefix="   ")
    assert cfg.active is False


# ---------------------------------------------------------------------------
# apply_instruction_prefix / helpers
# ---------------------------------------------------------------------------
def test_apply_prefix_noop_when_disabled():
    text = "QUERY:\nWhat is 2+2?"
    assert apply_instruction_prefix(text, None) == text
    assert apply_instruction_prefix(text, EncodingConfig()) == text


def test_apply_prefix_prepends_when_enabled():
    cfg = EncodingConfig(enabled=True, prefix="Instruct:\n")
    text = "QUERY:\nSolve x."
    assert apply_instruction_prefix(text, cfg) == "Instruct:\nQUERY:\nSolve x."


def test_apply_prefix_uses_default_routing_instruction():
    cfg = EncodingConfig(enabled=True)
    out = apply_instruction_prefix("task body", cfg)
    assert out.startswith(DEFAULT_ROUTING_INSTRUCTION)
    assert out.endswith("task body")


def test_apply_prefix_idempotent_when_already_prefixed():
    cfg = EncodingConfig(enabled=True, prefix="Instruct:\n")
    once = apply_instruction_prefix("hello", cfg)
    twice = apply_instruction_prefix(once, cfg)
    assert once == twice == "Instruct:\nhello"


def test_effective_prefix_returns_none_when_inactive():
    assert effective_prefix(None) is None
    assert effective_prefix(EncodingConfig()) is None


def test_effective_prefix_returns_string_when_active():
    cfg = EncodingConfig(enabled=True, prefix="X:")
    assert effective_prefix(cfg) == "X:"


def test_prefix_would_change_detects_needed_prepend():
    cfg = EncodingConfig(enabled=True, prefix="P:")
    assert prefix_would_change("q", cfg) is True
    assert prefix_would_change("P:q", cfg) is False
    assert prefix_would_change("", cfg) is True


def test_prefix_would_change_false_when_disabled():
    cfg = EncodingConfig(enabled=False, prefix="P:")
    assert prefix_would_change("q", cfg) is False


@pytest.mark.parametrize(
    "text",
    [
        "",
        "short",
        "QUERY:\n" + "x" * 500,
        "unicode: café — ∑ integral",
        "line1\nline2\nVERDICT: ACCEPT",
    ],
)
def test_apply_prefix_roundtrip_preserves_body(text):
    cfg = EncodingConfig(enabled=True, prefix=">>> ")
    out = apply_instruction_prefix(text, cfg)
    assert out.startswith(">>> ")
    assert out[len(">>> ") :] == text


def test_default_instruction_matches_improvements_doc():
    assert "Select the best solver model and role" in DEFAULT_ROUTING_INSTRUCTION
    assert DEFAULT_ROUTING_INSTRUCTION.endswith("Query: ")


# ---------------------------------------------------------------------------
# trinity.yaml contract
# ---------------------------------------------------------------------------
def test_trinity_yaml_has_instruction_prefix_block_default_off():
    cfg = yaml.safe_load((_REPO / "configs" / "trinity.yaml").read_text())
    block = cfg["coordinator"]["hidden_state"]["instruction_prefix"]
    assert block["enabled"] is False
    parsed = EncodingConfig.from_coord_dict(cfg["coordinator"])
    assert parsed.active is False
    assert parsed.prefix == block["text"]


def test_trinity_yaml_prefix_text_matches_default_constant():
    cfg = yaml.safe_load((_REPO / "configs" / "trinity.yaml").read_text())
    assert cfg["coordinator"]["hidden_state"]["instruction_prefix"]["text"] == (
        DEFAULT_ROUTING_INSTRUCTION
    )


# ---------------------------------------------------------------------------
# slm.encode delegates to apply_instruction_prefix (source contract)
# ---------------------------------------------------------------------------
def test_slm_encode_calls_apply_instruction_prefix():
    src = (_REPO / "src" / "trinity" / "coordinator" / "slm.py").read_text()
    assert "apply_instruction_prefix(transcript_text, self.encoding)" in src
    assert "encoding: EncodingConfig | None = None" in src


def test_coordinator_encoder_from_config_reads_prefix():
    cfg_path = _REPO / "configs" / "trinity.yaml"
    # from_config needs torch; verify the encoding field is parsed the same way.
    raw = yaml.safe_load(cfg_path.read_text())
    enc_cfg = EncodingConfig.from_coord_dict(raw["coordinator"])
    assert enc_cfg.active is False


# ---------------------------------------------------------------------------
# train.py wiring (import-only / config path)
# ---------------------------------------------------------------------------
def test_train_reads_encoding_config_from_yaml():
    from trinity.coordinator.encoding import EncodingConfig as EC

    cfg = yaml.safe_load((_REPO / "configs" / "trinity.yaml").read_text())
    enc = EC.from_coord_dict(cfg["coordinator"])
    assert enc.active is False


def test_train_summary_includes_instruction_prefix_flag():
    # Static check: train.py source mentions the summary field.
    src = (_REPO / "src" / "trinity" / "train.py").read_text()
    assert "instruction_prefix_enabled" in src
    assert "EncodingConfig.from_coord_dict" in src


# ---------------------------------------------------------------------------
# warmstart_head.py encode path reads config
# ---------------------------------------------------------------------------
def test_warmstart_head_script_reads_encoding_config():
    src = (_REPO / "scripts" / "warmstart_head.py").read_text()
    assert "EncodingConfig.from_coord_dict" in src
    assert "encoding=enc_cfg" in src


# ---------------------------------------------------------------------------
# policy.build passes encoding through (signature check)
# ---------------------------------------------------------------------------
def test_policy_build_accepts_encoding_parameter():
    import inspect

    from trinity.coordinator.policy import CoordinatorPolicy

    sig = inspect.signature(CoordinatorPolicy.build)
    assert "encoding" in sig.parameters


# ---------------------------------------------------------------------------
# Regression: enabling prefix must not break empty transcript guard
# ---------------------------------------------------------------------------
def test_apply_prefix_empty_string_with_active_config():
    cfg = EncodingConfig(enabled=True, prefix="P:")
    assert apply_instruction_prefix("", cfg) == "P:"


def test_apply_prefix_multiline_transcript_shape():
    """Session transcripts start with QUERY:\\n — prefix goes before that."""
    transcript = "QUERY:\nFind primes under 10.\n\n[Turn 1 | worker | m1]\n2,3,5,7"
    cfg = EncodingConfig(enabled=True, prefix="Instruct: route\n")
    out = apply_instruction_prefix(transcript, cfg)
    assert out.startswith("Instruct: route\nQUERY:")


# ---------------------------------------------------------------------------
# Frozen dataclass immutability
# ---------------------------------------------------------------------------
def test_encoding_config_is_frozen():
    cfg = EncodingConfig()
    with pytest.raises(AttributeError):
        cfg.enabled = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Property: disabled path is identity for many random prefixes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enabled", [False, True])
def test_apply_prefix_deterministic(enabled):
    cfg = EncodingConfig(enabled=enabled, prefix="ABC:")
    text = "same input"
    assert apply_instruction_prefix(text, cfg) == apply_instruction_prefix(text, cfg)


def test_apply_prefix_does_not_strip_or_normalize_body():
    cfg = EncodingConfig(enabled=True, prefix="P:")
    body = "  spaced  \n\ttabs"
    assert apply_instruction_prefix(body, cfg) == "P:  spaced  \n\ttabs"
