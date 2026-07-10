"""OpenRouter per-model token pricing for ledger cost totals."""
from __future__ import annotations

from pathlib import Path

from trinity.llm.cost_ledger import LedgerEntry, read_ledger_entries, verify_ledger_chain

__all__ = [
    "OPENROUTER_POOL_PRICES",
    "default_blended_rates",
    "normalize_model_slug",
    "resolve_rates",
    "token_cost",
    "sum_entry_costs",
    "sum_ledger_cost",
    "verified_ledger_total_usd",
]

OPENROUTER_POOL_PRICES: dict[str, tuple[float, float]] = {
    "qwen3.5-35b-a3b": (0.14, 1.00),
    "minimax-m3": (0.30, 1.20),
    "deepseek-v4-flash": (0.09, 0.18),
}


def normalize_model_slug(model: str) -> str:
    return model.rsplit("/", 1)[-1]


def default_blended_rates() -> tuple[float, float]:
    if not OPENROUTER_POOL_PRICES:
        return (0.0, 0.0)
    ins = [p[0] for p in OPENROUTER_POOL_PRICES.values()]
    outs = [p[1] for p in OPENROUTER_POOL_PRICES.values()]
    return (sum(ins) / len(ins), sum(outs) / len(outs))


def resolve_rates(model: str) -> tuple[float, float]:
    slug = normalize_model_slug(model)
    return OPENROUTER_POOL_PRICES.get(slug, default_blended_rates())


def token_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_rate, out_rate = resolve_rates(model)
    return int(prompt_tokens) / 1e6 * in_rate + int(completion_tokens) / 1e6 * out_rate


def sum_entry_costs(entries: list[LedgerEntry]) -> float:
    return sum(
        token_cost(e.model, e.prompt_tokens, e.completion_tokens) for e in entries
    )


def sum_ledger_cost(path: str | Path) -> float:
    return sum_entry_costs(read_ledger_entries(path))


def verified_ledger_total_usd(path: str | Path) -> float | None:
    try:
        valid, _, _ = verify_ledger_chain(path)
        if not valid:
            return None
        return sum_ledger_cost(path)
    except OSError:
        return None
