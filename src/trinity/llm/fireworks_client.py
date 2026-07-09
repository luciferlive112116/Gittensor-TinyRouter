"""Backward-compatible import shim for the old Fireworks client path.

The repo now uses OpenRouter as its only hosted model provider. New code should
import :mod:`trinity.llm.openrouter_client`.
"""
from __future__ import annotations

from .openrouter_client import ChatResult, OpenRouterPool, main

FireworksPool = OpenRouterPool

__all__ = ["ChatResult", "OpenRouterPool", "FireworksPool", "main"]
