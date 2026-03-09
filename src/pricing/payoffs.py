"""Simple payoff functions for basic option/asset positions."""

from __future__ import annotations

from typing import Optional


def call_payoff(S_T: float, K: float) -> float:
    """Return the terminal payoff of a long European call."""
    return max(S_T - K, 0.0)


def put_payoff(S_T: float, K: float) -> float:
    """Return the terminal payoff of a long European put."""
    return max(K - S_T, 0.0)


def long_stock_payoff(S_T: float, S_0: Optional[float] = None) -> float:
    """Return terminal stock value or PnL if ``S_0`` is provided."""
    if S_0 is None:
        return S_T
    return S_T - S_0


def bond_payoff(K: float) -> float:
    """Return the terminal payoff of a zero-coupon bond paying ``K`` at maturity."""
    return K
