"""Put-call parity utilities for European options."""

from __future__ import annotations

import math


def _validate_finite(value: float, name: str) -> None:
    """Validate that a value is finite."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")


def put_call_parity_gap(call_price: float, put_price: float, S: float, K: float, r: float, T: float) -> float:
    """Return the put-call parity gap for European options.

    The gap is computed as:
    ``(C - P) - (S - K * exp(-rT))``.

    A value near zero indicates parity is satisfied.
    """
    _validate_finite(call_price, "call_price")
    _validate_finite(put_price, "put_price")
    _validate_finite(S, "S")
    _validate_finite(K, "K")
    _validate_finite(r, "r")
    _validate_finite(T, "T")

    lhs = call_price - put_price
    rhs = S - K * math.exp(-r * T)
    return lhs - rhs


def is_put_call_parity_satisfied(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    tolerance: float = 1e-6,
) -> bool:
    """Return whether put-call parity is satisfied within a tolerance."""
    _validate_finite(tolerance, "tolerance")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")
    gap = put_call_parity_gap(call_price, put_price, S, K, r, T)
    return abs(gap) <= tolerance
