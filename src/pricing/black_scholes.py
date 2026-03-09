"""Black-Scholes pricing functions for European options."""

from __future__ import annotations

import math


def norm_cdf(x: float) -> float:
    """Return the standard normal cumulative distribution value at ``x``."""
    if not math.isfinite(x):
        raise ValueError("x must be a finite number.")
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _validate_positive_finite(value: float, name: str) -> None:
    """Validate that a numeric input is finite and strictly positive."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if value <= 0.0:
        raise ValueError(f"{name} must be greater than 0.")


def _validate_finite(value: float, name: str) -> None:
    """Validate that a numeric input is finite."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")


def _intrinsic_call(S: float, K: float) -> float:
    return max(S - K, 0.0)


def _intrinsic_put(S: float, K: float) -> float:
    return max(K - S, 0.0)


def black_scholes_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute the Black-Scholes price of a European call option.

    Handles edge cases as follows:
    - ``T <= 0``: returns intrinsic value ``max(S - K, 0)``
    - ``sigma <= 0``: returns discounted deterministic payoff

    Raises:
        ValueError: If any input is non-finite, or if ``S``/``K`` are invalid.
    """
    _validate_positive_finite(S, "S")
    _validate_positive_finite(K, "K")
    _validate_finite(T, "T")
    _validate_finite(r, "r")
    _validate_finite(sigma, "sigma")

    if T <= 0.0:
        return _intrinsic_call(S, K)

    if sigma <= 0.0:
        forward_spot = S * math.exp(r * T)
        return math.exp(-r * T) * max(forward_spot - K, 0.0)

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def black_scholes_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute the Black-Scholes price of a European put option.

    Handles edge cases as follows:
    - ``T <= 0``: returns intrinsic value ``max(K - S, 0)``
    - ``sigma <= 0``: returns discounted deterministic payoff

    Raises:
        ValueError: If any input is non-finite, or if ``S``/``K`` are invalid.
    """
    _validate_positive_finite(S, "S")
    _validate_positive_finite(K, "K")
    _validate_finite(T, "T")
    _validate_finite(r, "r")
    _validate_finite(sigma, "sigma")

    if T <= 0.0:
        return _intrinsic_put(S, K)

    if sigma <= 0.0:
        forward_spot = S * math.exp(r * T)
        return math.exp(-r * T) * max(K - forward_spot, 0.0)

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
