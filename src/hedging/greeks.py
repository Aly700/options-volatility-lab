"""Black-Scholes Greeks for European options."""

from __future__ import annotations

import math

from pricing.black_scholes import _validate_finite, _validate_positive_finite, norm_cdf

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_BOUNDARY_TOLERANCE = 1e-12


def _norm_pdf(x: float) -> float:
    """Return the standard normal probability density at ``x``."""
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _validate_inputs(S: float, K: float, T: float, r: float, sigma: float) -> None:
    """Validate common Black-Scholes Greek inputs."""
    _validate_positive_finite(S, "S")
    _validate_positive_finite(K, "K")
    _validate_finite(T, "T")
    _validate_finite(r, "r")
    _validate_finite(sigma, "sigma")


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """Return the Black-Scholes ``d1`` and ``d2`` terms."""
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def _discounted_strike(K: float, r: float, T: float) -> float:
    """Return the discounted strike used in zero-volatility limits."""
    return K * math.exp(-r * T)


def _boundary_value(value: float, lower: float, upper: float, midpoint: float) -> float:
    """Return a piecewise value with a midpoint at the decision boundary."""
    if math.isclose(value, midpoint, rel_tol=0.0, abs_tol=_BOUNDARY_TOLERANCE):
        return (lower + upper) / 2.0
    if value > midpoint:
        return upper
    return lower


def call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Return the Black-Scholes delta of a European call option."""
    _validate_inputs(S, K, T, r, sigma)

    if T <= 0.0:
        return _boundary_value(S, 0.0, 1.0, K)

    if sigma <= 0.0:
        return _boundary_value(S, 0.0, 1.0, _discounted_strike(K, r, T))

    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm_cdf(d1)


def put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Return the Black-Scholes delta of a European put option."""
    _validate_inputs(S, K, T, r, sigma)

    if T <= 0.0:
        return _boundary_value(S, -1.0, 0.0, K)

    if sigma <= 0.0:
        return _boundary_value(S, -1.0, 0.0, _discounted_strike(K, r, T))

    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm_cdf(d1) - 1.0


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Return the Black-Scholes gamma for European options."""
    _validate_inputs(S, K, T, r, sigma)

    if T <= 0.0 or sigma <= 0.0:
        return 0.0

    d1, _ = _d1_d2(S, K, T, r, sigma)
    return _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Return the Black-Scholes vega for European options."""
    _validate_inputs(S, K, T, r, sigma)

    if T <= 0.0 or sigma <= 0.0:
        return 0.0

    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * _norm_pdf(d1) * math.sqrt(T)


def call_theta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Return the Black-Scholes theta of a European call option."""
    _validate_inputs(S, K, T, r, sigma)

    if T <= 0.0:
        return 0.0

    discounted_strike = _discounted_strike(K, r, T)
    if sigma <= 0.0:
        if S > discounted_strike:
            return -r * discounted_strike
        return 0.0

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    time_decay = -(S * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    carry = -r * K * math.exp(-r * T) * norm_cdf(d2)
    return time_decay + carry


def put_theta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Return the Black-Scholes theta of a European put option."""
    _validate_inputs(S, K, T, r, sigma)

    if T <= 0.0:
        return 0.0

    discounted_strike = _discounted_strike(K, r, T)
    if sigma <= 0.0:
        if S < discounted_strike:
            return r * discounted_strike
        return 0.0

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    time_decay = -(S * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    carry = r * K * math.exp(-r * T) * norm_cdf(-d2)
    return time_decay + carry


def call_rho(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Return the Black-Scholes rho of a European call option."""
    _validate_inputs(S, K, T, r, sigma)

    if T <= 0.0:
        return 0.0

    discounted_strike = _discounted_strike(K, r, T)
    if sigma <= 0.0:
        if S > discounted_strike:
            return T * discounted_strike
        return 0.0

    _, d2 = _d1_d2(S, K, T, r, sigma)
    return K * T * math.exp(-r * T) * norm_cdf(d2)


def put_rho(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Return the Black-Scholes rho of a European put option."""
    _validate_inputs(S, K, T, r, sigma)

    if T <= 0.0:
        return 0.0

    discounted_strike = _discounted_strike(K, r, T)
    if sigma <= 0.0:
        if S < discounted_strike:
            return -T * discounted_strike
        return 0.0

    _, d2 = _d1_d2(S, K, T, r, sigma)
    return -K * T * math.exp(-r * T) * norm_cdf(-d2)
