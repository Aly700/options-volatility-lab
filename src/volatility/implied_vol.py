"""Implied volatility solvers for European Black-Scholes options."""

from __future__ import annotations

import math
from typing import Callable

from scipy.optimize import brentq

from pricing.black_scholes import (
    _validate_finite,
    _validate_positive_finite,
    black_scholes_call_price,
    black_scholes_put_price,
)

_VOL_LOWER_BOUND = 1e-6
_VOL_UPPER_BOUND = 5.0


def _validate_solver_inputs(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    initial_guess: float,
    tol: float,
    max_iter: int,
) -> None:
    """Validate common implied-volatility solver inputs."""
    _validate_finite(market_price, "market_price")
    if market_price < 0.0:
        raise ValueError("market_price must be non-negative.")

    _validate_positive_finite(S, "S")
    _validate_positive_finite(K, "K")
    _validate_finite(T, "T")
    if T <= 0.0:
        raise ValueError("T must be greater than 0.")

    _validate_finite(r, "r")
    _validate_finite(initial_guess, "initial_guess")
    if initial_guess <= 0.0:
        raise ValueError("initial_guess must be greater than 0.")

    _validate_finite(tol, "tol")
    if tol <= 0.0:
        raise ValueError("tol must be greater than 0.")

    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")


def _discounted_strike(K: float, r: float, T: float) -> float:
    """Return the present value of the strike."""
    return K * math.exp(-r * T)


def _call_arbitrage_bounds(S: float, K: float, T: float, r: float) -> tuple[float, float]:
    """Return the no-arbitrage price bounds for a European call."""
    discounted_strike = _discounted_strike(K, r, T)
    return max(0.0, S - discounted_strike), S


def _put_arbitrage_bounds(S: float, K: float, T: float, r: float) -> tuple[float, float]:
    """Return the no-arbitrage price bounds for a European put."""
    discounted_strike = _discounted_strike(K, r, T)
    return max(0.0, discounted_strike - S), discounted_strike


def _validate_market_price(market_price: float, lower_bound: float, upper_bound: float, option_type: str) -> None:
    """Validate that a market price lies within no-arbitrage bounds."""
    if market_price < lower_bound or market_price > upper_bound:
        raise ValueError(
            f"{option_type} market_price must lie within arbitrage bounds "
            f"[{lower_bound}, {upper_bound}]."
        )


def call_price_error(sigma: float, market_price: float, S: float, K: float, T: float, r: float) -> float:
    """Return Black-Scholes call pricing error for a candidate volatility."""
    return black_scholes_call_price(S=S, K=K, T=T, r=r, sigma=sigma) - market_price


def put_price_error(sigma: float, market_price: float, S: float, K: float, T: float, r: float) -> float:
    """Return Black-Scholes put pricing error for a candidate volatility."""
    return black_scholes_put_price(S=S, K=K, T=T, r=r, sigma=sigma) - market_price


def _solve_implied_vol(
    market_price: float,
    initial_guess: float,
    tol: float,
    max_iter: int,
    lower_bound: float,
    upper_bound: float,
    price_error: Callable[[float], float],
) -> float:
    """Solve for implied volatility on a bounded interval."""
    if math.isclose(market_price, lower_bound, rel_tol=0.0, abs_tol=tol):
        return 0.0

    if math.isclose(initial_guess, 0.0, rel_tol=0.0, abs_tol=tol):
        initial_guess = _VOL_LOWER_BOUND

    if abs(price_error(initial_guess)) <= tol:
        return initial_guess

    lower_sigma = _VOL_LOWER_BOUND
    upper_sigma = _VOL_UPPER_BOUND
    lower_error = price_error(lower_sigma)
    upper_error = price_error(upper_sigma)

    if lower_error > 0.0 or upper_error < 0.0:
        raise ValueError(
            "No implied volatility root was bracketed on the search interval "
            f"[{lower_sigma}, {upper_sigma}] for price bounds [{lower_bound}, {upper_bound}]."
        )

    return float(brentq(price_error, lower_sigma, upper_sigma, xtol=tol, rtol=tol, maxiter=max_iter))


def implied_vol_call(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    initial_guess: float = 0.2,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Return the Black-Scholes implied volatility of a European call option."""
    _validate_solver_inputs(market_price, S, K, T, r, initial_guess, tol, max_iter)
    lower_bound, upper_bound = _call_arbitrage_bounds(S, K, T, r)
    _validate_market_price(market_price, lower_bound, upper_bound, "Call")
    return _solve_implied_vol(
        market_price=market_price,
        initial_guess=initial_guess,
        tol=tol,
        max_iter=max_iter,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        price_error=lambda sigma: call_price_error(sigma, market_price, S, K, T, r),
    )


def implied_vol_put(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    initial_guess: float = 0.2,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Return the Black-Scholes implied volatility of a European put option."""
    _validate_solver_inputs(market_price, S, K, T, r, initial_guess, tol, max_iter)
    lower_bound, upper_bound = _put_arbitrage_bounds(S, K, T, r)
    _validate_market_price(market_price, lower_bound, upper_bound, "Put")
    return _solve_implied_vol(
        market_price=market_price,
        initial_guess=initial_guess,
        tol=tol,
        max_iter=max_iter,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        price_error=lambda sigma: put_price_error(sigma, market_price, S, K, T, r),
    )
