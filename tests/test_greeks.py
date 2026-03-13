"""Unit tests for Black-Scholes Greeks."""

from __future__ import annotations

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hedging.greeks import (
    call_delta,
    call_rho,
    call_theta,
    gamma,
    put_delta,
    put_rho,
    put_theta,
    vega,
)
from pricing.black_scholes import black_scholes_call_price


def test_call_delta_is_between_zero_and_one() -> None:
    value = call_delta(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
    assert 0.0 < value < 1.0


def test_put_delta_is_between_negative_one_and_zero() -> None:
    value = put_delta(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
    assert -1.0 < value < 0.0


def test_gamma_is_positive() -> None:
    assert gamma(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2) > 0.0


def test_vega_is_positive() -> None:
    assert vega(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2) > 0.0


def test_call_rho_is_positive() -> None:
    assert call_rho(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2) > 0.0


def test_put_rho_is_negative() -> None:
    assert put_rho(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2) < 0.0


def test_call_theta_is_typically_negative() -> None:
    assert call_theta(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2) < 0.0


def test_put_theta_is_typically_negative_for_standard_parameters() -> None:
    assert put_theta(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2) < 0.0


def test_call_delta_matches_finite_difference_sanity_check() -> None:
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    step = 1e-4
    analytic_delta = call_delta(S=S, K=K, T=T, r=r, sigma=sigma)
    finite_difference_delta = (
        black_scholes_call_price(S=S + step, K=K, T=T, r=r, sigma=sigma)
        - black_scholes_call_price(S=S - step, K=K, T=T, r=r, sigma=sigma)
    ) / (2.0 * step)
    assert math.isclose(analytic_delta, finite_difference_delta, rel_tol=1e-5, abs_tol=1e-5)
