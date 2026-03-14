"""Unit tests for Black-Scholes implied volatility solvers."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pricing.black_scholes import black_scholes_call_price, black_scholes_put_price
from volatility.implied_vol import implied_vol_call, implied_vol_put


def test_implied_vol_call_recovers_sigma_from_black_scholes_price() -> None:
    sigma = 0.24
    market_price = black_scholes_call_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=sigma)
    recovered_sigma = implied_vol_call(market_price=market_price, S=100.0, K=100.0, T=1.0, r=0.05)
    assert math.isclose(recovered_sigma, sigma, rel_tol=1e-7, abs_tol=1e-7)


def test_implied_vol_put_recovers_sigma_from_black_scholes_price() -> None:
    sigma = 0.31
    market_price = black_scholes_put_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=sigma)
    recovered_sigma = implied_vol_put(market_price=market_price, S=100.0, K=100.0, T=1.0, r=0.05)
    assert math.isclose(recovered_sigma, sigma, rel_tol=1e-7, abs_tol=1e-7)


def test_recovered_sigma_is_close_to_original_sigma_within_tolerance() -> None:
    sigma = 0.18
    market_price = black_scholes_call_price(S=95.0, K=100.0, T=0.75, r=0.02, sigma=sigma)
    recovered_sigma = implied_vol_call(
        market_price=market_price,
        S=95.0,
        K=100.0,
        T=0.75,
        r=0.02,
        initial_guess=0.3,
        tol=1e-10,
    )
    assert abs(recovered_sigma - sigma) < 1e-8


def test_invalid_market_price_raises_value_error() -> None:
    with pytest.raises(ValueError):
        implied_vol_call(market_price=150.0, S=100.0, K=100.0, T=1.0, r=0.05)

    with pytest.raises(ValueError):
        implied_vol_put(market_price=-1.0, S=100.0, K=100.0, T=1.0, r=0.05)


def test_deep_itm_and_otm_cases_still_recover_sigma() -> None:
    deep_itm_call_sigma = 0.42
    deep_itm_call_price = black_scholes_call_price(S=180.0, K=100.0, T=0.5, r=0.03, sigma=deep_itm_call_sigma)
    recovered_call_sigma = implied_vol_call(
        market_price=deep_itm_call_price,
        S=180.0,
        K=100.0,
        T=0.5,
        r=0.03,
    )

    deep_otm_put_sigma = 0.27
    deep_otm_put_price = black_scholes_put_price(S=180.0, K=100.0, T=0.5, r=0.03, sigma=deep_otm_put_sigma)
    recovered_put_sigma = implied_vol_put(
        market_price=deep_otm_put_price,
        S=180.0,
        K=100.0,
        T=0.5,
        r=0.03,
    )

    assert math.isclose(recovered_call_sigma, deep_itm_call_sigma, rel_tol=1e-7, abs_tol=1e-7)
    assert math.isclose(recovered_put_sigma, deep_otm_put_sigma, rel_tol=1e-7, abs_tol=1e-7)
