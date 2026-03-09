"""Unit tests for foundational Black-Scholes pricing behavior."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pricing.black_scholes import black_scholes_call_price, black_scholes_put_price


def test_call_price_positive() -> None:
    price = black_scholes_call_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
    assert price > 0.0


def test_put_price_positive() -> None:
    price = black_scholes_put_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
    assert price > 0.0


def test_call_price_increases_with_spot() -> None:
    low_spot_price = black_scholes_call_price(S=90.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
    high_spot_price = black_scholes_call_price(S=110.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
    assert high_spot_price > low_spot_price


def test_put_price_decreases_with_spot() -> None:
    low_spot_price = black_scholes_put_price(S=90.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
    high_spot_price = black_scholes_put_price(S=110.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
    assert high_spot_price < low_spot_price
