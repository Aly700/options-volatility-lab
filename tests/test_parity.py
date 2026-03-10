"""Unit tests for put-call parity validation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.parity import is_put_call_parity_satisfied, put_call_parity_gap
from pricing.black_scholes import black_scholes_call_price, black_scholes_put_price


def test_put_call_parity_gap_near_zero_with_black_scholes_prices() -> None:
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    call_price = black_scholes_call_price(S=S, K=K, T=T, r=r, sigma=sigma)
    put_price = black_scholes_put_price(S=S, K=K, T=T, r=r, sigma=sigma)
    gap = put_call_parity_gap(call_price=call_price, put_price=put_price, S=S, K=K, r=r, T=T)
    assert abs(gap) < 1e-10


def test_parity_checker_returns_true_for_consistent_prices() -> None:
    S, K, T, r, sigma = 105.0, 100.0, 0.75, 0.03, 0.25
    call_price = black_scholes_call_price(S=S, K=K, T=T, r=r, sigma=sigma)
    put_price = black_scholes_put_price(S=S, K=K, T=T, r=r, sigma=sigma)
    assert is_put_call_parity_satisfied(call_price, put_price, S, K, r, T)


def test_parity_checker_returns_false_for_inconsistent_prices() -> None:
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    call_price = black_scholes_call_price(S=S, K=K, T=T, r=r, sigma=sigma)
    put_price = black_scholes_put_price(S=S, K=K, T=T, r=r, sigma=sigma) + 0.5
    assert not is_put_call_parity_satisfied(call_price, put_price, S, K, r, T, tolerance=1e-6)
