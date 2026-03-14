"""Volatility smile generation utilities."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from pricing.black_scholes import _validate_finite, _validate_positive_finite, black_scholes_call_price
from volatility.implied_vol import implied_vol_call

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MPL_CONFIG_DIR = _PROJECT_ROOT / "outputs/.matplotlib"
_XDG_CACHE_DIR = _PROJECT_ROOT / "outputs/.cache"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_OUTPUT_PATH = _PROJECT_ROOT / "outputs/charts/volatility_smile.png"
_SUPPORTED_VOL_MODES = {"flat", "smile", "skew"}


def _validate_chain_inputs(
    S: float,
    r: float,
    T: float,
    base_sigma: float,
    num_strikes: int,
    strike_range: tuple[float, float],
    vol_mode: str,
) -> None:
    """Validate option-chain generation inputs."""
    _validate_positive_finite(S, "S")
    _validate_finite(r, "r")
    _validate_finite(T, "T")
    if T <= 0.0:
        raise ValueError("T must be greater than 0.")

    _validate_finite(base_sigma, "base_sigma")
    if base_sigma <= 0.0:
        raise ValueError("base_sigma must be greater than 0.")

    if isinstance(num_strikes, bool) or not isinstance(num_strikes, int) or num_strikes < 2:
        raise ValueError("num_strikes must be an integer greater than or equal to 2.")

    if len(strike_range) != 2:
        raise ValueError("strike_range must contain exactly two values.")

    lower_multiple, upper_multiple = strike_range
    _validate_positive_finite(lower_multiple, "strike_range[0]")
    _validate_positive_finite(upper_multiple, "strike_range[1]")
    if lower_multiple >= upper_multiple:
        raise ValueError("strike_range lower bound must be less than upper bound.")

    if vol_mode not in _SUPPORTED_VOL_MODES:
        raise ValueError(f"vol_mode must be one of {sorted(_SUPPORTED_VOL_MODES)}.")


def _validate_smile_inputs(df: pd.DataFrame, S: float, r: float, T: float) -> None:
    """Validate implied-vol smile computation inputs."""
    if "strike" not in df.columns or "call_price" not in df.columns:
        raise ValueError("DataFrame must contain 'strike' and 'call_price' columns.")

    _validate_positive_finite(S, "S")
    _validate_finite(r, "r")
    _validate_finite(T, "T")
    if T <= 0.0:
        raise ValueError("T must be greater than 0.")


def _strike_volatility(K: float, S: float, base_sigma: float, vol_mode: str) -> float:
    """Return the strike-dependent volatility for a synthetic chain."""
    moneyness = K / S

    if vol_mode == "flat":
        return base_sigma
    if vol_mode == "smile":
        return base_sigma + 0.12 * (moneyness - 1.0) ** 2
    if vol_mode == "skew":
        return base_sigma + 0.10 * max(0.0, 1.0 - moneyness)

    raise ValueError(f"Unsupported vol_mode: {vol_mode}")


def generate_option_chain(
    S: float,
    r: float,
    T: float,
    base_sigma: float,
    num_strikes: int = 21,
    strike_range: tuple[float, float] = (0.5, 1.5),
    vol_mode: str = "flat",
) -> pd.DataFrame:
    """Generate a synthetic European call option chain around spot."""
    _validate_chain_inputs(S, r, T, base_sigma, num_strikes, strike_range, vol_mode)
    lower_multiple, upper_multiple = strike_range
    strikes = np.linspace(lower_multiple * S, upper_multiple * S, num=num_strikes)
    true_sigmas = [_strike_volatility(float(strike), S, base_sigma, vol_mode) for strike in strikes]
    call_prices = [
        black_scholes_call_price(S=S, K=float(strike), T=T, r=r, sigma=float(sigma_k))
        for strike, sigma_k in zip(strikes, true_sigmas)
    ]
    chain = pd.DataFrame({"strike": strikes, "call_price": call_prices, "true_sigma": true_sigmas})
    chain.attrs["vol_mode"] = vol_mode
    return chain


def compute_implied_vol_smile(df: pd.DataFrame, S: float, r: float, T: float) -> pd.DataFrame:
    """Append Black-Scholes implied volatility to an option-chain DataFrame."""
    _validate_smile_inputs(df, S, r, T)
    smile_df = df.copy()
    smile_df["implied_vol"] = [
        implied_vol_call(market_price=float(call_price), S=S, K=float(strike), T=T, r=r)
        for strike, call_price in zip(smile_df["strike"], smile_df["call_price"])
    ]
    return smile_df


def plot_volatility_smile(df: pd.DataFrame, vol_mode: str = "flat") -> Path:
    """Plot strike versus implied volatility and save the chart."""
    if "strike" not in df.columns or "implied_vol" not in df.columns:
        raise ValueError("DataFrame must contain 'strike' and 'implied_vol' columns.")

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        df["strike"].to_numpy(),
        df["implied_vol"].to_numpy(),
        marker="o",
        linewidth=1.5,
        label="Implied vol",
    )
    if "true_sigma" in df.columns:
        ax.plot(
            df["strike"].to_numpy(),
            df["true_sigma"].to_numpy(),
            linestyle="--",
            linewidth=1.5,
            label="True sigma",
        )
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Volatility")
    ax.set_title(f"Volatility Smile ({vol_mode} mode)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(_OUTPUT_PATH, dpi=150)
    plt.close(fig)
    return _OUTPUT_PATH
