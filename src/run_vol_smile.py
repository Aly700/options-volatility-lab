"""Run synthetic volatility-smile generation."""

from __future__ import annotations

from volatility.vol_smile import compute_implied_vol_smile, generate_option_chain, plot_volatility_smile


def main() -> None:
    """Generate a synthetic option chain, recover implied vols, and plot the smile."""
    S = 100.0
    r = 0.03
    T = 0.5
    base_sigma = 0.2
    vol_mode = "smile"

    option_chain = generate_option_chain(S=S, r=r, T=T, base_sigma=base_sigma, vol_mode=vol_mode)
    smile_df = compute_implied_vol_smile(option_chain, S=S, r=r, T=T)
    output_path = plot_volatility_smile(smile_df, vol_mode=vol_mode)
    print(f"Saved volatility smile chart to {output_path} using {vol_mode} mode")


if __name__ == "__main__":
    main()
