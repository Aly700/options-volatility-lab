# Options Volatility Lab

A research environment for studying option pricing, implied volatility, Greeks, and dynamic hedging strategies.

## Current Status

The project currently includes:

- Black-Scholes pricing
- payoff functions
- put-call parity validation
- Greeks calculation
- implied volatility solver
- unit tests

## Volatility Smile

The project can generate a synthetic call option chain around spot, extract implied volatility from those prices, and save a volatility smile chart to `outputs/charts/volatility_smile.png`.

The initial version used a constant volatility to generate all synthetic option prices. Since implied volatility was then recovered using the same Black-Scholes model, the recovered implied vol was constant across strikes. This produced a flat line, which is correct under constant-vol Black-Scholes but does not resemble real market option data.

Real markets typically exhibit volatility smiles or skews. To make the project more realistic and visually informative, the synthetic option chain now supports strike-dependent vol structures so the implied volatility extraction step can recover a non-flat curve that is more useful for research and demonstration.

Supported modes:

- `flat`
- `smile`
- `skew`

This project is designed to support:

- Black-Scholes option pricing
- implied volatility estimation
- volatility smile and surface construction
- Greeks analysis
- delta hedging simulation
- PnL attribution and risk reporting

## Planned Features

- European call and put pricing engine
- implied volatility solvers
- volatility smile and surface visualization
- Greeks calculator
- delta hedging simulator
- scenario stress testing
- markdown research reports

## Repository Structure

- `config/` model and experiment settings
- `data/` raw and processed market or synthetic data
- `src/pricing/` option pricing logic
- `src/volatility/` implied vol and surface tools
- `src/hedging/` Greeks and delta hedging simulation
- `src/analytics/` PnL and risk analysis
- `src/reporting/` plots and report generation
- `tests/` unit tests
- `docs/` design and research notes

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- SciPy
- PyTest
