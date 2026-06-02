# ETF Portfolio Backtest System

## 1. Introduction

This project is a research-oriented ETF momentum backtest system. It builds a full workflow from ETF data preparation, market regime classification, factor validation, strategy construction, trading analysis, and cost-aware performance evaluation.

The current version focuses on two momentum families:

- **Cross-sectional momentum (CS-MOM)**: ranks ETFs against each other and rotates into the strongest names.
- **Time-series momentum (TS-MOM)**: evaluates each ETF against its own trend signal and allocates when momentum is positive.

The system is designed for research iteration rather than one-click production trading. Most outputs are saved as CSV/TXT reports, and `web_visualize.py` converts the latest outputs into a local HTML dashboard.

### Run the Workflow

From the project root, run both strategy pipelines before generating the dashboard:

```bash
python cs_mom.py
python ts_mom.py
python web_visualize.py
```

The first two commands download the required data, run the CS-MOM and TS-MOM backtests, and refresh the CSV/TXT reports. The final command reads the latest report artifacts and rebuilds `report/dashboard.html`.

---

## 2. Research Flow

### I. Data

- ETF universe: default ETF list from `src/data_preprocessing.py`
- Benchmark examples: `0050.TW` and `SPY`
- Price handling:
  - download ETF data
  - clean missing rows
  - forward-fill missing values
  - compute daily returns and growth index

### II. Regime

Market regime is used as the main lens for understanding when momentum works or fails.

Current regime dimensions include:

- Bull / Bear
- High Vol / Low Vol
- Breadth overlay
- Correlation overlay

### III. Factor Validation

Momentum factors are validated before strategy results are interpreted.

The validation pack includes:

- IC summary
- IC by regime
- lookback robustness
- minimum-observation robustness
- rolling out-of-sample validation

### IV. Strategy

The system tests several strategy variants:

- raw equal-weight momentum
- volatility-adjusted momentum
- capped volatility-adjusted momentum
- optional asset-stop variants
- monthly, 2-month, and 3-month rebalance frequencies

### V. Evaluation

Strategy performance is evaluated across:

- annual return
- volatility
- Sharpe ratio
- maximum drawdown
- VaR
- trade statistics
- turnover
- transaction cost
- net-vs-gross return drag

---

## 3. Dashboard

The dashboard presents the full research view through three sections:

- **Overview**: net performance highlights, equity curves, strategy ranking, and rolling out-of-sample results
- **Regime**: regime distribution, performance metrics, and trade statistics by entry regime
- **Cost**: turnover, transaction cost, return drag, and Sharpe drag

The top-level tabs switch between CS-MOM and TS-MOM. Inside each strategy family, the toggle bar switches between `Overview`, `Regime`, and `Cost`. Each section places its matching `*_*_Conclusion.txt` research note directly below the section heading.

---

## 4. Project Structure

```text
ETF_Portfolio_Backtest_System/
├── cs_mom.py                         # Cross-sectional momentum runner
├── ts_mom.py                         # Time-series momentum runner
├── web_visualize.py                  # Local HTML dashboard generator
├── requirement.txt                   # Python dependencies
├── data/
│   ├── ETF/                          # Cleaned ETF and benchmark data
│   └── strategy/                     # Strategy returns and weights
├── report/
│   ├── cross_sectional_momentum/     # CS-MOM CSV/TXT reports
│   ├── time_series_momentum/         # TS-MOM CSV/TXT reports
│   └── dashboard.html                # Generated dashboard
└── src/
    ├── data_preprocessing.py
    ├── indicators.py
    ├── market_regime.py
    ├── metric.py
    ├── portfolio.py
    ├── reporting.py
    ├── risk.py
    └── wrapper.py
```

---

## 5. Key Outputs

### Cross-Sectional Momentum

```text
report/cross_sectional_momentum/
data/strategy/cross_sectional_momentum/
```

Important files:

- `0050.TW_metrics_by_regime.csv`
- `0050.TW_metrics_by_multi_regime.csv`
- `0050.TW_trade_stats_by_regime.csv`
- `0050.TW_trade_stats_by_multi_regime.csv`
- `0050.TW_risk_cost_summary.csv`
- `0050.TW_Rolling_OOS_summary.csv`
- `strategy_returns.csv`

### Time-Series Momentum

```text
report/time_series_momentum/
data/strategy/time_series_momentum/
```

Important files:

- `0050.TW_ts_metrics_by_regime.csv`
- `0050.TW_ts_metrics_by_multi_regime.csv`
- `0050.TW_ts_trade_stats_by_regime.csv`
- `0050.TW_ts_trade_stats_by_multi_regime.csv`
- `0050.TW_ts_risk_cost_summary.csv`
- `0050.TW_ts_Rolling_OOS_summary.csv`
- `ts_strategy_returns.csv`

---

## 6. Conclusion

- Momentum should be judged by regime, not only by full-period performance.
- Gross performance can look attractive before turnover and transaction cost are included.
- Rolling out-of-sample validation helps separate robust parameter choices from overfit settings.
- Volatility adjustment improves the CS-MOM risk-return balance.
- Asset-level stop-loss overlays are not consistently beneficial.
- Transaction costs reduce returns but do not reverse the main ranking.
