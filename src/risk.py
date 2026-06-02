from __future__ import annotations

import numpy as np
import pandas as pd

from src.metric import(
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    historical_var,
    TRADING_DAYS,
)

# 統計買賣交易的頻繁程度
def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    if weights.empty:
        raise ValueError("空的權重表")
    
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    turnover.name = "Turnover"
    return turnover


# 每次買賣所需的交易成本
def apply_transaction_cost_model(
    gross_returns: pd.Series,
    weights: pd.DataFrame,
    fee_rate: float = 0.001,
    tax_rate: float = 0.0,
    slippage_rate: float = 0.0005,
) -> pd.DataFrame:

    turnover = compute_turnover(weights)

    total_cost_rate = fee_rate + tax_rate + slippage_rate
    transaction_cost = turnover * total_cost_rate

    net_returns = gross_returns - transaction_cost
    net_returns.name = "net_return"

    cost_detail = pd.DataFrame({
        "gross_return": gross_returns,
        "turnover": turnover,
        "transaction_cost": transaction_cost,
        "net_return": net_returns,
    })

    return cost_detail


def compute_risk_cost_summary(
    cost_detail: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    整合策略的報酬、風險、成本與執行風險。
    """
    summary = []

    turnover = cost_detail["turnover"]
    transaction_cost = cost_detail["transaction_cost"]
    gross_returns = cost_detail["gross_return"]
    net_returns = cost_detail["net_return"]
    observed_months = len(
        pd.period_range(
            start=turnover.index.min(),
            end=turnover.index.max(),
            freq="M",
        )
    )
    if observed_months == 0:
        raise ValueError("cost_detail must cover at least one month")

    annualized_turnover = turnover.sum() * 12.0 / observed_months

    summary.append({
        # gross performance
        "gross_annual_return": annualized_return(gross_returns),
        "gross_volatility": annualized_volatility(gross_returns),
        "gross_sharpe": sharpe_ratio(gross_returns, risk_free_rate=risk_free_rate),
        "gross_mdd": max_drawdown(gross_returns),
        "gross_var_95": historical_var(gross_returns, confidence_level=0.95),

        # net performance
        "net_annual_return": annualized_return(net_returns),
        "net_volatility": annualized_volatility(net_returns),
        "net_sharpe": sharpe_ratio(net_returns, risk_free_rate=risk_free_rate),
        "net_mdd": max_drawdown(net_returns),
        "net_var_95": historical_var(net_returns, confidence_level=0.95),

        # cost / execution
        "avg_daily_turnover": turnover.mean(),
        "annualized_turnover": annualized_turnover,
        "total_transaction_cost": transaction_cost.sum(),
        "avg_daily_cost": transaction_cost.mean(),

        # cost impact
        "return_drag": annualized_return(gross_returns) - annualized_return(net_returns),
        "sharpe_drag": sharpe_ratio(gross_returns, risk_free_rate=risk_free_rate) - sharpe_ratio(net_returns, risk_free_rate=risk_free_rate),
    })

    return pd.DataFrame(summary)


def compute_multiple_risk_cost_summary(
    strategies: dict[str, pd.DataFrame],
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:

    results = []

    for strategy_name, data in strategies.items():
        summary = compute_risk_cost_summary(
            cost_detail=data,
            risk_free_rate=risk_free_rate,
        )
        summary.insert(0, "strategy", strategy_name)
        results.append(summary)

    if not results:
        return pd.DataFrame()

    return (
        pd.concat(results, axis=0, ignore_index=True)
        .set_index("strategy")
        .sort_index()
    )


def asset_stop_loss(
    portfolio_returns: pd.DataFrame,
    weights: pd.DataFrame,
    stop_loss_pct: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    在每個 ETF 持有週期應用資產等級止損
    """
    adjusted = weights.copy()

    if stop_loss_pct is None:
        return adjusted

    rebalance_flag = weights.diff().abs().sum(axis=1) > 1e-12

    cum_asset_returns = {col: 0.0 for col in weights.columns}
    stopped_assets: set[str] = set()

    for i, dt in enumerate(weights.index):
        if rebalance_flag.loc[dt]:
            cum_asset_returns = {col: 0.0 for col in weights.columns}
            stopped_assets = set()

        current_weights = adjusted.loc[dt].copy()

        for ticker in weights.columns:
            # 沒有買入該 ETF 或已經止損賣出, 不更新累積報酬
            if current_weights[ticker] <= 0:
                continue

            if ticker in stopped_assets:
                adjusted.loc[dt, ticker] = 0.0
                continue

            r = portfolio_returns.loc[dt, ticker]
            cum_asset_returns[ticker] = (1.0 + cum_asset_returns[ticker]) * (1.0 + r) - 1.0

            if cum_asset_returns[ticker] <= stop_loss_pct:
                stopped_assets.add(ticker)

                # if i + 1 < len(adjusted.index):
                #     next_dt = adjusted.index[i + 1]
                #     adjusted.loc[next_dt:, ticker] = 0.0

    return adjusted
