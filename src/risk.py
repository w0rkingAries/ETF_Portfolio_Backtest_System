from __future__ import annotations

import numpy as np
import pandas as pd


# 統計買賣交易的頻繁程度
def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    if weights.empty:
        raise ValueError("空的權重表")
    
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    turnover.iloc[0] = 0.0
    turnover.name = "Turnover"
    return turnover


# 每次買賣所需的交易成本
def transaction_cost(
    portfolio_returns: pd.Series,
    turnover: pd.Series,
    fee_rate: float = 0.001,
    tax_rate: float = 0.0,
    slippage_rate: float = 0.0005
) -> pd.Series:
    
    cost = turnover * (fee_rate + tax_rate + slippage_rate)
    return portfolio_returns - cost


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
                continue

            r = portfolio_returns.loc[dt, ticker]
            cum_asset_returns[ticker] = (1.0 + cum_asset_returns[ticker]) * (1.0 + r) - 1.0

            if cum_asset_returns[ticker] <= stop_loss_pct:
                stopped_assets.add(ticker)

                if i + 1 < len(adjusted.index):
                    next_dt = adjusted.index[i + 1]
                    adjusted.loc[next_dt:, ticker] = 0.0

    return adjusted