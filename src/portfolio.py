from __future__ import annotations

import pandas as pd
import numpy as np

from src.risk import(
    asset_stop_loss,
    apply_transaction_cost_model,
)
from src.metric import rolling_volatility

TRADING_DAYS_M = 21

# 平均分配權重給所有 ETF
def equal_weight_vector(columns: list[str]) -> pd.Series:
    n = len(columns)
    if n == 0:
        raise ValueError("資產數量不能為 0。")

    weight = 1.0 / n
    return pd.Series(weight, index=columns, name="weight")


# 根據平均分配的權重計算報酬率
def equal_weight_portfolio_returns(asset_returns: pd.DataFrame) -> pd.Series:
    weights = equal_weight_vector(asset_returns.columns.tolist())
    portfolio_returns = asset_returns.dot(weights)
    portfolio_returns.name = "equal_weight_portfolio_return"
    return portfolio_returns


# 取得每個月最一個交易日的價格
def get_month_end_rebalance_dates(prices: pd.DataFrame, freq: str  = "M") -> pd.DatetimeIndex:
    if prices.empty:
        raise ValueError("空的價格表")
    #rebalance_dates = prices.groupby(prices.index.to_period("M")).apply(lambda x: x.index[-1])
    grouped = prices.groupby(pd.Grouper(freq=freq))
    rebalance_dates = grouped.apply(lambda x: x.index[-1] if not x.empty else pd.NaT).dropna()
    return pd.DatetimeIndex(rebalance_dates.values)


# 跟據momentum score 選擇最高的 n 個 ETF
def select_top_n_assets(score_row: pd.Series, top_n: int = 3) -> list[str]:
    valid_scores = score_row.dropna()
    if valid_scores.empty:
        return []

    selected = valid_scores.sort_values(ascending=False).head(top_n).index.tolist()
    return selected


def apply_max_position_limit_to_weights(
    weights: pd.Series,
    max_weight: float | None = None,
    renormalize: bool = True,
) -> pd.Series:

    if max_weight is None:
        return weights

    if max_weight <= 0 or max_weight > 1:
        raise ValueError("max_weight 必須介於 0 和 1 之間")

    adjusted = weights.copy()
    adjusted = adjusted.clip(upper=max_weight)

    if renormalize:
        total_weight = adjusted.sum()
        if total_weight > 0:
            adjusted = adjusted / total_weight

    return adjusted


# 在每月最後一個交易日, 跟據momentum score 選擇最高的 n 個 ETF
# 每個月選擇最高的 n 個 ETF, 並平均分配權重
def build_cs_mom_weights(
    prices: pd.DataFrame,
    mom_factor: pd.DataFrame,
    lb: int,
    top_n: int = 3,
    freq: str = "M",
    weighting: str = "equal",
    max_weight: float | None = None,
    renormalize_after_limit: bool = True,
) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("空的價格表")
    if top_n <= 0:
        raise ValueError("ETF 選擇數量必須大於0")
    
    momentum = mom_factor
    rebalance_dates = get_month_end_rebalance_dates(prices, freq)

    weights = pd.DataFrame(0.0, index = prices.index, columns = prices.columns, dtype = float)
    rebalance_dates = [d for d in rebalance_dates if d in prices.index]

    for i, rebalance_date in enumerate(rebalance_dates):
        row_mom = momentum.loc[rebalance_date]
        score = row_mom.copy()
        # 避免初期指標 NaN 時出錯
        score = score.dropna()

        selected_assets = score.nlargest(top_n).index
        if selected_assets.empty:
            continue

        current_weights = pd.Series(0.0, index=prices.columns, dtype=float)

        if weighting == "equal":
            current_weights.loc[selected_assets] = 1.0 / len(selected_assets)
        elif weighting == "vol_adj":
            ret = prices.pct_change()
            vol = rolling_volatility(ret, lb * TRADING_DAYS_M)
            row_vol = vol.loc[rebalance_date, selected_assets].replace(0, pd.NA).dropna().astype(float)
            inv_vol = 1.0 / row_vol
            vol_adj_weights = (inv_vol / inv_vol.sum()).astype(float)
            current_weights.loc[vol_adj_weights.index] = vol_adj_weights

        if max_weight:
            current_weights = apply_max_position_limit_to_weights(
            current_weights,
            max_weight=max_weight,
            renormalize=renormalize_after_limit,
        )
        
        # 生效區間：下一個交易日 到 下一次 rebalance date 當日
        # t 日策略, t+1 日生效, 避免 look-ahead bias
        rebalance_loc = prices.index.get_loc(rebalance_date)
        start_loc = rebalance_loc + 1 

        if start_loc >= len(prices.index):
            continue

        if i < len(rebalance_dates) - 1:
            next_rebalance_date = rebalance_dates[i + 1]
            end_loc = prices.index.get_loc(next_rebalance_date) + 1
        else:
            end_loc = len(prices.index)

        weights.iloc[start_loc:end_loc, :] = current_weights.values

    return weights


# long only
def build_ts_mom_weights(
    prices: pd.DataFrame,
    mom_factor: pd.DataFrame,
    lb: int,
    freq: str = "M",
    weighting: str = "equal",
    max_weight: float | None = None,
    renormalize_after_limit: bool = False,
) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("空的價格表")

    momentum = mom_factor
    rebalance_dates = get_month_end_rebalance_dates(prices, freq)

    weights = pd.DataFrame(
        0.0, index=prices.index, columns=prices.columns, dtype=float
    )
    rebalance_dates = [d for d in rebalance_dates if d in prices.index]

    for i, rebalance_date in enumerate(rebalance_dates):
        row_mom = momentum.loc[rebalance_date]
        score = row_mom.copy()

        # 避免初期指標 NaN 時出錯
        final_score = score.dropna()

        # Time-series momentum:
        # 每個 asset 獨立判斷，signal > 0 就持有
        selected_assets = final_score[final_score > 0].index

        if len(selected_assets) == 0:
            continue

        current_weights = pd.Series(0.0, index=prices.columns, dtype=float)

        if weighting == "equal":
            current_weights.loc[selected_assets] = 1.0 / len(selected_assets)
        elif weighting == "vol_adj":
            ret = prices.pct_change()
            vol = rolling_volatility(ret, lb * TRADING_DAYS_M)
            row_vol = vol.loc[rebalance_date, selected_assets].replace(0, pd.NA).dropna().astype(float)
            inv_vol = 1.0 / row_vol
            vol_adj_weights = (inv_vol / inv_vol.sum()).astype(float)
            current_weights.loc[vol_adj_weights.index] = vol_adj_weights

        if max_weight:
            current_weights = apply_max_position_limit_to_weights(
            current_weights,
            max_weight=max_weight,
            renormalize=renormalize_after_limit,
        )
        
        # 生效區間：下一個交易日 到 下一次 rebalance date 當日
        # t 日策略, t+1 日生效, 避免 look-ahead bias
        rebalance_loc = prices.index.get_loc(rebalance_date)
        start_loc = rebalance_loc + 1

        if start_loc >= len(prices.index):
            continue

        if i < len(rebalance_dates) - 1:
            next_rebalance_date = rebalance_dates[i + 1]
            end_loc = prices.index.get_loc(next_rebalance_date) + 1
        else:
            end_loc = len(prices.index)

        weights.iloc[start_loc:end_loc, :] = current_weights.values

    return weights


# 根據每個月 ETF 的權重計算對應的報酬率
def compute_portfolio_returns_from_weights(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    fee_rate: float = 0.001,
    tax_rate: float = 0.0,
    slippage_rate: float = 0.0005,
) -> pd.Series:
    if returns.empty:
        raise ValueError("空的報酬表")
    if weights.empty:
        raise ValueError("空的權重表")
    # 對齊 return 與 weights 日期
    weights = weights.loc[returns.index] 
    if not returns.index.equals(weights.index):
        raise ValueError("returns.index 和 weights.index 必須一致")
    if list(returns.columns) != list(weights.columns):
        raise ValueError("returns.columns 和 weights.columns 必須一致")
    
    # 沒有設定止損
    gross_returns = (returns * weights).sum(axis=1)

    cost_detail = apply_transaction_cost_model(
        gross_returns = gross_returns,
        weights = weights,
        fee_rate = fee_rate,
        tax_rate = tax_rate,
        slippage_rate = slippage_rate,
    )

    return cost_detail


# 策略經過 asset_stop 調整後, 計算報酬率
def compute_portfolio_returns_with_stop(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    asset_stop_pct: float | None = -0.12,
    fee_rate: float = 0.001,
    tax_rate: float = 0.0,
    slippage_rate: float = 0.0005,
) -> list[pd.Series]:
    if returns.empty:
        raise ValueError("空的報酬表")
    if weights.empty:
        raise ValueError("空的權重表")
    # len(weight) 會比 len(return) 少一天, 所以要對齊
    weights = weights.loc[returns.index] 
    if not returns.index.equals(weights.index):
        raise ValueError("returns.index 和 weights.index 必須一致")
    if list(returns.columns) != list(weights.columns):
        raise ValueError("returns.columns 和 weights.columns 必須一致")

    # Asset level 止損
    asset_stop_weights = asset_stop_loss(
        portfolio_returns = returns,
        weights = weights,
        stop_loss_pct = asset_stop_pct,
    )
    asset_stop_adjust = (returns * asset_stop_weights).sum(axis=1)

    cost_detail = apply_transaction_cost_model(
        gross_returns = asset_stop_adjust,
        weights = asset_stop_weights,
        fee_rate = fee_rate,
        tax_rate = tax_rate,
        slippage_rate = slippage_rate,
    )
    
    
    return  cost_detail, asset_stop_weights