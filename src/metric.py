from __future__ import annotations

import math
import numpy as np
import pandas as pd

TRADING_DAYS = 252

# 計算 ETF 的每日報酬率
def compute_daily_returns(combined: pd.DataFrame) -> pd.DataFrame:
    prices, dividends = combined["Close"], combined["Dividends"]
    returns = (prices.pct_change() + dividends / prices.shift(1)).dropna(how="any")
    return returns


# 計算報酬率的漲跌波動
def compute_growth_index(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    growth = initial_value * (1.0 + returns).cumprod()
    growth.name = "growth_index"
    return growth


# 年化報酬率
def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return float("nan")

    total_growth = (1.0 + returns).prod()
    n_periods = len(returns)

    if total_growth <= 0:
        return float("nan")

    ann_return = total_growth ** (periods_per_year / n_periods) - 1.0
    return float(ann_return)


# 年化波動率
def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")

    vol = returns.std(ddof=1) * math.sqrt(periods_per_year)
    return float(vol)


# 利用年化報酬率和年化波動率計算 Sharpe ratio
def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    ann_ret = annualized_return(returns, periods_per_year=periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year=periods_per_year)

    if np.isnan(ann_ret) or np.isnan(ann_vol) or ann_vol == 0:
        return float("nan")

    return float((ann_ret - risk_free_rate) / ann_vol)


# 計算最大回撤
def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    if dd.empty:
        return np.nan
    return float(dd.min())


# 回傳完整的回撤記錄
def drawdown_series(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    returns = returns.dropna()
    wealth_index = initial_value * (1.0 + returns).cumprod()
    running_peak = wealth_index.cummax()
    drawdown = wealth_index / running_peak - 1.0
    drawdown.name = "drawdown"
    return drawdown


# 計算風險值, 在95%時間的損失幅度, 但無法得知最大虧損(fat tail)
def historical_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return float("nan")

    alpha = 1.0 - confidence_level
    var_threshold = np.quantile(returns, alpha)
    return float(-var_threshold)


# 計算滾動波動率
def rolling_volatility(
    returns: pd.Series,
    window: int = 3,
    periods_per_year: int = TRADING_DAYS
) -> pd.Series:
    vol = returns.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)
    vol.name = f"RollingVol_{window}"
    return vol


# 根據完整時間線上的權重變化切分交易區間
def extract_trades(
    returns: pd.Series,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["entry_date", "exit_date", "trade_return"]

    if returns.empty or weights.empty:
        return pd.DataFrame(columns=columns)

    returns = returns.reindex(weights.index)
    if returns.isna().any():
        raise ValueError("returns and weights must have the same index")

    trades = []
    entry_date = None
    exit_date = None
    cum_return = 0.0

    rebalance_flag = weights.diff().abs().sum(axis=1) > 1e-8
    active_flag = weights.abs().sum(axis=1) > 1e-8

    for t in returns.index:
        if rebalance_flag.loc[t] and entry_date is not None:
            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "trade_return": cum_return,
            })
            entry_date = None
            exit_date = None
            cum_return = 0.0

        if not active_flag.loc[t]:
            continue

        if entry_date is None:
            entry_date = t
        r = returns.loc[t]
        cum_return = (1 + cum_return) * (1 + r) - 1
        exit_date = t

    if entry_date is not None:
        trades.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "trade_return": cum_return,
        })

    return pd.DataFrame(trades, columns=columns)


# 保留只回傳盈虧的介面，供整體交易統計使用
def extract_trade_returns(
    returns: pd.Series,
    weights: pd.DataFrame
) -> list[float]:
    trades = extract_trades(returns=returns, weights=weights)
    return trades["trade_return"].tolist()


# 計算各項交易統計數據
def compute_trade_stats(
    portfolios: dict[str,tuple[pd.Series, pd.DataFrame]],
) -> pd.DataFrame:
    rows = {}
    for name, extract in portfolios.items():
        series, weights = extract[0], extract[1]
        trades = extract_trade_returns(series, weights)
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]

        win_rate = len(wins) / len(trades) if trades else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else None
        profit_factor = abs(sum(wins) / sum(losses)) if losses else None

        rows[name] = pd.Series({
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "payoff_ratio": payoff_ratio,
            "profit_factor": profit_factor,
        })

    return pd.DataFrame(rows).T
