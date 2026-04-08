from __future__ import annotations

import pandas as pd
import numpy as np

from src.metric import(
    compute_daily_returns,
    compute_growth_index,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    historical_var,
    extract_trade_returns,
)

def compute_regime(
    combined: pd.Series,
    ma_window: int = 200,
    vol_window: int = 20,
    vol_quantile: float = 0.7,
    vol_threshold_window: int = 252,
) -> pd.DataFrame:
    
    # 還原價格趨勢
    ret = compute_daily_returns(combined)
    price = compute_growth_index(ret).iloc[:,0]

    df = pd.DataFrame(index=price.index)

    # 1) Trend: 用今日為止資料計 MA
    ma = price.rolling(ma_window).mean()
    trend_strength = price / ma - 1.0
    df["trend"] = trend_strength > 0.0

    # 2) Volatility: 用過去報酬計波動
    df["vol"] = ret.rolling(vol_window).std()

    # 每天計算當日的波動率落在觀測日期內的位置
    vol_pct = df["vol"].rolling(vol_threshold_window).rank(pct = True)
    df["high_vol"] = vol_pct > vol_quantile

    # 3) 4 regimes
    conditions = [
        df["trend"] & ~df["high_vol"],
        df["trend"] & df["high_vol"],
        ~df["trend"] & ~df["high_vol"],
        ~df["trend"] & df["high_vol"],
    ]
    choices = [
        "Bull Low Vol",
        "Bull High Vol",
        "Bear Low Vol",
        "Bear High Vol",
    ]

    df["regime"] = np.select(conditions, choices, default=np.nan)

    return df


def evaluate_strategy_by_regime(
    name: str,
    returns: pd.Series,
    regime_series: pd.Series,
) -> pd.DataFrame:
    
    df = pd.concat(
        [
            returns.rename("returns"),
            regime_series.rename("regime"),
        ],
        axis=1,
    ).dropna()

    if df.empty:
        return pd.DataFrame()

    rows = []

    for regime_name, group in df.groupby("regime"):
        r = group["returns"].dropna()

        if len(r) == 0:
            continue
        
        rows.append({
                "strategy": name,
                "regime": regime_name,
                "count": len(r),
                "annual_return": annualized_return(r),
                "annual_vol": annualized_volatility(r),
                "sharpe": sharpe_ratio(r),
                "max_drawdown": max_drawdown(r),
                "VaR(95%)": historical_var(r),
                "skew": r.skew(),
        })

    return pd.DataFrame(rows)


def evaluate_multiple_strategies_by_regime(
    strategy_returns: dict[str, pd.Series],
    regime_series: pd.Series,
) -> list[pd.DataFrame]:

    res, results = [], []

    for name, returns in strategy_returns.items():
        res = evaluate_strategy_by_regime(
            name = name,
            returns = returns,
            regime_series = regime_series,
        )
        if not res.empty:
            results.append(res)

    metric = pd.concat(results, axis=0, ignore_index=True)
    metric = metric.set_index(["strategy", "regime"]).sort_index()

    return metric


def compute_trade_stats_by_regime(
    name: str,
    returns: pd.Series,
    weights: pd.Series,
    regime_series: pd.Series,
) -> pd.DataFrame:

    df = pd.concat(
        [
            returns.rename("returns"),
            regime_series.rename("regime"),
            weights,
        ],
        axis=1,
    ).dropna()

    if df.empty:
        return pd.DataFrame()

    rows = []

    for regime_name, group in df.groupby("regime"):
        r = group["returns"].dropna()
        w = group.drop(columns=["returns", "regime"])

        if len(r) == 0:
            continue

        trades = extract_trade_returns(r, w)
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]

        win_rate = len(wins) / len(trades) if trades else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else None
        profit_factor = abs(sum(wins) / sum(losses)) if losses else None

        rows.append({
            "strategy": name,
            "regime": regime_name,
            "count": len(r),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "payoff_ratio": payoff_ratio,
            "profit_factor": profit_factor,
        })
        

    return pd.DataFrame(rows)


def evaluate_trade_stats_by_regime(
    combined: dict[str, list[pd.Series]],
    regime_series: pd.Series,
) -> list[pd.DataFrame]:
    
    res, results = [], []

    for name, extract in combined.items():
        returns, weights = extract[0], extract[1]
        res = compute_trade_stats_by_regime(
            name = name,
            returns = returns,
            weights = weights,
            regime_series = regime_series,
        )
        if not res.empty:
            results.append(res)

    trade = pd.concat(results, axis=0, ignore_index=True)
    trade = trade.set_index(["strategy", "regime"]).sort_index()

    return trade