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

def compute_breadth(
    prices: pd.DataFrame,
    breadth_ma_window: int = 50,
) -> pd.Series:

    if prices.empty:
        raise ValueError("prices 為空，無法計算 breadth")

    ma = prices.rolling(breadth_ma_window).mean()
    above_ma = prices > ma
    breadth = above_ma.sum(axis=1) / above_ma.shape[1]
    breadth.name = "breadth"

    return breadth


def compute_avg_pairwise_correlation(
    prices: pd.DataFrame,
    corr_window: int = 20,
) -> pd.Series:

    if prices.empty:
        raise ValueError("prices 為空，無法計算 correlation")

    returns = prices.pct_change().dropna(how="any")
    avg_corr = pd.Series(index=returns.index, dtype=float, name="avg_corr")

    for end_idx in range(corr_window - 1, len(returns)):
        window_ret = returns.iloc[end_idx - corr_window + 1 : end_idx + 1]

        # 只保留在該 window 內無缺值的資產，避免 corr matrix 不穩
        window_ret = window_ret.dropna(axis=1, how="any")

        # 至少要兩個資產先可以計 pairwise correlation
        if window_ret.shape[1] < 2:
            avg_corr.iloc[end_idx] = np.nan
            continue

        corr_mat = window_ret.corr()

        upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
        avg_corr.iloc[end_idx] = upper.stack().mean()

    return avg_corr

def compute_regime(
    combined: pd.Series | pd.DataFrame,
    trend_ma_window: int = 200,
    vol_window: int = 20,
    vol_quantile: float = 0.7,
    vol_threshold_window: int = 252,

    prices: pd.DataFrame | None = None,
    breadth_ma_window: int = 50,
    breadth_threshold: float = 0.6,
    corr_window: int = 20,
    corr_quantile: float = 0.7,
    corr_threshold_window: int = 252,
) -> pd.DataFrame:
    
    # 還原價格趨勢
    ret = compute_daily_returns(combined)
    price = compute_growth_index(ret).iloc[:,0]

    df = pd.DataFrame(index=price.index)

    # 1) Trend: 用今日為止資料計 MA
    ma = price.rolling(trend_ma_window).mean()
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

    # 4) Breadth & Correlation
    if prices is not None:
        if prices.empty:
            raise ValueError("prices 為空，無法計算 breadth / correlation")

        prices = prices.reindex(df.index)

        # Breadth
        df["breadth"] = compute_breadth(
            prices=prices,
            breadth_ma_window=breadth_ma_window,
        )
        df["strong_breadth"] = df["breadth"] >= breadth_threshold

        # Correlation
        df["avg_corr"] = compute_avg_pairwise_correlation(
            prices=prices,
            corr_window=corr_window,
        )

        corr_pct = df["avg_corr"].rolling(corr_threshold_window).rank(pct=True)
        df["corr_pct"] = corr_pct
        df["high_corr"] = corr_pct > corr_quantile

        breadth_label = pd.Series(
            np.where(df["strong_breadth"], "Strong Breadth", "Weak Breadth"),
            index=df.index,
        )
        corr_label = pd.Series(
            np.where(df["high_corr"], "High Corr", "Low Corr"),
            index=df.index,
        )

        df["breadth_label"] = breadth_label
        df["corr_label"] = corr_label

        # 2D regime × breadth
        df["regime_x_breadth"] = (
            df["regime"].astype("object")
            + " | "
            + df["breadth_label"]
        )

        # 2D regime × correlation
        df["regime_x_corr"] = (
            df["regime"].astype("object")
            + " | "
            + df["corr_label"]
        )

        # 2D regime × breadth × correlation
        df["regime_ext"] = (
            df["regime"].astype("object")
            + " | "
            + breadth_label
            + " | "
            + corr_label
        )

        invalid_breadth_mask = (
            df["regime"].isna()
            | df["breadth"].isna()
        )
        df.loc[invalid_breadth_mask, ["breadth_label", "regime_x_breadth"]] = np.nan

        invalid_corr_mask = (
            df["regime"].isna()
            | df["avg_corr"].isna()
        )
        df.loc[invalid_corr_mask, ["corr_label", "regime_x_corr"]] = np.nan

        invalid_mask = (
            df["regime"].isna()
            | df["breadth"].isna()
            | df["avg_corr"].isna()
        )
        df.loc[invalid_mask, "regime_ext"] = np.nan

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
) -> pd.DataFrame:

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
) -> pd.DataFrame:
    
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


def analyze_regime_overlay(
    returns: pd.Series,
    regime_df: pd.DataFrame,
    base_regime_col: str = "regime",
    overlay_col: str = "strong_breadth",
) -> pd.DataFrame:

    required_cols = [base_regime_col, overlay_col]
    missing_cols = [col for col in required_cols if col not in regime_df.columns]
    if missing_cols:
        raise ValueError(f"regime_df 缺少必要欄位: {missing_cols}")

    df = pd.concat(
        [
            returns.rename("returns"),
            regime_df[base_regime_col].rename("base_regime"),
            regime_df[overlay_col].rename("overlay"),
        ],
        axis=1,
    ).dropna()

    if df.empty:
        return pd.DataFrame()

    rows = []

    for (base_regime, overlay_state), group in df.groupby(["base_regime", "overlay"]):
        r = group["returns"].dropna()

        if len(r) == 0:
            continue

        overlay_label = f"{overlay_col}={overlay_state}"

        rows.append({
            "base_regime": base_regime,
            "overlay_col": overlay_col,
            "overlay_state": overlay_state,
            #"overlay_label": overlay_label,
            "count": len(r),
            "annual_return": annualized_return(r),
            "annual_vol": annualized_volatility(r),
            "sharpe": sharpe_ratio(r),
            "max_drawdown": max_drawdown(r),
            "VaR(95%)": historical_var(r),
            "skew": r.skew(),
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["base_regime", "overlay_state"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return out


def analyze_multiple_regime_overlays(
    returns: pd.Series,
    regime_df: pd.DataFrame,
    base_regime_col: str = "regime",
    overlay_cols: list[str] | None = None,
) -> pd.DataFrame:

    if overlay_cols is None:
        overlay_cols = ["strong_breadth", "high_corr"]

    results = []

    for overlay_col in overlay_cols:
        res = analyze_regime_overlay(
            returns=returns,
            regime_df=regime_df,
            base_regime_col=base_regime_col,
            overlay_col=overlay_col,
        )
        if not res.empty:
            results.append(res)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, axis=0, ignore_index=True)

def analyze_multiple_strategies_regime_overlays(
    strategy_returns: dict[str, pd.Series],
    regime_df: pd.DataFrame,
    base_regime_col: str = "regime",
    overlay_cols: list[str] = ["strong_breadth", "high_corr"],
) -> pd.DataFrame:

    results = []
    aligned_index = regime_df.index

    for strategy_name, returns in strategy_returns.items():
        aligned_returns = returns.reindex(aligned_index)

        for overlay_col in overlay_cols:
            res = analyze_regime_overlay(
                returns=aligned_returns,
                regime_df=regime_df,
                base_regime_col=base_regime_col,
                overlay_col=overlay_col,
            )
            if res.empty:
                continue

            res.insert(0, "strategy", strategy_name)
            results.append(res)

    if not results:
        return pd.DataFrame()

    return (
        pd.concat(results, axis=0, ignore_index=True)
        .set_index(["strategy", "base_regime", "overlay_col", "overlay_state"])
        .sort_index()
    )


def analyze_trade_stats_regime_overlay(
    returns: pd.Series,
    weights: pd.DataFrame,
    regime_df: pd.DataFrame,
    base_regime_col: str = "regime",
    overlay_col: str = "strong_breadth",
) -> pd.DataFrame:

    required_cols = [base_regime_col, overlay_col]
    missing_cols = [col for col in required_cols if col not in regime_df.columns]
    if missing_cols:
        raise ValueError(f"regime_df 缺少必要欄位: {missing_cols}")

    df = pd.concat(
        [
            returns.rename("returns"),
            regime_df[base_regime_col].rename("base_regime"),
            regime_df[overlay_col].rename("overlay"),
            weights,
        ],
        axis=1,
    ).dropna()

    if df.empty:
        return pd.DataFrame()

    rows = []

    for (base_regime, overlay_state), group in df.groupby(["base_regime", "overlay"]):
        r = group["returns"].dropna()
        w = group.drop(columns=["returns", "base_regime", "overlay"])

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
            "base_regime": base_regime,
            "overlay_col": overlay_col,
            "overlay_state": overlay_state,
            "count": len(r),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "payoff_ratio": payoff_ratio,
            "profit_factor": profit_factor,
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            by=["base_regime", "overlay_state"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


def analyze_multiple_strategies_trade_stats_regime_overlays(
    combined: dict[str, list[pd.Series | pd.DataFrame]],
    regime_df: pd.DataFrame,
    base_regime_col: str = "regime",
    overlay_cols: list[str] = ["strong_breadth", "high_corr"],
) -> pd.DataFrame:

    results = []
    aligned_index = regime_df.index

    for strategy_name, extract in combined.items():
        returns, weights = extract[0], extract[1]
        aligned_returns = returns.reindex(aligned_index)
        aligned_weights = weights.reindex(aligned_index)

        for overlay_col in overlay_cols:
            res = analyze_trade_stats_regime_overlay(
                returns=aligned_returns,
                weights=aligned_weights,
                regime_df=regime_df,
                base_regime_col=base_regime_col,
                overlay_col=overlay_col,
            )
            if res.empty:
                continue

            res.insert(0, "strategy", strategy_name)
            results.append(res)

    if not results:
        return pd.DataFrame()

    return (
        pd.concat(results, axis=0, ignore_index=True)
        .set_index(["strategy", "base_regime", "overlay_col", "overlay_state"])
        .sort_index()
    )
