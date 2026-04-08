from __future__ import annotations

import pandas as pd
import numpy as np
import scipy.stats as st


from src.indicators import cross_sectional_momentum, time_series_momentum


def compute_forward_return(
    prices: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    if horizon <= 0:
        raise ValueError("horizon must be > 0")

    fwd_ret = prices.shift(-horizon) / prices - 1.0
    return fwd_ret


def build_factor_panel(
    factor: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    regime_series: pd.Series,
) -> pd.DataFrame:
    if not factor.index.equals(fwd_ret.index):
        raise ValueError("factor and fwd_ret must have the same index")
    if not factor.columns.equals(fwd_ret.columns):
        raise ValueError("factor and fwd_ret must have the same columns")

    factor_long = factor.stack(dropna=False).rename("factor")
    ret_long = fwd_ret.stack(dropna=False).rename("future_return")

    panel = pd.concat([factor_long, ret_long], axis=1).reset_index()
    panel.columns = ["date", "asset", "factor", "future_return"]

    regime = regime_series.rename("regime").reset_index()
    regime.columns = ["date", "regime"]

    panel = panel.merge(regime, on="date", how="left")
    panel = panel.dropna(subset=["factor", "future_return", "regime"]).copy()

    return panel


def compute_ic(
    panel: pd.DataFrame,
    mode: str = "cross_sectional",
    method: str = "spearman",
    min_obs: int = 5,
) -> pd.DataFrame:
    if mode not in {"cross_sectional", "time_series"}:
        raise ValueError("mode must be 'cross_sectional' or 'time_series'")
    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'")

    group_key = "date" if mode == "cross_sectional" else "asset"
    records = []

    for key, group in panel.groupby(group_key):
        group = group.dropna(subset=["factor", "future_return"])
        n_obs = len(group)

        if n_obs < min_obs:
            continue

        if method == "spearman":
            ic, p_value = st.spearmanr(group["factor"], group["future_return"], nan_policy="omit")
        else:
            ic, p_value = st.pearsonr(group["factor"], group["future_return"])

        record = {
            group_key: key,
            "ic": ic,
            "p_value": p_value,
            "n_obs": n_obs,
        }

        if mode == "cross_sectional":
            regime_values = group["regime"].dropna().unique()
            if len(regime_values) == 0:
                continue
            record["regime"] = regime_values[0]

        records.append(record)

    result = pd.DataFrame(records)

    if result.empty:
        cols = [group_key, "ic", "p_value", "n_obs"]
        if mode == "cross_sectional":
            cols.insert(1, "regime")
        return pd.DataFrame(columns=cols)

    return result.sort_values(group_key).reset_index(drop=True)


def summarize_ic(
    ic_table: pd.DataFrame,
    mode: str,
) -> pd.Series:
    
    ic_series = ic_table["ic"].dropna()
    std_ic = ic_series.std(ddof=1)
    count_name = "num_periods" if mode == "cross_sectional" else "num_assets"

    p_vals = ic_table["p_value"].dropna() if "p_value" in ic_table.columns else pd.Series(dtype=float)

    return pd.Series({
        "mean_ic": ic_series.mean(),
        "std_ic": std_ic,
        "ic_ir": ic_series.mean() / std_ic if pd.notna(std_ic) and std_ic != 0 else np.nan,
        "pos_ic_ratio": (ic_series > 0).mean(),
        "sig_ic_ratio": (p_vals < 0.05).mean() if len(p_vals) > 0 else np.nan,
        "mean_p_value": p_vals.mean() if len(p_vals) > 0 else np.nan,
        count_name: ic_series.shape[0],
    })


def summarize_ic_by_regime(
    panel: pd.DataFrame,
    mode: str = "time_series",
    method: str = "spearman",
    min_obs: int = 21,
) -> pd.DataFrame:
    if mode not in {"cross_sectional", "time_series"}:
        raise ValueError("mode must be 'cross_sectional' or 'time_series'")
    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'")

    records = []

    if mode == "cross_sectional":
        for regime, group in panel.groupby("regime"):
            group = group.dropna(subset=["factor", "future_return"])
            for date, date_group in group.groupby("date"):
                n_obs = len(date_group)
                if n_obs < min_obs:
                    continue

                if method == "spearman":
                    ic, p_value = st.spearmanr(date_group["factor"], date_group["future_return"], nan_policy="omit")
                else:
                    ic, p_value = st.pearsonr(date_group["factor"], date_group["future_return"])

                records.append({
                    "regime": regime,
                    "unit": date,
                    "ic": ic,
                    "p_value": p_value,
                    "n_obs": n_obs,
                })

        unit_name = "num_periods"

    else:
        for regime, group in panel.groupby("regime"):
            group = group.dropna(subset=["factor", "future_return"])
            for asset, asset_group in group.groupby("asset"):
                asset_group = asset_group.sort_values("date")
                n_obs = len(asset_group)
                if n_obs < min_obs:
                    continue

                if method == "spearman":
                    ic, p_value = st.spearmanr(asset_group["factor"], asset_group["future_return"], nan_policy="omit")
                else:
                    ic, p_value = st.pearsonr(asset_group["factor"], asset_group["future_return"])

                records.append({
                    "regime": regime,
                    "unit": asset,
                    "ic": ic,
                    "p_value": p_value,
                    "n_obs": n_obs,
                })

        unit_name = "num_assets"

    if not records:
        return pd.DataFrame(columns=[
            "regime", "mean_ic", "std_ic", "ic_ir",
            "pos_ic_ratio", "sig_ic_ratio", 
            "mean_p_value", unit_name, "avg_n_obs"
        ])

    df = pd.DataFrame(records)

    summary = (
        df.groupby("regime")
        .agg(
            mean_ic=("ic", "mean"),
            std_ic=("ic", "std"),
            pos_ic_ratio=("ic", lambda x: (x > 0).mean()),
            sig_ic_ratio=("p_value", lambda x: (x < 0.05).mean()),
            mean_p_value=("p_value", "mean"),
            num_units=("ic", "count"),
            avg_n_obs=("n_obs", "mean"),
        )
        .reset_index()
    )

    summary["ic_ir"] = summary["mean_ic"] / summary["std_ic"]
    summary = summary.rename(columns={"num_units": unit_name})

    cols = list(summary.columns)
    cols.insert(cols.index("pos_ic_ratio"), cols.pop(cols.index("ic_ir")))
    summary = summary[cols]

    return summary


def run_factor_validation(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    regime_series: pd.Series,
    mode: str = "cross_sectional",
    method: str = "spearman",
    min_obs: int = 5,
    horizon: int = 21,
) -> dict:
    
    fwd_ret = compute_forward_return(prices, horizon)
    fwd_ret = fwd_ret.loc[factor.index]

    panel = build_factor_panel(
        factor=factor,
        fwd_ret=fwd_ret,
        regime_series=regime_series,
    )

    ic_table = compute_ic(
        panel=panel,
        mode=mode,
        method=method,
        min_obs=min_obs,
    )

    ic_summary = summarize_ic(
        ic_table=ic_table,
        mode=mode,
    )

    ic_by_regime = summarize_ic_by_regime(
        panel=panel,
        mode=mode,
        method=method,
        min_obs=horizon,
    )

    return {
        "panel": panel,
        "ic_table": ic_table,
        "ic_summary": ic_summary,
        "ic_by_regime": ic_by_regime,
    }


def run_lookback_robustness(
    prices: pd.DataFrame,
    lookbacks: list[int],
    regime_series: pd.Series,
    horizon: int = 21,
    mode: str = "cross_sectional",
    method: str = "spearman",
    min_obs: int = 5,
) -> dict:
    fwd_ret = compute_forward_return(prices, horizon)
    all_results = {}
    summary_rows = []

    for lb in lookbacks:
        if mode == "cross_sectional":
            factor = cross_sectional_momentum(prices, lb)
        elif mode == "time_series":
            factor = time_series_momentum(prices, cum_months = lb)
        factor = factor.loc[fwd_ret.index]

        result = run_factor_validation(
            factor=factor,
            prices=prices,
            regime_series=regime_series,
            mode=mode,
            method=method,
            min_obs=min_obs,
            horizon=horizon,
        )

        all_results[lb] = result
        row = result["ic_summary"].to_dict()
        row["lookback"] = lb
        summary_rows.append(row)

    robustness_summary = pd.DataFrame(summary_rows).sort_values("lookback").reset_index(drop=True)

    return {
        "lookback_summary": robustness_summary,
        "lookback_results": all_results,
    }