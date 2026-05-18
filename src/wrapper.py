from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Callable

import pandas as pd

from src.metric import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
)
from src.factor_validation import (
    run_factor_validation,
    run_lookback_robustness,
    run_top_n_robustness,
)
from src.portfolio import (
    build_cs_mom_weights,
    compute_portfolio_returns_from_weights,
    compute_portfolio_returns_with_stop,
)


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_mode: str


def _validate_dates(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    d1 = pd.to_datetime(start)
    d2 = pd.to_datetime(end)
    if d1 > d2:
        raise ValueError("start date must be before the end date")
    return d1, d2


def _get_extended_start(start: str, lookback_buffer_years: int = 1) -> str:
    return (pd.to_datetime(start) - pd.DateOffset(years=lookback_buffer_years)).strftime("%Y-%m-%d")


def _slice_by_date(obj: pd.DataFrame | pd.Series, start: str | pd.Timestamp, end: str | pd.Timestamp):
    return obj.loc[pd.to_datetime(start): pd.to_datetime(end)]


def _compute_factor(
    prices: pd.DataFrame,
    factor_func: Callable[..., pd.DataFrame],
    lookback: int,
    factor_param_name: str = "lookback",
) -> pd.DataFrame:
    return factor_func(prices=prices, **{factor_param_name: lookback})


def _build_weights(
    portfolio_builder: Callable[..., pd.DataFrame],
    prices: pd.DataFrame,
    mom_factor: pd.DataFrame,
    lookback: int,
    freq: str,
    weight: str,
    top_n: int | None = None,
    max_weights: float | None = None,
    renormalize: bool = True,
) -> pd.DataFrame:
    params = signature(portfolio_builder).parameters
    kwargs = {
        "prices": prices,
        "mom_factor": mom_factor,
        "lb": lookback,
        "freq": freq,
        "weighting": weight,
        "max_weight": max_weights,
        "renormalize_after_limit": renormalize,
    }
    if "top_n" in params:
        kwargs["top_n"] = top_n

    return portfolio_builder(**kwargs)


def make_walk_forward_folds(
    dates: pd.DatetimeIndex,
    start: str,
    end: str,
    train_months: int = 24,
    test_months: int = 12,
    step_months: int = 12,
    train_mode: str = "rolling",
) -> list[WalkForwardFold]:

    if train_months <= 0:
        raise ValueError("train_months must be positive")
    if test_months <= 0:
        raise ValueError("test_months must be positive")
    if step_months <= 0:
        raise ValueError("step_months must be positive")

    mode_alias = {
        "rolling": "rolling",
        "non_reuse": "rolling",
        "expanding": "expanding",
        "reuse": "expanding",
    }
    if train_mode not in mode_alias:
        raise ValueError("train_mode must be one of: rolling, non_reuse, expanding, reuse")

    normalized_mode = mode_alias[train_mode]

    dates = pd.DatetimeIndex(dates).sort_values().unique()
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    eval_dates = dates[(dates >= start_ts) & (dates <= end_ts)]
    if eval_dates.empty:
        return []

    folds: list[WalkForwardFold] = []
    fold_id = 1

    base_start = start_ts
    rolling_train_start = base_start

    while True:
        if normalized_mode == "expanding":
            train_start = base_start
        else:
            train_start = rolling_train_start

        test_start_calendar = rolling_train_start + pd.DateOffset(months=train_months)
        train_end_calendar = test_start_calendar - pd.Timedelta(days=1)
        test_end_calendar = (
            test_start_calendar
            + pd.DateOffset(months=test_months)
            - pd.Timedelta(days=1)
        )

        if test_start_calendar > end_ts:
            break

        if test_end_calendar > end_ts:
            test_end_calendar = end_ts

        train_dates = eval_dates[
            (eval_dates >= train_start)
            & (eval_dates <= train_end_calendar)
        ]
        test_dates = eval_dates[
            (eval_dates >= test_start_calendar)
            & (eval_dates <= test_end_calendar)
        ]

        if not train_dates.empty and not test_dates.empty:
            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train_start=train_dates[0],
                    train_end=train_dates[-1],
                    test_start=test_dates[0],
                    test_end=test_dates[-1],
                    train_mode=normalized_mode,
                )
            )
            fold_id += 1

        rolling_train_start = rolling_train_start + pd.DateOffset(months=step_months)

    return folds


# 將單一 factor 的 IC、IC by regime、lookback robustness、top-n robustness 包成一組
def _run_factor_validation_pack(
    adj_prices: pd.DataFrame,
    regime_series: pd.Series,
    factor_func: Callable[..., pd.DataFrame],
    factor_name: str,
    lookback: int,
    lookbacks: list[int],
    top_n_range: list[int],
    horizon: int,
    min_obs: int,
    mode: str = "cross_sectional",
    factor_param_name: str = "lookback",
) -> dict[str, pd.DataFrame | pd.Series]:
    
    factor = _compute_factor(
        prices=adj_prices,
        factor_func=factor_func,
        lookback=lookback,
        factor_param_name=factor_param_name,
    )

    ic_result = run_factor_validation(
        factor=factor,
        prices=adj_prices,
        regime_series=regime_series,
        mode=mode,
        horizon=horizon,
        method="spearman",
        min_obs=min_obs,
    )

    lb_robustness = run_lookback_robustness(
        prices=adj_prices,
        lookbacks=lookbacks,
        regime_series=regime_series,
        factor_func=factor_func,
        horizon=horizon,
        mode=mode,
        min_obs=min_obs,
    )

    top_n_robustness = run_top_n_robustness(
        prices=adj_prices,
        lookbacks=lookback,
        regime_series=regime_series,
        factor_func=factor_func,
        horizon=horizon,
        mode=mode,
        min_obs=top_n_range,
    )

    return {
        f"{factor_name}_ic_summary": ic_result["ic_summary"],
        f"{factor_name}_ic_by_regime": ic_result["ic_by_regime"],
        f"{factor_name}_ic_robustness_lb": lb_robustness["lookback_summary"],
        f"{factor_name}_ic_robustness_top_n": top_n_robustness["top_n_summary"],
    }


def _build_strategy_returns(
    adj_prices: pd.DataFrame,
    returns: pd.DataFrame,
    mom_factor: pd.DataFrame,
    lookback: int,
    top_n: int | None,
    eval_start: str | pd.Timestamp,
    eval_end: str | pd.Timestamp,
    configs: list,
    weight: str = "equal",
    max_weights: float | None = None,
    renormalize: bool = True,
    portfolio_builder: Callable[..., pd.DataFrame] = build_cs_mom_weights,
) -> tuple[
    dict[str, pd.Series],
    dict[str, list[pd.Series | pd.DataFrame]],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:

    strategy_returns: dict[str, pd.Series] = {}
    strategy_trade: dict[str, list[pd.Series | pd.DataFrame]] = {}
    strategy_weights: dict[str, pd.DataFrame] = {}
    strategy_cost_detail: dict[str, pd.DataFrame] = {}

    for strategy_name, freq in configs:
        weights = _build_weights(
            portfolio_builder=portfolio_builder,
            prices=adj_prices,
            mom_factor=mom_factor,
            lookback=lookback,
            top_n=top_n,
            freq=freq,
            weight=weight,
            max_weights=max_weights,
            renormalize=renormalize,
        )
        weights = _slice_by_date(weights, eval_start, eval_end)
        ret_slice = _slice_by_date(returns, eval_start, eval_end)

        cost_detail = compute_portfolio_returns_from_weights(returns=ret_slice, weights=weights)
        stop_cost_detail, stop_weights = compute_portfolio_returns_with_stop(returns=ret_slice, weights=weights)

        strategy_returns[strategy_name] = cost_detail["net_return"]
        strategy_returns[f"{strategy_name}_w_asset_stop"] = stop_cost_detail["net_return"]

        strategy_trade[strategy_name] = [cost_detail["net_return"], weights]
        strategy_trade[f"{strategy_name}_w_asset_stop"] = [stop_cost_detail["net_return"], stop_weights]

        strategy_weights[strategy_name] = weights
        strategy_weights[f"{strategy_name}_w_asset_stop"] = stop_weights

        strategy_cost_detail[strategy_name] = cost_detail
        strategy_cost_detail[f"{strategy_name}_w_asset_stop"] = stop_cost_detail

    return strategy_returns, strategy_trade, strategy_weights, strategy_cost_detail


def run_walk_forward_validation(
    adj_prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime_series: pd.Series,
    factor_func: Callable[..., pd.DataFrame],
    lookback: int,
    top_n: int | None,
    start: str,
    end: str,
    configs: list,
    train_months: int = 24,
    test_months: int = 12,
    step_months: int = 12,
    train_mode: str = "rolling",
    horizon: int = 21,
    min_obs: int = 13,
    factor_param_name: str = "lookback",
    validation_mode: str = "cross_sectional",
    portfolio_builder: Callable[..., pd.DataFrame] = build_cs_mom_weights,
    primary_strategy_name: str | None = None,
) -> dict[str, pd.DataFrame]:

    mode_alias = {
        "rolling": "rolling",
        "non_reuse": "rolling",
        "expanding": "expanding",
        "reuse": "expanding",
    }
    if train_mode not in mode_alias:
        raise ValueError("train_mode must be one of: rolling, non_reuse, expanding, reuse")

    normalized_mode = mode_alias[train_mode]

    common_dates = adj_prices.index.intersection(returns.index).intersection(regime_series.index)
    common_dates = pd.DatetimeIndex(common_dates).sort_values()

    folds = make_walk_forward_folds(
        dates=common_dates,
        start=start,
        end=end,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
        train_mode=normalized_mode,
    )

    fold_rows: list[dict[str, object]] = []
    fold_returns: dict[str, pd.Series] = {}

    full_factor = _compute_factor(
        prices=adj_prices,
        factor_func=factor_func,
        lookback=lookback,
        factor_param_name=factor_param_name,
    )

    for fold in folds:
        train_prices = _slice_by_date(adj_prices, fold.train_start, fold.train_end)
        train_regime = _slice_by_date(regime_series, fold.train_start, fold.train_end)

        train_factor = _compute_factor(
            prices=train_prices,
            factor_func=factor_func,
            lookback=lookback,
            factor_param_name=factor_param_name,
        )

        train_ic_result = run_factor_validation(
            factor=train_factor,
            prices=train_prices,
            regime_series=train_regime,
            mode=validation_mode,
            horizon=horizon,
            method="spearman",
            min_obs=min_obs,
        )

        train_ic = train_ic_result["ic_summary"]

        test_strategy_returns, _, _, _ = _build_strategy_returns(
            adj_prices=adj_prices,
            returns=returns,
            mom_factor=full_factor,
            lookback=lookback,
            top_n=top_n,
            eval_start=fold.test_start,
            eval_end=fold.test_end,
            configs=configs,
            portfolio_builder=portfolio_builder,
        )

        fold_key_prefix = f"{normalized_mode}_fold_{fold.fold_id}"

        for strategy_name, ret in test_strategy_returns.items():
            ret = ret.dropna()
            fold_returns[f"{fold_key_prefix}_{strategy_name}"] = ret

        main_strategy = primary_strategy_name or configs[0][0]
        main_ret = test_strategy_returns[main_strategy].dropna()

        fold_rows.append(
            {
                #"wf_mode": normalized_mode,
                "fold_id": fold.fold_id,
                "train_start": fold.train_start.strftime("%Y-%m-%d"),
                "train_end": fold.train_end.strftime("%Y-%m-%d"),
                "test_start": fold.test_start.strftime("%Y-%m-%d"),
                "test_end": fold.test_end.strftime("%Y-%m-%d"),
                # "train_mths": train_months,
                # "test_mths": test_months,
                # "step_mths": step_months,
                # "lookback": lookback,
                # "top_n": top_n,
                "train_mean_ic": train_ic.get("mean_ic", pd.NA) if hasattr(train_ic, "get") else pd.NA,
                "train_ic_ir": train_ic.get("ic_ir", pd.NA) if hasattr(train_ic, "get") else pd.NA,
                "train_pos_ic_ratio": train_ic.get("pos_ic_ratio", pd.NA) if hasattr(train_ic, "get") else pd.NA,
                "test_annu_return": annualized_return(main_ret),
                "test_annu_vol": annualized_volatility(main_ret),
                # "test_avg_daily_return": main_ret.mean() if not main_ret.empty else pd.NA,
                # "test_daily_vol": main_ret.std() if not main_ret.empty else pd.NA,
                "test_sharpe": sharpe_ratio(main_ret),
                # "test_num_days": int(main_ret.shape[0]),
            }
        )

    return {
        "walk_forward_summary": pd.DataFrame(fold_rows),
        "walk_forward_returns": pd.DataFrame(fold_returns),
    }
