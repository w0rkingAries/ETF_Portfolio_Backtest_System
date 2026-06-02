from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_preprocessing import(
    download_etf,
    DEFAULT_TICKERS,
    clean_prices,
    save_dataframe,
)
from src.metric import( 
    compute_daily_returns, 
    compute_growth_index,
)
from src.market_regime import(
    compute_regime,
    evaluate_multiple_strategies_by_regime,
    evaluate_trade_stats_by_regime,
    analyze_multiple_strategies_regime_overlays,
    analyze_multiple_strategies_trade_stats_regime_overlays,
)
from src.indicators import raw_cs_momentum
from src.reporting import(
    build_horizontal_metrics,
    build_report_block,
    build_regime_report,
    build_overlay_regime_report,
    build_risk_cost_report,
    export_report,
)
from src.portfolio import equal_weight_portfolio_returns
from src.risk import compute_multiple_risk_cost_summary
from src.wrapper import (
    _build_strategy_returns,
    _get_extended_start,
    _run_factor_validation_pack,
    _slice_by_date,
    _validate_dates,
    run_rolling_oos_validation,
)

# ================================================================================
# 設定輸出資料儲存路徑
# ================================================================================
ETF_DATA_PATH = Path("data/ETF")
STRATEGY_DATA_PATH = Path("data/strategy/cross_sectional_momentum")
REPORT_OUTPUT_PATH = Path("report/cross_sectional_momentum")


def main(
    etf: str = "0050.TW",
    mode: str = "s",
    start: str = "2022-01-01",
    end: str = "2025-12-31",
    lookback: int = 7,
    top_n: int = 13,
    lookback_buffer_years: int = 1,
    run_factor: bool = True,
    run_rolling_oos: bool = True,
    wf_train_months: int = 18,
    wf_test_months: int = 6,
    wf_step_months: int = 6,
    wf_modes: str = "rolling",
):
    # 確保資料輸出模式正確
    if mode not in {"print", "save", "p", "s"}:
        raise ValueError("mode must be 'print', or 'save'")

    _validate_dates(start, end)

    mapping = {"0050.TW": "SPY", "SPY": "0050.TW"}
    another_base = mapping[etf]  # 最後比較指標的另一個 benchmark

    data_start = _get_extended_start(start, lookback_buffer_years=lookback_buffer_years)

    # ================================================================================
    # 資料預處理：下載區間往前多抓一年，但正式績效只評估 start ~ end
    # ================================================================================
    raw_prices = download_etf(tickers=DEFAULT_TICKERS, start=data_start, end=end)
    prices_ext = clean_prices(raw_prices, drop_all_na_rows=True, forward_fill=True)

    returns_ext = compute_daily_returns(prices_ext)
    adj_prices_ext = compute_growth_index(returns_ext)

    returns = _slice_by_date(returns_ext, start, end)
    adj_prices = _slice_by_date(adj_prices_ext, start, end)

    # ================================================================================
    # 市場條件：同樣用 extended data 計算 regime，避免 MA / rolling vol 初期失真
    # ================================================================================
    raw_etf = download_etf(tickers=etf, start=data_start, end=end)
    ETF_ext = clean_prices(raw_etf, drop_all_na_rows=True, forward_fill=True)
    ret_ETF_ext = compute_daily_returns(ETF_ext)
    regime_ETF_ext = compute_regime(ETF_ext, prices=adj_prices_ext)

    ETF = _slice_by_date(ETF_ext, start, end)
    ret_ETF = _slice_by_date(ret_ETF_ext, start, end)
    regime_ETF = _slice_by_date(regime_ETF_ext, start, end)

    regime_setup = build_horizontal_metrics(
        s=[
            regime_ETF["regime"].value_counts(normalize=True),
            regime_ETF["regime_x_breadth"].value_counts(normalize=True),
            regime_ETF["regime_x_corr"].value_counts(normalize=True),
        ],
        title=f"Market Regime -- {etf}",
        mode=mode,
        per_row=4,
    )

    # ================================================================================
    # 因子/訊號：用 extended prices 建 factor，再切回正式回測區間
    # ================================================================================
    raw_cs_mom_ext = raw_cs_momentum(prices=adj_prices_ext, lookback=lookback)

    # ================================================================================
    # 因子驗證：IC / regime IC / lookback robustness / top-n robustness
    # ================================================================================
    cs_ic_report = None
    cs_ic_summary = {}

    if run_factor:
        raw_pack = _run_factor_validation_pack(
            adj_prices=adj_prices,
            regime_series=regime_ETF["regime"],
            factor_func=raw_cs_momentum,
            factor_name="Raw_cs_momentum",
            lookback=lookback,
            lookbacks=[i for i in range(2, 13)],
            top_n_range=[i for i in range(9, 19)],
            horizon=21,
            min_obs=13,
        )
        cs_ic_summary = raw_pack

        cs_ic_report = build_report_block(
            object=cs_ic_summary,
            title="Cross Sectional Momentum Factor Validation",
            mode=mode,
        )

    # ================================================================================
    # 策略回測報酬率：用 extended factor 建權重，再切回正式區間計績效
    # ================================================================================
    benchmark_returns = equal_weight_portfolio_returns(ret_ETF)

    raw_base = download_etf(tickers=another_base, start=data_start, end=end)
    base_ext = clean_prices(raw_base, drop_all_na_rows=True, forward_fill=True)
    ret_base_ext = compute_daily_returns(base_ext)
    base = _slice_by_date(base_ext, start, end)
    ret_base = _slice_by_date(ret_base_ext, start, end)
    another_benchmark_returns = equal_weight_portfolio_returns(ret_base)

    raw_configs = [
        ("Cross_sectional_momentum_1M", "M"),
        ("Cross_sectional_momentum_2M", "2M"),
        ("Cross_sectional_momentum_3M", "3M"),
    ]
    raw_returns, raw_trades, raw_weights, raw_cost_details = _build_strategy_returns(
        adj_prices = adj_prices_ext,
        returns = returns_ext,
        mom_factor = raw_cs_mom_ext,
        lookback = lookback,
        top_n = top_n,
        eval_start = start,
        eval_end = end,
        configs=raw_configs,
    )
    
    vol_adj_configs = [
        ("Vol_adj_cross_sectional_momentum_1M", "M"),
        ("Vol_adj_cross_sectional_momentum_2M", "2M"),
        ("Vol_adj_cross_sectional_momentum_3M", "3M"),
    ]
    vol_adj_returns, vol_adj_trades, vol_adj_weights, vol_adj_cost_details = _build_strategy_returns(
        adj_prices = adj_prices_ext,
        returns = returns_ext,
        mom_factor = raw_cs_mom_ext,
        lookback = lookback,
        top_n = top_n,
        eval_start = start,
        eval_end = end,
        weight = "vol_adj",  
        configs = vol_adj_configs,
    )

    wei_adj_configs = [
        ("Wei_adj_cross_sectional_momentum_1M", "M"),
        ("Wei_adj_cross_sectional_momentum_2M", "2M"),
        ("Wei_adj_cross_sectional_momentum_3M", "3M"),
    ]
    wei_adj_returns, wei_adj_trades, wei_adj_weights, wei_adj_cost_details = _build_strategy_returns(
        adj_prices = adj_prices_ext,
        returns = returns_ext,
        mom_factor = raw_cs_mom_ext,
        lookback = lookback,
        top_n = top_n,
        eval_start = start,
        eval_end = end,
        weight = "vol_adj",
        max_weights = 0.2,
        configs = wei_adj_configs,
    )

    strategy_returns = raw_returns | vol_adj_returns | wei_adj_returns
    strategy_trades = raw_trades | vol_adj_trades | wei_adj_trades
    strategy_weights = raw_weights | vol_adj_weights | wei_adj_weights
    strategy_cost_details = raw_cost_details | vol_adj_cost_details | wei_adj_cost_details

    # 把 benchmark 加進整體比較
    bennckmark1 = f"Benchmark({etf})"
    bennckmark2 = f"Benchmark({another_base})"
    strategy_returns[bennckmark1] = benchmark_returns
    strategy_returns[bennckmark2] = another_benchmark_returns

    # ================================================================================
    # Walk-forward / cross validation：每個 fold 訓練驗證，再做 OOS test
    # ================================================================================
    wf_report = None
    wf_result = {}

    if run_rolling_oos:
        wf_summary_parts: list[pd.DataFrame] = []
        wf_return_parts: list[pd.DataFrame] = []

        raw_configs = [("Cross_sectional_momentum_1M", "M"),]
        #for wf_mode in wf_modes:
        mode_result = run_rolling_oos_validation(
                adj_prices=adj_prices_ext,
                returns=returns_ext,
                benchmark_returns=benchmark_returns,
                regime_series=regime_ETF_ext["regime"],
                factor_func=raw_cs_momentum,
                lookback=lookback,
                top_n=top_n,
                start=start,
                end=end,
                configs=raw_configs,
                train_months=wf_train_months,
                test_months=wf_test_months,
                step_months=wf_step_months,
                train_mode=wf_modes,
                horizon=21,
                min_obs=13,
        )
        wf_summary_parts.append(mode_result["Rolling_OOS_summary"])

        mode_returns = mode_result["Rolling_OOS_returns"].copy()
        mode_returns.columns = [f"{wf_modes}_{col}" for col in mode_returns.columns]
        wf_return_parts.append(mode_returns)
        #

        wf_result = {
            "Rolling_OOS_summary": pd.concat(wf_summary_parts, axis=0, ignore_index=True)
            if wf_summary_parts else pd.DataFrame(),
            "Rolling_OOS_returns": pd.concat(wf_return_parts, axis=1)
            if wf_return_parts else pd.DataFrame(),
        }
        wf_result["Rolling_OOS_summary"].pop("fold_id")

        wf_report = build_report_block(
            object={"Rolling_OOS_summary": wf_result["Rolling_OOS_summary"]},
            title=f"Rolling OOS Validation, train/test/step: {wf_train_months}/{wf_test_months}/{wf_step_months} months",
            mode=mode,
        )

    # ================================================================================
    # 回測結果：比較不同策略在不同市場條件下的各項指標
    # ================================================================================
    metirc_by_regime = evaluate_multiple_strategies_by_regime(
        strategy_returns=strategy_returns,
        regime_series=regime_ETF["regime"],
    )
    metric_report = build_regime_report(
        object=metirc_by_regime,
        title="Metrics Comparison",
        mode=mode,
    )

    metirc_by_multi_regime = analyze_multiple_strategies_regime_overlays(
        strategy_returns=strategy_returns,
        regime_df=regime_ETF,
    )
    multi_regime_report = build_overlay_regime_report(
        object=metirc_by_multi_regime,
        title="Metrics Comparison by Multi-Regime",
        mode=mode,
    )

    trade_stats_by_regime = evaluate_trade_stats_by_regime(
        combined=strategy_trades,
        regime_series=regime_ETF["regime"],
    )
    trade_stats_report = build_regime_report(
        object=trade_stats_by_regime,
        title="Trading Statistic Comparison",
        mode=mode,
    )

    trade_stats_by_multi_regime = analyze_multiple_strategies_trade_stats_regime_overlays(
        combined=strategy_trades,
        regime_df=regime_ETF,
    )
    trade_stats_multi_regime_report = build_overlay_regime_report(
        object=trade_stats_by_multi_regime,
        title="Trading Statistic Comparison by Multi-Regime",
        mode=mode,
    )

    risk_summary = compute_multiple_risk_cost_summary(strategy_cost_details)
    risk_report = build_risk_cost_report(
        summary=risk_summary,
        mode=mode,
    )

    # ================================================================================
    # 儲存模式
    # ================================================================================
    if mode in {"save", "s"}:
        save_dataframe(returns, output_path=ETF_DATA_PATH, filename="ETF_returns.csv")
        save_dataframe(adj_prices, output_path=ETF_DATA_PATH, filename="ETF_adjusted_prices.csv")

        df_benchmark = pd.concat([ETF, ret_ETF, base, ret_base], axis=1)
        df_benchmark.columns = [
            f"{etf}_close", f"{etf}_dividends", f"{etf}_returns",
            f"{another_base}_close", f"{another_base}_dividends", f"{another_base}_returns",
        ]
        save_dataframe(df_benchmark, output_path=ETF_DATA_PATH, filename="Benchmark_returns.csv")

        for name, weights in strategy_weights.items():
            save_dataframe(weights, output_path=STRATEGY_DATA_PATH, filename=f"{etf}/{name}_weights.csv")

        df_strategy_ret = pd.concat(strategy_returns, axis=1)
        save_dataframe(df_strategy_ret, output_path=STRATEGY_DATA_PATH, filename=f"{etf}/strategy_returns.csv")

        if run_rolling_oos and wf_result:
            save_dataframe(
                wf_result["Rolling_OOS_summary"],
                output_path=REPORT_OUTPUT_PATH,
                filename=f"{etf}_Rolling_OOS_summary.csv",
            )
            save_dataframe(
                wf_result["Rolling_OOS_returns"],
                output_path=STRATEGY_DATA_PATH,
                filename=f"{etf}/Rolling_OOS_returns.csv",
            )

        save_dataframe(metirc_by_regime, output_path=REPORT_OUTPUT_PATH, filename=f"{etf}_metrics_by_regime.csv")
        save_dataframe(metirc_by_multi_regime, output_path=REPORT_OUTPUT_PATH, filename=f"{etf}_metrics_by_multi_regime.csv")
        save_dataframe(trade_stats_by_regime, output_path=REPORT_OUTPUT_PATH, filename=f"{etf}_trade_stats_by_regime.csv")
        save_dataframe(trade_stats_by_multi_regime, output_path=REPORT_OUTPUT_PATH, filename=f"{etf}_trade_stats_by_multi_regime.csv")
        save_dataframe(risk_summary, output_path=REPORT_OUTPUT_PATH, filename=f"{etf}_risk_cost_summary.csv")

        regime_blocks = [regime_setup]
        if cs_ic_report is not None:
            regime_blocks.append(cs_ic_report)
        if wf_report is not None:
            regime_blocks.append(wf_report)
        
        metirc_blocks = [metric_report, multi_regime_report]
        trade_stat_blocks = [trade_stats_report, trade_stats_multi_regime_report]
        risk_blocks = [risk_report]

        outputs = {
            "Regime_Setup": regime_blocks,
            "Metrics_by_Regime": metirc_blocks, 
            "Trade_Stats_by_Regime": trade_stat_blocks, 
            "Risk_and_Cost": risk_blocks,
        }
        for names, values in outputs.items():
            export_report(
                blocks=values,
                title=f"Cross_Sectional_{names}_Summary.txt",
                mode=mode,
                save_path=REPORT_OUTPUT_PATH,
                filename=f"Cross_Sectional_{names}_Summary.txt",
            )

    # return {
    #     "regime": regime_ETF,
    #     "factor_validation": cs_ic_summary,
    #     "Rolling_OOS": wf_result,
    #     "strategy_returns": strategy_returns,
    #     "trade_stats_by_regime": trade_stats_by_regime,
    #     "metrics_by_regime": metirc_by_regime,
    # }


if __name__ == "__main__":
    # for etf in ["0050.TW", "SPY"]:
    #     main(etf=etf, mode="s")
    main()
