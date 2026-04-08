from __future__ import annotations
from pathlib import Path
from datetime import datetime

import pandas as pd

from src.data_preprocessing import(
    download_etf, 
    DEFAULT_TICKERS,
    clean_prices,
    save_dataframe,
)
from src.metric import compute_daily_returns, compute_growth_index
from src.market_regime import( 
    compute_regime, 
    evaluate_multiple_strategies_by_regime,
    evaluate_trade_stats_by_regime,
)
from src.indicators import cross_sectional_momentum, time_series_momentum
from src.factor_validation import run_factor_validation, run_lookback_robustness
from src.reporting import( 
    build_horizontal_metrics,
    build_report_block, 
    build_regime_report,
    export_report
)
from src.portfolio import( 
    equal_weight_portfolio_returns,
    build_cs_mom_weights,
    build_ts_mom_weights,
    compute_portfolio_returns_from_weights,
    compute_portfolio_returns_with_stop,  
)

# ================================================================================
# 設定輸出資料儲存路徑
# ================================================================================
ETF_DATA_PATH = Path("data/ETF")
STRATEGY_DATA_PATH = Path("data/strategy")
REPORT_OUTPUT_PATH = Path("report")

def main(
    etf: str = "0050.TW", 
    mode: str = "print", 
    start: str = "2021-01-01", 
    end: str = "2025-12-31"
):
    # 確保資料輸出模式正確
    if mode not in {"print", "save", "p", "s"}:
        raise ValueError("mode must be 'print', or 'save'")
    
    d1 = datetime.strptime(start, "%Y-%m-%d")
    d2 = datetime.strptime(end, "%Y-%m-%d")
    if d1 > d2:
        raise ValueError("start date must be before the end date")
    
    mapping = {"0050.TW": "SPY", "SPY": "0050.TW"}
    another_base = mapping[etf] # 最後比較指標的另一個 benchmark
    
    # ================================================================================
    # 資料預處理:取得需要的 ETF 資料集, 檢查缺值與資料清洗
    # ================================================================================
    raw_prices = download_etf(tickers = DEFAULT_TICKERS, start = start, end = end,)
    prices = clean_prices(raw_prices, drop_all_na_rows = True, forward_fill = True)

    returns = compute_daily_returns(prices) # 使用收盤價和股息計算 ETF 的每日報酬率
    adj_prices = compute_growth_index(returns) # 通過每日報酬率回推出每日的價格變化

    # ================================================================================
    # 市場條件:使用 趨勢x波動 對市場進行分類
    # ================================================================================
    raw_etf = download_etf(tickers = etf, start = start, end = end,) 
    ETF = clean_prices(raw_etf, drop_all_na_rows = True, forward_fill = True)
    ret_ETF = compute_daily_returns(ETF)
    regime_ETF = compute_regime(ETF)

    regime_setup = build_horizontal_metrics(
        s = regime_ETF["regime"].value_counts(normalize=True),
        title = f"Market Regime -- {etf}",
        mode = mode
    )

    # ================================================================================
    # 因子/訊號: 橫截面動量(根據過去9個月的累積報酬率進行排序, 選擇前8名買入)
    #           時間序列動量(根據過去2個月的累積報酬率決定是否買入)
    # ================================================================================
    cs_mom = cross_sectional_momentum(prices = adj_prices, lookback = 9)
    ts_mom = time_series_momentum(prices = adj_prices, cum_months = 2)
    # 從因子驗證結果觀察到 lb7 的效能很好
    ts_mom_lb7 = time_series_momentum(prices = adj_prices, cum_months = 7) 

    # ================================================================================
    # 因子驗證: 將 橫截面動量訊號 和 時間序列動量訊號 放到市場條件中檢驗因子的效益
    #          並驗證採用的時間區間的穩健性
    # ================================================================================
    # cross sectional momentum
    cs_ic_by_regime = run_factor_validation(
        factor=cs_mom,
        prices=adj_prices,
        regime_series=regime_ETF["regime"],
        mode="cross_sectional",
        horizon=21,
        method="spearman",
        min_obs=8,
    )

    cs_ic_robustness = run_lookback_robustness(
        prices= adj_prices,
        lookbacks= [i for i in range(2,13)],
        regime_series = regime_ETF["regime"],
        horizon = 21,
        mode = "cross_sectional",
        min_obs = 8,
    )

    cs_ic_summary = {
        "Cross_sectional_momentum_ic_summary": cs_ic_by_regime["ic_summary"],
        "Cross_sectional_momentum_ic_by_regime": cs_ic_by_regime["ic_by_regime"],
        "Cross_sectional_momentum_ic_robustness": cs_ic_robustness["lookback_summary"],
    }
    cs_ic_report = build_report_block(
        object = cs_ic_summary, 
        title = "Cross Sectional Momentum Factor Validation", 
        mode = mode
    )

    # time series momentum
    ts_ic_by_regime = run_factor_validation(
        factor=ts_mom,
        prices=adj_prices,
        regime_series=regime_ETF["regime"],
        mode="time_series",
        horizon=21,
        method="spearman",
        min_obs=21,
    )
    ts_ic_by_regime_lb7 = run_factor_validation(
        factor=ts_mom_lb7,
        prices=adj_prices,
        regime_series=regime_ETF["regime"],
        mode="time_series",
        horizon=21,
        method="spearman",
        min_obs=21,
    )

    ts_ic_robustness = run_lookback_robustness(
        prices= adj_prices,
        lookbacks= [i for i in range(2,13)],
        regime_series= regime_ETF["regime"],
        horizon = 21,
        mode = "time_series",
        method = "spearman",
        min_obs = 21,
    )

    ts_ic_summary = {
        "Time_series_momentum_ic_summary_lb2": ts_ic_by_regime["ic_summary"],
        "Time_series_momentum_ic_by_regime_lb2": ts_ic_by_regime["ic_by_regime"],
        "Time_series_momentum_ic_summary_lb7": ts_ic_by_regime_lb7["ic_summary"],
        "Time_series_momentum_ic_by_regime_lb7": ts_ic_by_regime_lb7["ic_by_regime"],
        "Time_series_momentum_ic_robustness": ts_ic_robustness["lookback_summary"],
    }
    ts_ic_report = build_report_block(
        object = ts_ic_summary,
        title = "Time Series Momentum Factor Validation", 
        mode = mode
    )

    # lb2 與 lb7 之間的correaltrion分析, 結果介於0.49-0.53之間, 增加 lb7 作為比較對象
    # ic_series_2 = ts_ic_by_regime["ic_table"]["ic"].dropna()
    # ic_series_7 = ts_ic_by_regime_7["ic_table"]["ic"].dropna()
    # print(ic_series_2.corr(ic_series_7)) 

    # ================================================================================
    # 策略回測報酬率: 每月根據訊號調整 ETF 倉位, 回測計算策略的總報酬率
    # ================================================================================
    # benchmark
    ew_returns = equal_weight_portfolio_returns(returns)
    benchmark_returns = equal_weight_portfolio_returns(ret_ETF)

    # 建立與市場條件相反的 0050/SPY benckmark
    raw_base = download_etf(tickers = another_base, start = start, end = end,) 
    base = clean_prices(raw_base, drop_all_na_rows = True, forward_fill = True)
    ret_base = compute_daily_returns(base)
    another_benchmark_returns = equal_weight_portfolio_returns(ret_base)

    # cross sectional momentum
    cs_mom_wei = build_cs_mom_weights(prices = adj_prices, mom_factor = cs_mom, top_n = 8,)
    cs_mom_returns = compute_portfolio_returns_from_weights(returns = returns, weights = cs_mom_wei)
    cs_mom_asset_stop, cs_mom_stop_wei = compute_portfolio_returns_with_stop(returns = returns, weights = cs_mom_wei,)

    # time series momentum
    ts_mom_wei = build_ts_mom_weights(prices = adj_prices, mom_factor = ts_mom)
    ts_mom_returns = compute_portfolio_returns_from_weights(returns = returns, weights = ts_mom_wei)
    ts_mom_asset_stop, ts_mom_stop_wei = compute_portfolio_returns_with_stop(returns = returns, weights = ts_mom_wei,)

    ts_mom_wei_lb7 = build_ts_mom_weights(prices = adj_prices, mom_factor = ts_mom_lb7)
    ts_mom_returns_lb7 = compute_portfolio_returns_from_weights(returns = returns, weights = ts_mom_wei_lb7)
    ts_mom_asset_stop_lb7, ts_mom_stop_wei_lb7 = compute_portfolio_returns_with_stop(returns = returns, weights = ts_mom_wei_lb7,)

    # ================================================================================
    # 回測結果: 比較不同策略在不同市場條件下的各項指標
    # ================================================================================
    # metrics comparison
    bennckmark1 = f"Benchmark({etf})"
    bennckmark2 = f"Benchmark({another_base})"

    strategy_returns = {
        bennckmark1: benchmark_returns,
        bennckmark2: another_benchmark_returns,
        "Benchmark(equal_weight)": ew_returns,
        "Cross_sectional_momentum": cs_mom_returns,
        "Cross_sectional_momentum_w_asset_stop": cs_mom_asset_stop,
        "Time_series_momentum_lb2": ts_mom_returns,
        "Time_series_momentum_lb2_w_asset_stop":ts_mom_asset_stop,
        "Time_series_momentum_lb7": ts_mom_returns_lb7,
        "Time_series_momentum_lb7_w_asset_stop":ts_mom_asset_stop_lb7,
    }

    metirc_by_regime = evaluate_multiple_strategies_by_regime(
        strategy_returns=strategy_returns,
        regime_series=regime_ETF["regime"],
    )
    metric_report = build_regime_report(
        object = metirc_by_regime,
        title = "Metrics Comparison",
        mode = mode,
    )

    # trading statistic comparison
    strategy_trade = {
        "Cross_sectional_momentum": [cs_mom_returns, cs_mom_wei],
        "Cross_sectional_momentum_w_asset_stop": [cs_mom_asset_stop, cs_mom_stop_wei],
        "Time_series_momentum_lb2": [ts_mom_returns, ts_mom_wei],
        "Time_series_momentum_lb2_w_asset_stop": [ts_mom_asset_stop, ts_mom_stop_wei],
        "Time_series_momentum_lb7": [ts_mom_returns_lb7, ts_mom_wei_lb7],
        "Time_series_momentum_lb7_w_asset_stop": [ts_mom_asset_stop_lb7, ts_mom_stop_wei_lb7],
    }

    trade_stats_by_regime = evaluate_trade_stats_by_regime(
        combined = strategy_trade,
        regime_series = regime_ETF["regime"],
    )
    trade_stats_report = build_regime_report(
        object = trade_stats_by_regime,
        title = "Trading Statistic Comparison",
        mode = mode,
    )

    # 儲存模式, 把檔案存到指定位置
    if mode == "save" or mode == "s":
        # ETF 每日的報酬率與價格波動
        save_dataframe(returns, output_path = ETF_DATA_PATH, filename = "ETF_returns.csv")
        save_dataframe(adj_prices, output_path = ETF_DATA_PATH, filename = "ETF_adjusted_prices.csv")

        # benchmark 每日的報酬率
        df_benchmark = pd.concat([ETF, ret_ETF, base, ret_base], axis = 1)
        df_benchmark.columns = [
            f"{etf}_close", f"{etf}_dividends", f"{etf}_returns",
            f"{another_base}_close", f"{another_base}_dividends", f"{another_base}_returns"
        ]
        save_dataframe(df_benchmark, output_path = ETF_DATA_PATH, filename = "Benchmark_returns.csv")
        
        # 策略在市場條件下的權重與報酬率變化
        save_dataframe(cs_mom_wei, output_path = STRATEGY_DATA_PATH, filename = f"{etf}/cs_mom_wei.csv")
        save_dataframe(cs_mom_stop_wei, output_path = STRATEGY_DATA_PATH, filename = f"{etf}/cs_mom_stop_wei.csv")
        save_dataframe(ts_mom_wei, output_path = STRATEGY_DATA_PATH, filename = f"{etf}/ts_mom_wei_lb2.csv")
        save_dataframe(ts_mom_stop_wei, output_path = STRATEGY_DATA_PATH, filename = f"{etf}/ts_mom_stop_wei_lb2.csv")
        save_dataframe(ts_mom_wei_lb7, output_path = STRATEGY_DATA_PATH, filename = f"{etf}/ts_mom_wei_lb7.csv")
        save_dataframe(ts_mom_stop_wei_lb7, output_path = STRATEGY_DATA_PATH, filename = f"{etf}/ts_mom_stop_wei_lb7.csv")

        df_strategy_ret = pd.concat([
            cs_mom_returns, cs_mom_asset_stop,
            ts_mom_returns, ts_mom_asset_stop,
            ts_mom_returns_lb7, ts_mom_asset_stop_lb7
        ], axis = 1)
        df_strategy_ret.columns = [
            "cs_momentum_ret","cs_momentum_asset_stop_ret",
            "ts_momentum_lb2_ret","ts_momentum_lb2_asset_stop_ret",
            "ts_momentum_lb7_ret","ts_momentum_lb7_asset_stop_ret",
        ]
        save_dataframe(df_strategy_ret, output_path = STRATEGY_DATA_PATH, filename = f"{etf}/all_mom_strategy_returns.csv")

        export_report(
            blocks = [regime_setup, cs_ic_report, ts_ic_report, metric_report, trade_stats_report],
            title = f"{etf}_base_ETF_research.txt",
            mode = mode,
            save_path = REPORT_OUTPUT_PATH,
            filename = f"{etf}_base_ETF_research.txt",
        )


if __name__ == '__main__':
    for etf in ["0050.TW", "SPY"]:
        main(etf, "s")