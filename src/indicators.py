from __future__ import annotations
from dataclasses import dataclass

import pandas as pd
import numpy as np

from src.metric import rolling_volatility

TRADING_DAYS_M = 21

# cross sectional momentum
def raw_cs_momentum(prices: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    if lookback <= 0:
        raise ValueError("lookback 必須大於 0")
    
    raw_cs_mom = prices.pct_change(lookback * TRADING_DAYS_M)

    return raw_cs_mom


def vol_adj_cs_momentum(prices: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    if lookback <= 0:
        raise ValueError("lookback 必須大於 0")
    
    lb = lookback * TRADING_DAYS_M
    
    raw_cs_mom = prices.pct_change(lb)
    
    ret = prices.pct_change()
    vol = rolling_volatility(returns = ret, window = lb)
    vol_adj_cs_mom = raw_cs_mom / vol

    return vol_adj_cs_mom


# time series momentum
def raw_ts_momentum(prices: pd.DataFrame, cum_months: int = 12, ignore_months: int = 1) -> dict:
    if cum_months < ignore_months:
        raise ValueError("cum_months 必須大於 ignore_months")
    
    raw_ts_mom = prices.pct_change(cum_months * TRADING_DAYS_M) \
                .shift(ignore_months * TRADING_DAYS_M)
    
    return raw_ts_mom


def vol_adj_ts_momentum(prices: pd.DataFrame, cum_months: int = 12, ignore_months: int = 1) -> dict:
    if cum_months < ignore_months:
        raise ValueError("cum_months 必須大於 ignore_months")
    
    lb, ignore = cum_months * TRADING_DAYS_M, ignore_months * TRADING_DAYS_M
    
    raw_ts_mom = prices.pct_change(lb).shift(ignore)

    ret = prices.pct_change()
    vol = rolling_volatility(returns = ret, window = lb)
    vol_adj_ts_mom = raw_ts_mom / vol
    
    return  vol_adj_ts_mom