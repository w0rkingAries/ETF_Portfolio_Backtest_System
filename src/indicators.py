from __future__ import annotations
from dataclasses import dataclass

import pandas as pd
import numpy as np


TRADING_DAYS_M = 21

# cross sectional momentum
def cross_sectional_momentum(prices: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    if lookback <= 0:
        raise ValueError("lookback 必須大於 0")
    
    cross_sectional_mom = prices / prices.shift(lookback * TRADING_DAYS_M) - 1.0

    
    return cross_sectional_mom


# time series momentum
def time_series_momentum(prices: pd.DataFrame, cum_months: int = 12, ignore_months: int = 1) -> pd.DataFrame:
    if cum_months < ignore_months:
        raise ValueError("cum_months 必須大於 ignore_months")
    
    time_series_mom = prices.pct_change(cum_months * TRADING_DAYS_M) \
                        .shift(ignore_months * TRADING_DAYS_M)
    
    return time_series_mom