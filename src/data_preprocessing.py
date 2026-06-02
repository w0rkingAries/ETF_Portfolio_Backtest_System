from __future__ import annotations
from pathlib import Path

import pandas as pd
import yfinance as yf

# 選擇的 ETF 列表
DEFAULT_TICKERS = [
    "0050.TW","006208.TW","006203.TW","006204.TW","00850.TW", # 台股大盤
    "0053.TW","00881.TW",  # 科技/半導體產業
    "0056.TW","00701.TW","00713.TW", # 高股息
    "00646.TW","00662.TW","00757.TW","00636.TW","006207.TW", # 美國市場
    "00635U.TW","00642U.TW", # 商品
    "00631L.TW","00632R.TW","00663L.TW", # 槓桿
    "TLT","IEF","SHY", # 美國債券 -- 利率
    "LQD","HYG", # 美國債券 -- 信用
]


# 下載需要的 ETF 資料
def download_etf(
        tickers: list | str, 
        start: str, 
        end: str | None = None,
) -> pd.DataFrame:

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,  
        actions=True,        
        progress=False,
    )
    if(type(tickers)==str):
        tickers = [tickers]

    if isinstance(data.columns, pd.MultiIndex):
        mult_raw = set(data.columns.get_level_values(0))

        if "Close" not in mult_raw:
            raise ValueError("找不到 Close 欄位")
        if "Dividends" not in mult_raw:
            # 某些情況可能完全冇 dividend 欄，補 0
            close = data["Close"].copy()
            div = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        else:
            close = data["Close"].copy()
            div = data["Dividends"].copy()

        # 補齊欄位順序，避免某些 ticker 缺值
        close = close.reindex(columns=tickers)
        div = div.reindex(columns=tickers).fillna(0.0)

    close.index = pd.to_datetime(close.index)
    div.index = pd.to_datetime(div.index)

    close = close.sort_index()
    div = div.sort_index().fillna(0.0)

    result = pd.concat({
            "Close": close,
            "Dividends": div,
        },axis=1,
    ).sort_index()

    return result


# 將dataframe儲存成csv檔案
def save_dataframe(df: pd.DataFrame, output_path: str | Path, filename: str) -> None:
    output_path = (Path(output_path) / filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=True)


# 將csv導入
def load_csv(input_path: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df


# 檢查資料是否存在缺失
def inspect_missing_values(df: pd.DataFrame) -> pd.Series:
    print(df.isna().sum().sort_values(ascending=False))
    print()


# 先將資料按時間排序, 再將缺值的列補上/移除
# 資料會與真實市場有偏差
def clean_prices(
    prices: pd.DataFrame,
    drop_all_na_rows: bool = True,
    forward_fill: bool = True,
) -> pd.DataFrame:
    df = prices.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # 將前一個交易日資料代入, 假設價格沒有變動, 可能會做成低估波動性後果
    if forward_fill:
        df = df.ffill()

    # 若資料被大量拋棄會做成資料失真, 失去代表性
    if drop_all_na_rows:
        df = df.dropna(how="any")

    if df.empty:
        raise ValueError("清理後資料為空")

    return df


# 檢查是否存在異常return
def check_extreme_returns(
    returns: pd.DataFrame,
    threshold: float = 0.2,
) -> pd.DataFrame:
    mask = returns.abs() > threshold
    extreme_rows = returns[mask.any(axis=1)]
    return extreme_rows