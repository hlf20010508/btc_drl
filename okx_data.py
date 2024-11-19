from okx.app import OkxSPOT
from features import (
    min_max_unify_features,
    min_max_features,
    linear_features,
    standard_features,
)
import pandas as pd
from datetime import datetime
from indicator import add_indicator
import os
import json


market = OkxSPOT(key="", secret="", passphrase="").market


def get_market_data(symbol, interval, start, end=datetime.now()):
    klines = market.get_history_candle(
        instId=symbol,
        start=start,
        end=end,
        bar=interval,
    )["data"]

    if len(klines) > 0:
        data = pd.DataFrame(klines).iloc[:, :6]

        data.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

        data["Date"] = pd.to_datetime(data["Date"], unit="ms")

        return data


def download_data(interval: str = "1H", start: str = "2020_01_01"):
    if not os.path.exists("data"):
        os.mkdir("data")

    print("Downloading BTCUSDT data...")

    datetime_format = "%Y_%m_%d"

    path = f"data/btcusdt_{interval}_{start}.csv"
    path_backtrace = f"data/btcusdt_{interval}_backtrace.csv"
    path_scale_info = f"data/btcusdt_{interval}_{start}_scale_info.json"
    if (
        not os.path.exists(path)
        or not os.path.exists(path_backtrace)
        or not os.path.exists(path_scale_info)
    ):
        btc_data = get_market_data(
            "BTC-USDT",
            interval,
            datetime.strptime(start, datetime_format),
            datetime.strptime("2024_11_01", datetime_format),
        )

        add_indicator(btc_data)
        btc_data.dropna(inplace=True)

        scale_info = {}

        for feature in min_max_unify_features:
            scale_info[feature] = {
                "min": btc_data["Low"].min(),
                "max": btc_data["High"].max(),
            }

        for feature in min_max_features:
            scale_info[feature] = {
                "min": btc_data[feature].min(),
                "max": btc_data[feature].max(),
            }

        for feature in linear_features:
            scale_info[feature] = 100

        for feature in standard_features:
            scale_info[feature] = {
                "mean": btc_data[feature].mean(),
                "std": btc_data[feature].std(),
            }

        with open(path_scale_info, "w") as f:
            json.dump(scale_info, f)
            print(f"Saved scale info to {path_scale_info}")

        btc_backtest_data = btc_data[
            btc_data["Date"] >= pd.Timestamp("2024/04/30 16:00:00")
        ]
        btc_backtest_data.to_csv(path_backtrace, index=False)
        print(f"Saved backtrace data to {path}")

        btc_data = btc_data[btc_data["Date"] < pd.Timestamp("2024/04/30 16:00:00")]
        btc_data.to_csv(path, index=False)
        print(f"Saved data to {path}")
    else:
        print("BTCUSDT data already exists")


if __name__ == "__main__":
    download_data("1H", "2020_01_01")
