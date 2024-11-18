from okx.app import OkxSPOT
import pandas as pd
from datetime import datetime
from indicator import add_indicator
import os


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
    if not os.path.exists(path):
        btc_data = get_market_data(
            "BTC-USDT",
            interval,
            datetime.strptime(start, datetime_format),
            datetime.strptime("2024_04_30", datetime_format),
        )

        add_indicator(btc_data)
        btc_data.dropna(inplace=True)
        btc_data.to_csv(path, index=False)
        print(f"Saved data to {path}")
    else:
        print("BTCUSDT data already exists")

    path_backtrace = f"data/btcusdt_{interval}_backtrace.csv"
    if not os.path.exists(path_backtrace):
        btc_backtest_data = get_market_data(
            "BTC-USDT",
            interval,
            datetime.strptime("2024_05_01", datetime_format),
            datetime.strptime("2024_11_01", datetime_format),
        )
        btc_backtest_data.to_csv(path_backtrace, index=False)
        print(f"Saved backtrace data to {path}")
    else:
        print("BTCUSDT backtrace data already exists")


if __name__ == "__main__":
    download_data("1H", "2020_01_01")
