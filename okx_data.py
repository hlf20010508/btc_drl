from okx.app import OkxSPOT
import pandas as pd
from datetime import datetime
import os
from finrl.preprocessors import FeatureEngineer, data_split
from finrl.config import INDICATORS
from var import PATH_TRAIN, PATH_BACKTRACE

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

        data.columns = ["date", "open", "high", "low", "close", "volume"]

        data["date"] = pd.to_datetime(data["date"], unit="ms").dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        data["tic"] = "BTC-USDT"

        return data


def add_indicator(data):
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(data)

    return processed


def download_data(interval: str = "1H", start: str = "2020-01-01"):
    backtrace_date = "2024-05-01"
    end = "2024-11-01"

    if not os.path.exists("data"):
        os.mkdir("data")

    print("Downloading BTC-USDT data...")

    datetime_format = "%Y-%m-%d"

    path_train = PATH_TRAIN % (interval, start)
    path_backtrace = PATH_BACKTRACE % (interval, start)
    if not os.path.exists(path_train) or not os.path.exists(path_backtrace):
        btc_data = get_market_data(
            "BTC-USDT",
            interval,
            datetime.strptime(start, datetime_format),
            datetime.strptime(end, datetime_format),
        )

        btc_data = add_indicator(btc_data)

        train_data = data_split(btc_data, start, backtrace_date)
        backtrace_data = data_split(btc_data, backtrace_date, end)

        train_data.to_csv(path_train)
        print(f"Saved data to {path_train}")
        backtrace_data.to_csv(path_backtrace)
        print(f"Saved data to {path_backtrace}")

        return train_data, backtrace_data
    else:
        print("BTC-USDT data already exists")

        train_data = pd.read_csv(path_train)
        backtrace_data = pd.read_csv(path_backtrace)

        return train_data, backtrace_data


if __name__ == "__main__":
    download_data()
