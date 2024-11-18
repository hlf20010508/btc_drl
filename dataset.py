import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from okx_data import download_data


class BTCDataset(Dataset):
    def __init__(self, seq_len: int, interval: str = "1H", start: str = "2020_01_01"):
        self.data = self._read_data(interval, start)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, index: int):
        x = self.data.iloc[index : index + self.seq_len].drop(columns=["Close"]).values
        y = self.data.iloc[index + self.seq_len]["Close"]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )

    def _read_data(interval: str = "1H", start: str = "2020_01_01"):
        download_data(interval, start)

        data = pd.read_csv(f"data/btcusdt_{interval}_{start}.csv")

        data["Date"] = pd.to_datetime(data["Date"])
        data.set_index("Date", inplace=True)
        data = data.sort_index()

        data.fillna(method="ffill", inplace=True)

        price_highest = data["High"].max()
        price_lowest = data["Low"].min()

        def min_max_unify_scaler(keys: list[str]):
            for key in keys:
                data[key] = (data[key] - price_lowest) / (price_highest - price_lowest)

        min_max_unify_scaler(
            [
                "Open",
                "High",
                "Low",
                "Close",
                "EMA7",
                "EMA14",
                "EMA21",
                "BBANDS-upper",
                "BBANDS-middle",
                "BBANDS-lower",
                "VEGAS-filter",
                "VEGAS-tunnel1-upper",
                "VEGAS-tunnel1-lower",
                "VEGAS-tunnel2-upper",
                "VEGAS-tunnel2-lower",
            ]
        )

        def min_max_scaler(keys: list[str]):
            for key in keys:
                data[key] = (data[key] - data[key].min()) / (
                    data[key].max() - data[key].min()
                )

        min_max_scaler(["Volume", "OBV"])

        def linear_scaler(keys: list[str], rate: float = 100):
            for key in keys:
                data[key] = data[key] / rate

        linear_scaler(["RSI7", "RSI14", "RSI21", "KDJ-K", "KDJ-D", "KDJ-J"])

        def standard_scaler(keys: list[str]):
            for key in keys:
                data[key] = (data[key] - data[key].mean()) / data[key].std()

        standard_scaler(["MACD", "MACD-signal", "MACD-hist"])

        return data

    def to_dataloader(self, batch_size: int = 32, shuffle: bool = False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
