import pandas as pd
import torch
import json
from torch.utils.data import Dataset, DataLoader
from okx_data import download_data
from features import (
    min_max_unify_features,
    min_max_features,
    linear_features,
    standard_features,
)


class BTCDataset(Dataset):
    def __init__(
        self,
        seq_len: int,
        interval: str = "1H",
        start: str = "2020_01_01",
        mode="train",  # train, backtrace
        features=[],
    ):
        self.seq_len = seq_len
        self.data = self._load_data(interval, start, mode)
        self.scale_info = self._load_scale_info(interval, start)
        self._preprocess_data()
        if features:
            self.data = self.data.loc[:, features]

    @property
    def feature_num(self):
        return self.data.shape[1]

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, index: int):
        x = self.data.iloc[index : index + self.seq_len].values
        y = self.data.iloc[index + self.seq_len]["Close"]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )

    def _load_data(self, interval: str = "1H", start: str = "2020_01_01", mode="train"):
        download_data(interval, start)

        data = pd.read_csv(f"data/btcusdt_{interval}_{start}.csv")
        data["Date"] = pd.to_datetime(data["Date"])

        backtrace_start_time = "2024/04/30 16:00:00"
        if mode == "train":
            return data[data["Date"] < pd.Timestamp(backtrace_start_time)]
        elif mode == "backtrace":
            backtrace_idx = data[
                data["Date"] >= pd.Timestamp(backtrace_start_time)
            ].index[0]
            start_idx = max(0, backtrace_idx - self.seq_len)
            return data.iloc[start_idx:]
        else:
            raise ValueError("BTCDataset mode must be train or backtrace")

    def _load_scale_info(self, interval: str = "1H", start: str = "2020_01_01"):
        with open(f"data/btcusdt_{interval}_{start}_scale_info.json", "r") as f:
            return json.load(f)

    def _preprocess_data(self):
        self.data.drop(columns=["Date"], inplace=True)

        for feature in min_max_unify_features + min_max_features:
            self.data[feature] = (
                self.data[feature] - self.scale_info[feature]["min"]
            ) / (self.scale_info[feature]["max"] - self.scale_info[feature]["min"])

        for feature in linear_features:
            self.data[feature] = self.data[feature] / self.scale_info[feature]

        for feature in standard_features:
            self.data[feature] = (
                self.data[feature] - self.scale_info[feature]["mean"]
            ) / self.scale_info[feature]["std"]
