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
        self, seq_len: int, interval: str = "1H", start: str = "2020_01_01", features=[]
    ):
        self.data = self._load_data(interval, start)
        self.scale_info = self._load_scale_info(interval, start)
        self._preprocess_data()
        self.seq_len = seq_len
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

    def _load_data(self, interval: str = "1H", start: str = "2020_01_01"):
        download_data(interval, start)

        return pd.read_csv(f"data/btcusdt_{interval}_{start}.csv")

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
