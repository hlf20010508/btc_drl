from dataset import BTCDataset
import numpy as np
import torch


class TradingEnv:
    def __init__(self, dataset: BTCDataset, transaction_cost=0.001):
        self.dataset = dataset
        self.transaction_cost = transaction_cost
        self.reset()

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.sold_price = 0
        self.assets = 1
        self.assets_last = 1
        self.actions = [2] * self.dataset.seq_len
        self.positions = [0] * self.dataset.seq_len

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_step > len(self.dataset) - 1:
            raise StopIteration
        else:
            data, y = self.dataset[self.current_step]
            one_hot_actions = np.eye(3)[self.actions[-self.dataset.seq_len :]]
            positions = np.array(self.positions).reshape(-1, 1)
            x = np.hstack((data, one_hot_actions, positions))
            return torch.tensor(x, dtype=torch.float32), torch.tensor(
                y, dtype=torch.float32
            )

    def step(self, action):
        self.actions.append(action)
        # action - 0: buy 1: sell 2: hold
        reward = 0

        current_price = self.dataset.data["Close"].iloc[self.current_step]

        if action == 0:
            if self.position == 0:
                # buy only when not in position
                self.position = 1
                self.entry_price = current_price
                self.assets = (1 - self.transaction_cost) * current_price
                reward = 0
            else:
                reward = -10

        elif action == 1:
            if self.position == 1:
                # sell only when in position
                self.assets = (
                    self.assets
                    / self.entry_price
                    * current_price
                    * (1 - self.transaction_cost)
                )
                reward = (self.assets - self.assets_last) / self.assets_last

                self.position = 0
                self.entry_price = 0
                self.sold_price = current_price
            else:
                reward = -10

        elif action == 2:
            if self.position == 1:
                reward = (current_price - self.entry_price) / self.entry_price * 0.1
            else:
                if self.sold_price > 0:
                    reward = (self.sold_price - current_price) / self.sold_price * 0.1
                else:
                    reward = 0

        self.current_step += 1

        self.positions.append(self.position)
        if len(self.positions) > self.dataset.seq_len:
            self.positions.pop(0)

        return reward, self.assets
