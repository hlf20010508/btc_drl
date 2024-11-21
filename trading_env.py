import numpy as np


class TradingEnv:
    def __init__(self, data, transaction_cost=0.001):
        self.data = data
        self.transaction_cost = transaction_cost
        self.reset()

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.sold_price = 0
        self.assets = 1
        self.assets_last = 1
        self.actions = []

    def step(self, action_prob):
        # action - 0: buy 1: sell 2: hold
        if self.position == 0:
            action_prob[1] = 0
        elif self.position == 1:
            action_prob[0] = 0

        action = np.argmax(action_prob)

        self.actions.append(action)

        reward = 0

        current_price = self.data.iloc[self.current_step]

        if action == 0:
            self.position = 1
            self.entry_price = current_price
            self.assets = (1 - self.transaction_cost) * current_price
            reward = 0

        elif action == 1:
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

        elif action == 2:
            if self.position == 1:
                reward = (current_price - self.entry_price) / self.entry_price * 0.1
            else:
                if self.sold_price > 0:
                    reward = (self.sold_price - current_price) / self.sold_price * 0.1
                else:
                    reward = 0

        self.current_step += 1

        return reward, self.assets, action
