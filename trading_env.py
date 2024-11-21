import numpy as np


class TradingEnv:
    def __init__(self, data, transaction_cost=0.001):
        self.data = data
        self.transaction_cost = transaction_cost
        self.trade_window = []
        self.trade_window_size = 24
        self.max_trades_in_window = 5
        self.reset()

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.sold_price = 0
        self.assets = 1
        self.actions = []

    def step(self, action_prob):
        # action - 0: buy 1: sell 2: hold
        action_real = np.argmax(action_prob)

        if self.position == 0:
            action_prob[1] = 0
        elif self.position == 1:
            action_prob[0] = 0

        action = np.argmax(action_prob)

        self.actions.append(action)

        reward = 0

        if action_real != action:
            reward = -0.1

        current_price = self.data.iloc[self.current_step]

        price_diff = current_price - self.entry_price

        if action == 0:
            self.position = 1
            self.entry_price = current_price
            self.assets = (1 - self.transaction_cost) * self.assets

            self.trade_window.append(self.current_step)
            if len(self.trade_window) > self.max_trades_in_window:
                if self.current_step - self.trade_window[0] > self.trade_window_size:
                    reward -= 2
                    self.trade_window.pop(0)

        elif action == 1:
            self.assets = (
                self.assets
                / self.entry_price
                * current_price
                * (1 - self.transaction_cost)
            )

            if price_diff > 0:
                reward += 1 + price_diff / self.entry_price
            else:
                reward += -1 + price_diff / self.entry_price

            self.position = 0
            self.entry_price = 0
            self.sold_price = current_price

            self.trade_window.append(self.current_step)
            if len(self.trade_window) > self.max_trades_in_window:
                if self.current_step - self.trade_window[0] > self.trade_window_size:
                    reward -= 2
                    self.trade_window.pop(0)

        elif action == 2:
            if self.position == 1:
                self.assets = (
                    self.assets
                    / self.entry_price
                    * current_price
                    * (1 - self.transaction_cost)
                )
                reward += price_diff / self.entry_price
            else:
                if self.sold_price > 0:
                    reward += price_diff / self.sold_price
                else:
                    reward -= 0.1

        # print(action, reward, self.assets)

        self.current_step += 1

        return reward, self.assets, action
