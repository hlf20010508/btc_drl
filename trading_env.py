import numpy as np


class TradingEnv:
    def __init__(self, data, transaction_cost=0.001):
        self.data = data
        self.transaction_cost = transaction_cost
        self.assets_window_size = 4000
        self.trading_window_size = 24
        self.trading_frequency = 5
        self.min_transaction_usdt = 0.01
        self.reset()

    def reset(self):
        self.actions = []
        self.current_step = 0
        self.btc = 0
        self.usdt = 1
        self.assets_window = [self.usdt]
        self.assets_all = []
        self.assets_initial = self.usdt

    @property
    def assets(self):
        current_price = self.data.iloc[self.current_step]

        return self.usdt + self.btc * current_price

    def step(self, action, ratio):
        # action - 0: buy 1: sell 2: hold
        current_price = self.data.iloc[self.current_step]

        if action == 0:
            if self._check_frequency():
                bought = self.usdt * ratio * (1 - self.transaction_cost)

                if bought >= self.min_transaction_usdt:
                    self.actions.append(action)

                    self.usdt = self.usdt * (1 - ratio)
                    self.btc += bought / current_price
                    self._move_assets_window()

                    reward = 0
                else:
                    self.actions.append(-1)

                    reward = -1
            else:
                self.actions.append(-1)
                reward = -1

        elif action == 1:
            if self._check_frequency():
                if self.btc > 0:
                    sold = self.btc * ratio * (1 - self.transaction_cost)
                    if sold >= self.min_transaction_usdt:
                        self.actions.append(action)

                        self.btc = self.btc * (1 - ratio)
                        self.usdt += sold * current_price
                        self._move_assets_window()

                        reward = self._calc_reward()
                    else:
                        self.actions.append(-1)

                        reward = -1
                else:
                    self.actions.append(-1)
                    self._move_assets_window()

                    reward = -1
            else:
                self.actions.append(-1)

                reward = -1

        elif action == 2:
            self.actions.append(action)

            self._move_assets_window()

            reward = 0

        assets = self.assets
        self.assets_all.append(assets)

        self.current_step += 1

        return reward, assets

    def _move_assets_window(self):
        self.assets_window.append(self.assets)
        self.assets_window = self.assets_window[-self.assets_window_size :]
        self.assets_initial = self.assets_window[0]

    def _check_frequency(self):
        trading_window = self.actions[-self.trading_window_size :]
        if trading_window.count(0) + trading_window.count(1) < self.trading_frequency:
            return True
        else:
            return False

    def _calc_reward(self):
        current_price = self.data.iloc[self.current_step]
        start_price = self.data.iloc[self.current_step - self.assets_window_size]

        total_profit_rate = (self.assets - self.assets_initial) / self.assets_initial
        market_profit_rate = (current_price - start_price) / start_price

        # sortino ratio
        diff = total_profit_rate - market_profit_rate

        reward = diff / self.downside_deviation()

        return reward

    def downside_deviation(self, target_return=0):
        assets_window = np.array(self.assets_window)

        downside_diff = np.maximum(
            0,
            target_return
            - (assets_window[1:] - assets_window[:-1]) / assets_window[:-1],
        )

        downside_dev = np.sqrt(np.mean(downside_diff**2))

        return downside_dev
