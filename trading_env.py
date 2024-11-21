class TradingEnv:
    def __init__(self, data, transaction_cost=0.001):
        self.data = data
        self.transaction_cost = transaction_cost
        self.reset()

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.assets = 1
        self.assets_initial = self.assets
        self.assets_last = self.assets
        self.actions = []

    def step(self, action):
        # action - 0: buy 1: sell 2: hold
        reward = 0

        current_price = self.data.iloc[self.current_step]

        if action == 0:
            if self.position == 0:
                # buy only when not in position
                self.actions.append(action)

                self.position = 1
                self.entry_price = current_price
                self.assets = (1 - self.transaction_cost) * self.assets
                reward = 0
            else:
                self.actions.append(-1)
                reward = -1

        elif action == 1:
            if self.position == 1:
                # sell only when in position
                self.actions.append(action)

                self._update_assets()

                total_profit_rate = (
                    self.assets - self.assets_initial
                ) / self.assets_initial
                this_profit_rate = (self.assets - self.assets_last) / self.assets_last
                reward = total_profit_rate + this_profit_rate

                self.position = 0
                self.entry_price = 0
                self.assets_last = self.assets
            else:
                self.actions.append(-1)
                reward = -1

        elif action == 2:
            self.actions.append(action)

            if self.position == 1:
                reward = (current_price - self.entry_price) / self.entry_price
            else:
                reward = 0

        self.current_step += 1

        return reward, self.assets

    def _update_assets(self):
        current_price = self.data.iloc[self.current_step]

        self.assets = (
            self.assets / self.entry_price * current_price * (1 - self.transaction_cost)
        )
