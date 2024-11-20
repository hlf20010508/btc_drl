class TradingEnv:
    def __init__(self, data, transaction_cost=0.001):
        self.data = data
        self.current_step = 0
        self.transaction_cost = transaction_cost
        self.position = 0
        self.entry_price = 0
        self.assets = 1
        self.assets_last = 1

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.assets = 1
        self.assets_last = 1

    def step(self, action):
        # action - 0: buy 1: hold 2: sell
        done = False
        reward = 0

        current_price = self.data.iloc[self.current_step]

        if action == 0:
            if self.position == 0:
                # buy only when not in position
                self.position = 1
                self.entry_price = current_price
                self.assets = (1 - self.transaction_cost) * current_price
                reward = 0
            else:
                reward = -1

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
            else:
                reward = -1

        elif action == 2:
            if self.position == 1:
                reward = (current_price - self.entry_price) / self.entry_price
            else:
                reward = 0

        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True

        return reward, done, self.assets
