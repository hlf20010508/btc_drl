class TradingEnv:
    def __init__(self, data, transaction_cost=0.001):
        self.data = data
        self.current_step = 0
        self.transaction_cost = transaction_cost
        self.position = 0  # 0: long, 1: short
        self.entry_price = 0
        self.profit = 0

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.profit = 0

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
                reward = 0
            else:
                reward = -1

        elif action == 1:
            if self.position == 1:
                # sell only when in position
                gross_profit = current_price - self.entry_price
                net_profit = gross_profit - self.transaction_cost * current_price
                reward = net_profit
                self.profit += reward

                self.position = 0
                self.entry_price = 0
            else:
                reward = -1

        elif action == 2:
            if self.position == 1:
                reward = (current_price - self.entry_price) * 0.1
            else:
                reward = 0

        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True

        return reward, done, self.profit
