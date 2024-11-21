import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from model import GRUPPO
from dataset import BTCDataset
from trading_env import TradingEnv


def run(
    epochs,
    batch_size,
    seq_len,
    interval="1H",
    start="2020_01_01",
    hidden_dim=64,
    action_dim=3,
    num_layers=2,
    dropout=0.2,
    learning_rate=0.001,
    gamma=0.99,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dataset = BTCDataset(
        seq_len, interval, start, features=["Open", "High", "Low", "Close", "Volume"]
    )
    trading_env = TradingEnv(dataset)

    input_dim = dataset.feature_num

    model = GRUPPO(input_dim, hidden_dim, action_dim, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        trading_env.reset()
        hidden_state = torch.zeros(num_layers, batch_size, hidden_dim).to(device)

        for x, _ in trading_env:
            x = x.unsqueeze(0).to(device)

            action_probs, value, hidden_state = model(x, hidden_state)

            action = torch.multinomial(action_probs, 1).item()

            reward, assets = trading_env.step(action)

            returns = torch.tensor([reward], dtype=torch.float32).to(device)

            advantage = returns - value.detach()

            log_prob = torch.log(action_probs.squeeze(0)[action])
            policy_loss = -log_prob * advantage
            value_loss = (returns - value) ** 2
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Earnings: {(assets - 1) * 100:.2f}%"
        )

        actions_slice = trading_env.actions
        data_slice = dataset.data["Close"].values
        buy_points = [
            (i, data_slice[i])
            for i in range(len(actions_slice))
            if actions_slice[i] == 0
        ]
        sell_points = [
            (i, data_slice[i])
            for i in range(len(actions_slice))
            if actions_slice[i] == 1
        ]

        plt.figure()
        plt.plot(data_slice, label="price", zorder=1)
        if buy_points:
            buy_x, buy_y = zip(*buy_points)
            plt.scatter(
                buy_x, buy_y, color="green", label="Buy", marker="^", s=100, zorder=2
            )
        if sell_points:
            sell_x, sell_y = zip(*sell_points)
            plt.scatter(
                sell_x, sell_y, color="red", label="Sell", marker="v", s=100, zorder=2
            )
        plt.legend()
        plt.show()

    if not os.path.exists("output"):
        os.mkdir("output")
    torch.save(model.state_dict, f"output/btcusdt_{interval}_{start}.pth")


if __name__ == "__main__":
    run(epochs=10, batch_size=128, seq_len=24)
