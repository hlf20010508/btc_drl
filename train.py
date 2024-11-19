import torch
from torch.utils.data import DataLoader
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
    learning_rate=0.001,
    gamma=0.99,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dataset = BTCDataset(seq_len, interval, start)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    trading_env = TradingEnv(dataset.data)

    input_dim = dataset.feature_num

    model = GRUPPO(input_dim, hidden_dim, action_dim, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        state = trading_env.reset()
        hidden_state = torch.zeros(num_layers, batch_size, hidden_dim).to(device)

        all_rewards = []
        all_log_probs = []
        all_values = []

        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)

            action_probs, values, hidden_state = model(batch_x, hidden_state)

            actions = torch.multinomial(action_probs, 1).squeeze(1)

            log_probs = torch.log(
                action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            )
            all_log_probs.append(log_probs)
            all_values.append(values.squeeze(1))

            for action in actions:
                next_state, reward, done, profit = trading_env.step(action.item())
                all_rewards.append(reward)
                state = next_state
                if done:
                    break

        all_values = torch.cat(all_values, dim=0).to(device)
        all_log_probs = torch.cat(all_log_probs, dim=0).to(device)

        returns = []
        G = 0
        for r in reversed(all_rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        advantages = returns - all_values.detach()

        policy_loss = -torch.mean(all_log_probs * advantages)
        value_loss = torch.mean((returns - all_values) ** 2)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Total Profit: {profit:.2f}"
        )

    torch.save(model.state_dict, f"output/btcusdt_{interval}_{start}.pth")


if __name__ == "__main__":
    run(epochs=100, batch_size=32, seq_len=24 * 7)
