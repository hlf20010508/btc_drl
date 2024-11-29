import torch
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import os
from model import GRUPPO
from dataset import BTCDataset
from trading_env import TradingEnv
import backtrace
from draw import draw


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
    features=[],
    should_draw=False,
    should_draw_backtrace=False,
    trading_env_class=TradingEnv,
    dataset=None,
    dataset_backtrace=None,
    model=None,
    device="cpu",
):
    if not os.path.exists("output"):
        os.mkdir("output")

    device = torch.device(device)

    if dataset is None:
        dataset = BTCDataset(seq_len, interval, start, "train", features)
    if dataset_backtrace is None:
        dataset_backtrace = BTCDataset(seq_len, interval, start, "backtrace", features)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    trading_env = trading_env_class(dataset.data["Close"][dataset.seq_len :])

    input_dim = dataset.feature_num

    if model is None:
        model = GRUPPO(input_dim, hidden_dim, action_dim, num_layers, dropout).to(
            device
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    best_loss = None
    best_earnings = None

    for epoch in range(epochs):
        model.train()
        trading_env.reset()
        hidden_state = torch.zeros(num_layers, batch_size, hidden_dim).to(device)

        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)

            action_probs, action_ratio, values, hidden_state = model(
                batch_x, hidden_state
            )

            actions = torch.multinomial(action_probs, 1).squeeze(1)
            action_ratio = action_ratio.squeeze(1)

            rewards = []
            for action, ratio in zip(actions, action_ratio):
                reward, assets = trading_env.step(action.item(), ratio.item())
                rewards.append(reward)

            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32).to(device)

            advantages = returns - values.detach()

            log_probs = torch.log(
                action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            )
            policy_loss = -torch.mean(log_probs * advantages)
            value_loss = torch.mean((returns - values) ** 2)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Earnings: {(assets - 1) * 100:.2f}%"
        )

        torch.save(model.state_dict(), f"output/btcusdt_{interval}_{start}.pth")
        if best_loss is None or loss < best_loss:
            best_loss = loss
            torch.save(
                model.state_dict(), f"output/btcusdt_{interval}_{start}_best_loss.pth"
            )
        if best_earnings is None or assets > best_earnings:
            best_earnings = assets
            torch.save(
                model.state_dict(),
                f"output/btcusdt_{interval}_{start}_best_earnings.pth",
            )

        if should_draw:
            draw(dataset, trading_env.actions, trading_env.assets_all, title="Train")

        backtrace.run(
            batch_size=batch_size,
            seq_len=seq_len,
            interval=interval,
            start=start,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_layers=num_layers,
            dropout=dropout,
            features=features,
            should_draw=should_draw_backtrace,
            trading_env_class=trading_env_class,
            dataset=dataset_backtrace,
            model=model,
            device=device,
        )

    print(
        f"Best Loss: {best_loss.item():.4f}, Best Earnings: {(best_earnings - 1) * 100:.2f}%"
    )


if __name__ == "__main__":
    run(
        epochs=5,
        batch_size=32,
        seq_len=24 * 7,
        # features=["Open", "High", "Low", "Close", "Volume"],
        should_draw=True,
        should_draw_backtrace=True,
    )
