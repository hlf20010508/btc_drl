import torch
from torch.utils.data import DataLoader
from model import GRUPPO
from dataset import BTCDataset
from trading_env import TradingEnv
from draw import draw
import pandas as pd


def run(
    batch_size,
    seq_len,
    interval="1H",
    start="2020_01_01",
    hidden_dim=64,
    action_dim=3,
    num_layers=2,
    dropout=0.2,
    features=[],
    should_draw=False,
    trading_env_class=TradingEnv,
    dataset=None,
    model=None,
    device="cpu",
):
    device = torch.device(device)

    if dataset is None:
        dataset = BTCDataset(seq_len, interval, start, "backtrace", features)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    trading_env = trading_env_class(dataset.data["Close"][dataset.seq_len :])

    input_dim = dataset.feature_num

    if model is None:
        model = GRUPPO(input_dim, hidden_dim, action_dim, num_layers, dropout).to(
            device
        )
        model.load_state_dict(
            torch.load(
                f"output/btcusdt_{interval}_{start}_best_earnings.pth",
                map_location=device,
            )
        )

    model.eval()

    trading_env.reset()
    hidden_state = torch.zeros(num_layers, batch_size, hidden_dim).to(device)

    for batch_x, _ in dataloader:
        batch_x = batch_x.to(device)

        action_probs, action_ratio, values, hidden_state = model(batch_x, hidden_state)

        actions = torch.multinomial(action_probs, 1).squeeze(1)
        action_ratio = action_ratio.squeeze(1)

        for action, ratio in zip(actions, action_ratio):
            _, assets = trading_env.step(action.item(), ratio.item())

    print(f"Backtrace Earnings: {(assets - 1) * 100:.2f}%")

    if should_draw:
        draw(dataset, trading_env.actions, trading_env.assets_all, title="Backtrace")

    result = pd.DataFrame(
        {
            "date": dataset._load_data(mode="backtrace")["Date"].tolist()[
                dataset.seq_len :
            ],
            "assets": trading_env.assets_all,
        }
    )

    result.to_csv("output/backtrace.csv")


if __name__ == "__main__":
    run(
        batch_size=32,
        seq_len=7,
        features=["Open", "High", "Low", "Close", "Volume"],
        should_draw=True,
    )
