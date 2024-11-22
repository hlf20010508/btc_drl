import torch
from torch.utils.data import DataLoader
from model import GRUPPO
from dataset import BTCDataset
from trading_env import TradingEnv
from draw import draw


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
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

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

        action_probs, values, hidden_state = model(batch_x, hidden_state)

        actions = torch.multinomial(action_probs, 1).squeeze(1)

        for action in actions:
            _, assets = trading_env.step(action.item())

    print(f"Backtrace Earnings: {(assets - 1) * 100:.2f}%")

    if should_draw:
        draw(dataset, trading_env.actions, title="Backtrace")


if __name__ == "__main__":
    run(
        batch_size=32,
        seq_len=7,
        features=["Open", "High", "Low", "Close", "Volume"],
        should_draw=True,
    )
