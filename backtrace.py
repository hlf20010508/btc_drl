from finrl.env_stocktrading import StockTradingEnv
from finrl.config import INDICATORS
from finrl.models import DRLAgent
from okx_data import download_data
from stable_baselines3 import A2C
import matplotlib.pyplot as plt


def run(
    interval="1H", start="2020-01-01", hmax=100, initial_amount=100, reward_scaling=1e-4
):
    _, backtrace_data = download_data(interval, start)

    stock_dimension = len(backtrace_data.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": hmax,
        "initial_amount": initial_amount,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": reward_scaling,
    }

    e_trade_gym = StockTradingEnv(df=backtrace_data, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    trained_a2c = A2C.load("output/a2c/agent", device="cpu")

    df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
        model=trained_a2c, environment=e_trade_gym
    )

    df_result_a2c = df_account_value_a2c.set_index(df_account_value_a2c.columns[0])

    plt.figure()
    plt.plot(df_result_a2c)
    print(df_result_a2c)
    plt.show()


if __name__ == "__main__":
    run()
