from okx_data import download_data
from finrl.config import INDICATORS
from finrl.env_stocktrading import StockTradingEnv
from finrl.models import DRLAgent
from stable_baselines3.common.logger import configure


def run(
    interval="1H", start="2020-01-01", hmax=100, initial_amount=100, reward_scaling=1e-4
):
    train_data, _ = download_data(interval, start)

    stock_dimension = len(train_data.tic.unique())
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

    e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    agent = DRLAgent(env=env_train)
    model = agent.get_model("a2c", model_kwargs={"device": "cpu"})

    RESULTS_DIR = "output"
    tmp_path = RESULTS_DIR + "/a2c"
    new_logger_a2c = configure(tmp_path, ["stdout", "csv"])
    # Set new logger
    model.set_logger(new_logger_a2c)

    trained_a2c = agent.train_model(
        model=model, tb_log_name="a2c", total_timesteps=50000
    )

    trained_a2c.save(RESULTS_DIR + "/a2c/agent")


if __name__ == "__main__":
    run()
