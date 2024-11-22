import matplotlib.pyplot as plt


def draw(dataset, actions):
    actions_slice = actions
    data_slice = dataset.data["Close"][dataset.seq_len :].values
    buy_points = [
        (i, data_slice[i]) for i in range(len(actions_slice)) if actions_slice[i] == 0
    ]
    sell_points = [
        (i, data_slice[i]) for i in range(len(actions_slice)) if actions_slice[i] == 1
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
