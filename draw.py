import matplotlib.pyplot as plt


def draw(dataset, actions, assets, title=""):
    data = dataset.data["Close"][dataset.seq_len :].values
    buy_points = [(i, data[i]) for i in range(len(actions)) if actions[i] == 0]
    sell_points = [(i, data[i]) for i in range(len(actions)) if actions[i] == 1]

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.title(f"{title} actions")
    plt.plot(data, label="price", zorder=1)

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

    plt.subplot(2, 1, 2)
    plt.title(f"{title} assets")
    plt.plot(assets, label="assets")

    plt.show()
