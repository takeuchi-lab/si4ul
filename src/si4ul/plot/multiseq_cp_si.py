import copy
import matplotlib.pyplot as plt
import numpy as np


def plot(X, true_cp=None, true_mean=None):
    diff_list = [0]
    X_plot = copy.deepcopy(X)
    for d in range(1, X_plot.shape[0]):
        minv = min(X_plot[d])
        maxv = max(X_plot[d - 1])
        diff = maxv - minv + 0.3
        X_plot[d] += diff
        diff_list.append(diff)

    ylim = [
        np.min(X_plot) - (np.max(X_plot) - np.min(X_plot)) * 0.05,
        np.max(X_plot) + (np.max(X_plot) - np.min(X_plot)) * 0.05,
    ]

    plt.ylim(ylim)
    plt.yticks([])

    cmap = plt.get_cmap("tab10")
    for i in range(len(X)):
        plt.plot(X_plot[i], color=cmap(i), linewidth=3)

    if (true_cp is not None) and (true_mean is not None):
        for i in range(len(X)):
            values = np.array(true_mean[i])
            values = values + diff_list[i]
            for s in range(len(values)):
                plt.hlines(
                    values[s], true_cp[i][s] - 1, true_cp[i][s + 1] - 1, color=cmap(i)
                )
                if s > 0:
                    plt.vlines(
                        true_cp[i][s] - 1,
                        min(values[s], values[s - 1]),
                        max(values[s], values[s - 1]),
                        color=cmap(i),
                    )

    plt.show()


def plot_si(X, result, width=0, true_cp=None, true_mean=None):
    diff_list = [0]
    X_plot = copy.deepcopy(X)
    for d in range(1, X_plot.shape[0]):
        minv = min(X_plot[d])
        maxv = max(X_plot[d - 1])
        diff = maxv - minv + 0.3
        X_plot[d] += diff
        diff_list.append(diff)

    ylim = [
        np.min(X_plot) - (np.max(X_plot) - np.min(X_plot)) * 0.05,
        np.max(X_plot) + (np.max(X_plot) - np.min(X_plot)) * 0.05,
    ]

    plt.ylim(ylim)
    plt.yticks([])

    t_list = np.unique(np.asarray(result)[:, 0])
    for t in t_list:
        t = int(t)
        plt.vlines(
            t - 1,
            ylim[1],
            ylim[0],
            linestyle="dashed",
            linewidth=2.5,
        )
        if not width == 0:
            plt.axvspan(t - 1 - width, t - 1 + width, color="red", alpha=0.1)

    cmap = plt.get_cmap("tab10")
    for i in range(len(X)):
        plt.plot(X_plot[i], color=cmap(i), linewidth=3)

    if (true_cp is not None) and (true_mean is not None):
        for i in range(len(X)):
            values = np.array(true_mean[i])
            values = values + diff_list[i]
            for s in range(len(values)):
                plt.hlines(
                    values[s], true_cp[i][s] - 1, true_cp[i][s + 1] - 1, color=cmap(i)
                )
                if s > 0:
                    plt.vlines(
                        true_cp[i][s] - 1,
                        min(values[s], values[s - 1]),
                        max(values[s], values[s - 1]),
                        color=cmap(i),
                    )

    for res in result:
        t = int(res[0])
        dim = int(res[1])
        if res[2]:
            c = "r"
        else:
            c = "b"
        plt.plot([t - 1], [X_plot[dim, t - 1]], marker="o", markersize=15, color=c)
    plt.show()
