from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import pandas as pd


def histogram(obs_model, comparison_clusters, hist_gene, is_plot_norm):
    c_0, c_1 = comparison_clusters
    X_0 = obs_model.X[obs_model.labels_ == c_0, hist_gene]
    X_1 = obs_model.X[obs_model.labels_ == c_1, hist_gene]

    bins = np.linspace(-3,3,50)
    plt.hist(X_0, bins=bins, alpha=0.5, density=True, label="c_" + str(c_0+1))
    plt.hist(X_1, bins=bins, alpha=0.5, density=True, label="c_" + str(c_1+1))
    plt.legend()

    if is_plot_norm:
        X_target = np.concatenate([X_0, X_1])
        x = np.arange(min(X_target), max(X_target), 0.01)
        y = norm.pdf(x, loc=0, scale=1)
        plt.plot(x,y)

    plt.xlabel("gene" + str(hist_gene+1), fontsize=20)#15
    plt.ylabel("frequency", fontsize=20)#15
    plt.show()


def scatter(obs_model, comparison_clusters, show_dims):
    c_0, c_1 = comparison_clusters
    dim_0, dim_1 = show_dims
    _, ax = plt.subplots(figsize=(4,4))
    sns.set()

    for j in comparison_clusters:
        plt.scatter(obs_model.X[obs_model.labels_ == j, dim_0], obs_model.X[obs_model.labels_ == j, dim_1])
    plt.xlabel("gene" + str(dim_0+1), fontsize=20)#15
    plt.ylabel("gene" + str(dim_1+1), fontsize=20)#15

    X_0 = obs_model.X[obs_model.labels_ == c_0, :]
    X_1 = obs_model.X[obs_model.labels_ == c_1, :]
    X_target = np.concatenate([X_0, X_1])
    ax.set_xlim(min(X_target[:, dim_0])*1.1, max(X_target[:, dim_0])*1.1)
    ax.set_ylim(min(X_target[:, dim_1])*1.1, max(X_target[:, dim_1])*1.1)
    plt.show()


def violin(obs_model, gene_n):
    plt_data = [obs_model.X[:,gene_n].tolist(), (obs_model.labels_+1).tolist()]
    df = pd.DataFrame(plt_data, index=['expression', 'Cluster Index']).T
    df['Cluster Index'] = pd.Series(df['Cluster Index'], dtype='int') #float->int

    fig, _ = plt.subplots(figsize=(2.5, 5))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle("gene%d"%(gene_n+1), fontsize=15, x=0.61)

    plt.gca().get_yaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    v = sns.violinplot(x=df['Cluster Index'], y=df['expression'], cut=0)
    v.set_xlabel('Cluster Index', fontsize=15)
    v.set_ylabel('expression', fontsize=15)
    plt.show()


def p_matrix(matrix, digit, alpha):
    sns.heatmap(matrix, cmap='Reds_r', vmin=0.0, vmax=1.0)
    plt.title("p_values")
    for y, col in enumerate(matrix):
        for x, p in enumerate(col):
            if x <= y:
                continue
            if p <= alpha:
                text = round(p, digit)
                fontsize = "large"
                color = "white"
            else:
                fontsize = "medium"
                text = round(p, digit)
                color = "black"
            plt.annotate(text, xy=(x+0.5, y+0.5), horizontalalignment='center', verticalalignment='center', color=color, fontsize=fontsize)

    plt.show()


def statistics_matrix(matrix, digit):
    sns.heatmap(matrix, cmap='Reds', vmin=0.0)
    plt.title("test statistics")
    for y, col in enumerate(matrix):
        for x, stat in enumerate(col):
            if x <= y:
                continue
            color = "black"
            plt.annotate(round(stat, digit), xy=(x+0.5, y+0.5), horizontalalignment='center', verticalalignment='center', color=color, fontsize="large")

    plt.show()
