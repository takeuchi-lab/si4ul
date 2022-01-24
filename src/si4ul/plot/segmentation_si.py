import numpy as np
import matplotlib.pyplot as plt


def histogram_region(local_white, display_statistics):
    _, ax = plt.subplots()
    X_0 = []
    X_1 = []

    for i, row in enumerate(local_white.image_segmentation):
        for j, value in enumerate(row):
            if value < 128:
                X_0.append(local_white.image[i, j])
            else:
                X_1.append(local_white.image[i, j])
    X_0_mean = np.mean(X_0)
    X_1_mean = np.mean(X_1)

    bins = np.linspace(0,256,16)
    plt.hist(X_0, bins=bins, alpha=0.5, label="black region", color="blue")
    plt.hist(X_1, bins=bins, alpha=0.5, label="white region", color="orange")
    if display_statistics:
        y_min, y_max = ax.get_ylim()
        y_middle = (y_min + y_max) / 2
        plt.vlines(X_0_mean, y_min, y_max, color="blue")
        plt.vlines(X_1_mean, y_min, y_max, color="orange")
        plt.annotate("test statistics", (X_0_mean, y_middle), (X_1_mean, y_middle), arrowprops=dict(arrowstyle="<->"))
    plt.legend()

    plt.xlabel("image color value", fontsize=20)#15
    plt.ylabel("frequency", fontsize=20)#15
    plt.show()


def histogram(local_white):
    bins = np.linspace(0,256,16)
    plt.hist(local_white.image.flatten(), bins=bins, label="all pixel")
    plt.legend()

    plt.xlabel("image color value", fontsize=20)#15
    plt.ylabel("frequency", fontsize=20)#15
    plt.show()
