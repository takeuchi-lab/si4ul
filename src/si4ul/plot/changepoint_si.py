import matplotlib.pyplot as plt
import numpy as np


def plot(x, sg_results, mean_vector, n, segment_size, label):
    plt.plot(x, 'o', color='grey', fillstyle='none', markersize=6, markeredgewidth='0.4')

    # Estimated CP
    parametric_x = []

    for element in sg_results:
        if (element == 0) or (element == n):
            parametric_x.append(element)
        else:
            parametric_x.append(element + 0.5)
            parametric_x.append(element + 0.5)

    parametric_y = []
    for element in calculate_mean_for_each_segment(x, sg_results):
        parametric_y.append(element)
        parametric_y.append(element)

    plt.plot(parametric_x, parametric_y, color='red', linewidth='2', label=label)
    
    # True CP
    if mean_vector is not None:
        true_x = [0]
        curr_idx = 0

        while curr_idx < (n - 20):
            curr_idx = curr_idx + segment_size
            true_x.append(curr_idx + 0.5)
            true_x.append(curr_idx + 0.5)

        true_x.append(n)

        vector = []
        for each_mean in mean_vector:
            vector.append(each_mean)
            vector.append(each_mean)

        vector = np.array(vector)

        true_y = np.append(vector, vector)
        plt.plot(true_x, true_y, color='blue', linewidth='2', label='True Signal')

    plt.legend(loc='lower left')
    plt.show()


def calculate_mean_for_each_segment(x, sg_results):
    list_mean = []

    for i in range(len(sg_results) - 1):
        no_elements = 0
        sum = 0
        for j in range(sg_results[i], sg_results[i + 1]):
            no_elements = no_elements + 1
            sum = sum + x[j]

        list_mean.append(sum / no_elements)

    return list_mean
