import numpy as np
from tqdm import tqdm

from si4ul.si import changepoint_si as si
from si4ul.plot import changepoint_si as plot


def optseg_si_oc(x, sigma, beta_param):
    n = len(x)
    beta = beta_param * np.log(n)
    segment_index, list_condition_matrix, number_of_segments, sg_results = si.dp_si(x, sigma, beta)
    p_value_list = []

    if number_of_segments < 2:
        print("No Changing Point")
        return sg_results, p_value_list

    for k_index in tqdm(range(number_of_segments - 1)):
        first_segment_index = k_index + 1
        second_segment_index = first_segment_index + 1

        selective_p = si.inference_oc(x, segment_index, list_condition_matrix, sigma, first_segment_index, second_segment_index)
        p_value = float(selective_p)
        p_value_list.append(p_value)

    return sg_results[1:-1], p_value_list


def optseg_si(x, sigma, beta_param):
    n = len(x)
    cov = np.identity(len(x)) * sigma ** 2
    beta = beta_param * np.log(n)
    x_flip = np.flip(x)
    sg_results, cost = si.pelt(x, n, beta)
    p_value_list = []

    if len(sg_results) < 3:
        print("No Changing Point")
        return sg_results, p_value_list

    sg_results_len = len(sg_results)
    sg_results_flip, cost_flip = si.pelt(x_flip, n, beta)
    for i in tqdm(range(1, sg_results_len - 1)):
        if sg_results[i] > (n / 2):
            x_prime, eta_vec, eta_T_x = si.parametrize_x(x, n, sg_results[i - 1], sg_results[i], sg_results[i + 1], cov)
            opt_funct_set, opt_funct = si.pelt_perturbed_x(x_prime, sg_results, n, beta)

            p_value = si.inference(opt_funct_set, opt_funct, eta_vec, eta_T_x, cov, n)
            p_value_list.append(p_value)
        else:
            x_prime_flip, eta_vec, eta_T_x = si.parametrize_x(x_flip, n, sg_results_flip[sg_results_len - 1 - i - 1], sg_results_flip[sg_results_len - i - 1], sg_results_flip[sg_results_len - 1 - i + 1], cov)
            eta_vec = - eta_vec
            eta_T_x = - eta_T_x
            opt_funct_set, opt_funct = si.pelt_perturbed_x(x_prime_flip, sg_results_flip, n, beta)

            p_value = si.inference(opt_funct_set, opt_funct, eta_vec, eta_T_x, cov, n)
            p_value_list.append(p_value)

    return sg_results[1:-1], p_value_list


def plot_changepoint_detection(x, mean_vector, segment_size, alpha, sg_results, p_value_list, title):
    n = len(x)
    rejected_sg_results = [0]
    for i in range(len(sg_results)):
        p_value = p_value_list[i]
        if p_value < (alpha / len(sg_results)):
            rejected_sg_results.append(sg_results[i])
    rejected_sg_results.append(n)
    plot.plot(x, rejected_sg_results, mean_vector, n, segment_size, title)