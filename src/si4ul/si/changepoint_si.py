import numpy as np
from mpmath import mp
import ast

ERR_THRESHOLD = 1e-10
mp.dps = 500


def check_zero(value):
    if - ERR_THRESHOLD <= value <= ERR_THRESHOLD:
        return 0

    return value


def check_coef_zero(f_tau_t):
    if - ERR_THRESHOLD <= f_tau_t[0] <= ERR_THRESHOLD:
        f_tau_t[0] = 0

    if - ERR_THRESHOLD <= f_tau_t[1] <= ERR_THRESHOLD:
        f_tau_t[1] = 0

    if - ERR_THRESHOLD <= f_tau_t[2] <= ERR_THRESHOLD:
        f_tau_t[2] = 0

    return f_tau_t


def intersect(range_1, range_2):
    lower = max(range_1[0], range_2[0])
    upper = min(range_1[1], range_2[1])

    if upper < lower:
        return []
    else:
        return [lower, upper]


def intersect_range(prime_range, tilde_range, range_tau_greater_than_zero, list_2_range):
    initial_range = intersect(prime_range, tilde_range)
    initial_range = intersect(initial_range, range_tau_greater_than_zero)

    if len(initial_range) == 0:
        return []

    final_list = [initial_range]

    for each_2_range in list_2_range:

        lower_range = [np.NINF, each_2_range[0]]
        upper_range = [each_2_range[1], np.Inf]

        new_final_list = []

        for each_1_range in final_list:
            local_range_1 = intersect(each_1_range, lower_range)
            local_range_2 = intersect(each_1_range, upper_range)

            if len(local_range_1) > 0:
                new_final_list.append(local_range_1)

            if len(local_range_2) > 0:
                new_final_list.append(local_range_2)

        final_list = new_final_list

    return final_list


def inference_oc(data, cluster_index, global_list_conditioning_matrix,
              sigma, first_cluster_index, second_cluster_index):

    x = np.array(data)
    n = len(x)
    x = x.reshape((x.shape[0], 1))

    vector_1_C_a = np.zeros(n)
    vector_1_C_b = np.zeros(n)

    n_a = 0
    n_b = 0

    for i in range(n):
        if cluster_index[i] == second_cluster_index:
            n_a = n_a + 1
            vector_1_C_a[i] = 1.0

        elif cluster_index[i] == first_cluster_index:
            n_b = n_b + 1
            vector_1_C_b[i] = 1.0

    vector_1_C_a = np.reshape(vector_1_C_a, (vector_1_C_a.shape[0], 1))
    vector_1_C_b = np.reshape(vector_1_C_b, (vector_1_C_b.shape[0], 1))

    first_element = np.dot(vector_1_C_a.T, x)[0][0]
    second_element = np.dot(vector_1_C_b.T, x)[0][0]

    tau = first_element / n_a - second_element / n_b

    if tau < 0:
        temp = vector_1_C_a
        vector_1_C_a = vector_1_C_b
        vector_1_C_b = temp

        temp = n_a
        n_a = n_b
        n_b = temp

    first_element = np.dot(vector_1_C_a.T, x)[0][0]
    second_element = np.dot(vector_1_C_b.T, x)[0][0]

    tau = first_element / n_a - second_element / n_b

    big_sigma = sigma * sigma * np.identity(n)

    eta_a_b = vector_1_C_a / n_a - vector_1_C_b / n_b

    c_vector = np.dot(big_sigma, eta_a_b) / ((np.dot(eta_a_b.T, np.dot(big_sigma, eta_a_b)))[0][0])

    z = np.dot((np.identity(n) - np.dot(c_vector, eta_a_b.T)), x)

    L_prime = np.NINF
    U_prime = np.Inf

    L_tilde = np.NINF
    U_tilde = np.Inf

    list_2_range = []
    range_tau_greater_than_zero = [0, np.Inf]

    for each_element in global_list_conditioning_matrix:
        matrix = each_element[0]
        constant = each_element[1]

        c_T_A_c = np.dot(c_vector.T, np.dot(matrix, c_vector))[0][0]
        z_T_A_c = np.dot(z.T, np.dot(matrix, c_vector))[0][0]
        c_T_A_z = np.dot(c_vector.T, np.dot(matrix, z))[0][0]
        z_T_A_z = np.dot(z.T, np.dot(matrix, z))[0][0]

        a = c_T_A_c
        b = z_T_A_c + c_T_A_z
        c = z_T_A_z + constant

        if -ERR_THRESHOLD <= a <= ERR_THRESHOLD:
            a = 0

        if -ERR_THRESHOLD <= b <= ERR_THRESHOLD:
            b = 0

        if -ERR_THRESHOLD <= c <= ERR_THRESHOLD:
            c = 0

        if a == 0:
            if b == 0:
                if c > 0:
                    print('z_T_A_z > 0')
            elif b < 0:
                temporal_lower_bound = -c / b
                L_prime = max(L_prime, temporal_lower_bound)

                if L_prime > tau:
                    print('L_prime > tau')
            elif b > 0:
                temporal_upper_bound = -c / b
                U_prime = min(U_prime, temporal_upper_bound)

                if U_prime < tau:
                    print('U_prime < tau')
        else:
            delta = b ** 2 - 4 * a * c

            if -ERR_THRESHOLD <= delta <= ERR_THRESHOLD:
                delta = 0

            if delta == 0:
                if a > 0:
                    print('c_T_A_c > 0 and delta = 0')
            elif delta < 0:
                if a > 0:
                    print('c_T_A_c > 0 and delta < 0')
            elif delta > 0:
                if a > 0:
                    x_lower = (-b - np.sqrt(delta)) / (2 * a)
                    x_upper = (-b + np.sqrt(delta)) / (2 * a)

                    if x_lower > x_upper:
                        print('x_lower > x_upper')

                    L_tilde = max(L_tilde, x_lower)
                    U_tilde = min(U_tilde, x_upper)

                else:
                    x_1 = (-b - np.sqrt(delta)) / (2 * a)
                    x_2 = (-b + np.sqrt(delta)) / (2 * a)

                    x_low = min(x_1, x_2)
                    x_up = max(x_1, x_2)
                    list_2_range.append([x_low, x_up])

    final_list_range = intersect_range([L_prime, U_prime], [L_tilde, U_tilde], range_tau_greater_than_zero, list_2_range)

    numerator = 0
    denominator = 0

    tn_sigma = np.sqrt(np.dot(eta_a_b.T, np.dot(big_sigma, eta_a_b))[0][0])

    for each_final_range in final_list_range:
        al = each_final_range[0]
        ar = each_final_range[1]

        denominator = denominator + (mp.ncdf(ar / tn_sigma) - mp.ncdf(al / tn_sigma))
        if tau >= ar:
            numerator = numerator + (mp.ncdf(ar / tn_sigma) - mp.ncdf(al / tn_sigma))
        elif (tau >= al) and (tau < ar):
            numerator = numerator + (mp.ncdf(tau / tn_sigma) - mp.ncdf(al / tn_sigma))

    if denominator != 0:
        F = numerator / denominator
        return 1 - F
    else:
        print('denominator = 0', final_list_range, tau, tn_sigma)
        return None


def dp_si(data, sigma, beta):
    x = np.array(data[:])
    x = x.reshape((x.shape[0], 1))

    n = len(data)

    sum_x = np.zeros(n)
    sum_x_matrix = []

    list_matrix = [[]]
    list_condition_matrix = []

    F = np.zeros(n + 1)
    cp = []

    for i in range(n):
        cp.append([])
        list_matrix.append([])
        if i == 0:
            sum_x[0] = data[0]

            e_n_0 = np.zeros(n)
            e_n_0[0] = 1
            e_n_0 = e_n_0.reshape((e_n_0.shape[0], 1))

            sum_x_matrix.append(e_n_0)
        else:
            sum_x[i] = sum_x[i-1] + data[i]

            e_n_i = np.zeros(n)
            e_n_i[i] = 1
            e_n_i = e_n_i.reshape((e_n_i.shape[0], 1))

            sum_x_matrix.append(sum_x_matrix[i - 1] + e_n_i)

    cp.append([])
    F[0] = - beta
    list_matrix[0] = [np.zeros((n, n)), - beta]

    for tstar in range(1, n + 1):
        F[tstar] = np.Inf
        current_opt_index = None

        list_matrix_A_plus_constant_b = []

        for t in range(tstar):
            if t > 0:
                temp = F[t] - (sum_x[tstar - 1] - sum_x[t - 1])**2 / (tstar - 1 - (t - 1)) \
                       + sigma**2 * np.log((tstar - 1 - (t - 1)) / n) + beta

                matrix_A = list_matrix[t][0] - \
                           np.dot(sum_x_matrix[tstar - 1] - sum_x_matrix[t-1],
                                  (sum_x_matrix[tstar - 1] - sum_x_matrix[t-1]).T) / (tstar - 1 - (t - 1))

                constant_b = list_matrix[t][1] + sigma**2 * np.log((tstar - 1 - (t - 1)) / n) + beta

                list_matrix_A_plus_constant_b.append([matrix_A, constant_b])

                if temp < F[tstar]:
                    F[tstar] = temp
                    current_opt_index = t
                    list_matrix[tstar] = [matrix_A, constant_b]

            elif t == 0:
                temp = F[t] - (sum_x[tstar - 1])**2 / (tstar - 1 - (t - 1)) \
                       + sigma**2 * np.log((tstar - 1 - (t - 1)) / n) + beta

                matrix_A = list_matrix[t][0] - \
                           np.dot(sum_x_matrix[tstar - 1],
                                  (sum_x_matrix[tstar - 1]).T) / (tstar - 1 - (t - 1))

                constant_b = list_matrix[t][1] + sigma ** 2 * np.log((tstar - 1 - (t - 1)) / n) + beta

                list_matrix_A_plus_constant_b.append([matrix_A, constant_b])

                if temp < F[tstar]:
                    F[tstar] = temp
                    current_opt_index = t
                    list_matrix[tstar] = [matrix_A, constant_b]

        for each_element in list_matrix_A_plus_constant_b:
            list_condition_matrix.append([list_matrix[tstar][0] - each_element[0], list_matrix[tstar][1] - each_element[1]])

        cp[tstar] = cp[current_opt_index - 1][:]
        cp[tstar].append(current_opt_index)

    cluster_index = np.zeros(n)

    final_cp = cp[-1][:]
    final_cp.append(n)

    for i in range(1, len(final_cp)):
        for j in range(final_cp[i - 1], final_cp[i]):
            cluster_index[j] = i

    return cluster_index, list_condition_matrix, len(final_cp) - 1, final_cp


def pelt(x, n, beta):

    def ssq(j, i, sum_x, sum_x_sq):
        if j > 0:
            muji = (sum_x[i] - sum_x[j-1]) / (i - j + 1)
            sji = sum_x_sq[i] - sum_x_sq[j-1] - (i - j + 1) * muji ** 2
        else:
            sji = sum_x_sq[i] - sum_x[i] ** 2 / (i+1)

        return 0 if sji < 0 else sji

    sum_x = np.zeros(n)
    sum_x_sq = np.zeros(n)

    F = np.zeros(n + 1)
    R = []
    cp = []

    for i in range(n):
        cp.append([])
        R.append([])
        if i == 0:
            sum_x[0] = x[0]
            sum_x_sq[0] = x[0] ** 2
        else:
            sum_x[i] = sum_x[i-1] + x[i]
            sum_x_sq[i] = sum_x_sq[i - 1] + x[i] ** 2

    cp.append([])
    R.append([])
    F[0] = - beta
    R[1].append(0)

    for tstar in range(1, n + 1):
        F[tstar] = np.Inf
        current_opt_index = None

        for t in R[tstar]:
            temp = F[t] + ssq(t, tstar - 1, sum_x, sum_x_sq) + beta

            if temp < F[tstar]:
                F[tstar] = temp
                current_opt_index = t

        if tstar < n:
            for t in R[tstar]:
                temp = F[t] + ssq(t, tstar - 1, sum_x, sum_x_sq)

                if temp <= F[tstar]:
                    R[tstar + 1].append(t)

            R[tstar + 1].append(tstar)

        cp[tstar] = cp[current_opt_index][:]
        cp[tstar].append(current_opt_index)

    cp[-1].append(n)

    return cp[-1], F[-1]


def parametrize_x(x, n, cp_idx_left, cp_idx_curr, cp_idx_right, cov):
    x_vec = np.reshape(x, (n, 1))

    seg_1_vec = np.zeros(n)
    seg_1_len = 0
    seg_2_vec = np.zeros(n)
    seg_2_len = 0

    for i in range(cp_idx_left, cp_idx_curr):
        seg_1_vec[i] = 1.0
        seg_1_len += 1

    for i in range(cp_idx_curr, cp_idx_right):
        seg_2_vec[i] = 1
        seg_2_len += 1

    eta = seg_1_vec / seg_1_len - seg_2_vec / seg_2_len
    eta_vec = eta.reshape(n, 1)
    eta_T_x = np.dot(eta_vec.T, x_vec)[0][0]

    eta_T_cov_eta = np.dot(eta_vec.T, np.dot(cov, eta_vec))[0][0]
    cov_eta = np.dot(cov, eta_vec)

    x_prime = []

    for i in range(n):
        a = cov_eta[i][0] / eta_T_cov_eta
        b = x[i] - (eta_T_x / eta_T_cov_eta) * cov_eta[i][0]
        x_prime.append([a, b])

    return x_prime, eta_vec, eta_T_x


def check_min_q_term(f_tau_t, min_quad_term, min_quad_term_funct_cost):
    if f_tau_t[0] < min_quad_term:
        return True
    elif f_tau_t[0] == min_quad_term:
        if f_tau_t[1] > min_quad_term_funct_cost[1][1]:
            return True
        elif f_tau_t[1] == min_quad_term_funct_cost[1][1]:
            if f_tau_t[2] < min_quad_term_funct_cost[1][2]:
                return True

    return False




def quadratic_solver(a, b, c):
    delta = b**2 - 4*a*c

    if delta < 0:
        return None, None

    sqrt_delta = np.sqrt(delta)
    x_1 = (-b - sqrt_delta) / (2*a)
    x_2 = (-b + sqrt_delta) / (2*a)

    if x_1 <= x_2:
        return x_1, x_2
    else:
        return x_2, x_1


def quadratic_min(a, b, c):
    if a == 0:
        if b != 0:
            print('ERROR')
        else:
            return 0, c
    else:
        min_z = (-b) / (2 * a)
        min_f = a * (min_z ** 2) + b * min_z + c
        return min_z, min_f


def second_pruning(list_remove_funct_cand, list_z, list_opt_funct_corres_to_z, beta):
    temp_list_remove_funct_cand = list_remove_funct_cand.copy()

    list_remove_key = []
    z_curr = list_z[0]
    funct_curr_key = list_opt_funct_corres_to_z[0][0]
    funct_curr = list_opt_funct_corres_to_z[0][1]

    new_temp_list_remove_funct_cand = temp_list_remove_funct_cand.copy()
    for key, funct in temp_list_remove_funct_cand.items():
        if funct[0] < funct_curr[0]:
            del new_temp_list_remove_funct_cand[key]
        elif funct[0] == funct_curr[0]:
            if funct[1] > funct_curr[1]:
                del new_temp_list_remove_funct_cand[key]
            elif funct[1] == funct_curr[1]:
                if (funct[2] - beta) < funct_curr[2]:
                    del new_temp_list_remove_funct_cand[key]
                elif (funct[2] - beta) >= funct_curr[2]:
                    list_remove_key.append(key)
                    del new_temp_list_remove_funct_cand[key]

    temp_list_remove_funct_cand = new_temp_list_remove_funct_cand.copy()

    for i in range(1, len(list_z)):
        new_temp_list_remove_funct_cand = temp_list_remove_funct_cand.copy()
        next_z_curr = list_z[i]

        for key, funct in temp_list_remove_funct_cand.items():
            a = funct[0] - funct_curr[0]
            b = funct[1] - funct_curr[1]
            c = funct[2] - funct_curr[2] - beta

            if a == 0:
                if b == 0:
                    if c < 0:
                        print('error')

                    del new_temp_list_remove_funct_cand[key]
                    list_remove_key.append(key)
                else:
                    x_1 = x_2 = - c / b
                    if x_1 <= z_curr:
                        del new_temp_list_remove_funct_cand[key]
                        list_remove_key.append(key)
                    else:
                        if x_1 <= next_z_curr:
                            del new_temp_list_remove_funct_cand[key]

            else:
                x_1, x_2 = quadratic_solver(a, b, c)

                if (x_1 is None) and (x_2 is None):
                    del new_temp_list_remove_funct_cand[key]
                    list_remove_key.append(key)

                else:
                    if x_2 <= z_curr:
                        del new_temp_list_remove_funct_cand[key]
                        list_remove_key.append(key)
                    elif (x_1 <= next_z_curr) or (x_2 <= next_z_curr):
                        del new_temp_list_remove_funct_cand[key]

        temp_list_remove_funct_cand = new_temp_list_remove_funct_cand.copy()
        z_curr = next_z_curr
        funct_curr = list_opt_funct_corres_to_z[i][1]

    return list_remove_key

def find_opt_funct_set(local_f_cost_dict, min_quad_term_funct_cost):
    tempt_f_cost_dict = local_f_cost_dict.copy()
    opt_funct_set = {}
    list_remove_funct = {}

    z_curr = np.NINF
    funct_curr_key = min_quad_term_funct_cost[0]
    funct_curr = min_quad_term_funct_cost[1]

    list_z = [z_curr]
    list_opt_funct_corres_to_z = [[funct_curr_key, funct_curr]]

    opt_funct_set.update({funct_curr_key: funct_curr})

    while len(tempt_f_cost_dict) > 1:
        new_funct_set = tempt_f_cost_dict.copy()
        list_new_z = []

        for key, funct in tempt_f_cost_dict.items():
            if np.array_equal(funct, funct_curr):
                if key != funct_curr_key:
                    list_remove_funct.update({key: funct})
                    del new_funct_set[key]

                continue

            a = funct[0] - funct_curr[0]
            b = funct[1] - funct_curr[1]
            c = funct[2] - funct_curr[2]

            a = check_zero(a)
            b = check_zero(b)
            c = check_zero(c)

            if a == 0:
                if b == 0:
                    if key not in opt_funct_set:
                        list_remove_funct.update({key: funct})

                    del new_funct_set[key]
                else:
                    x_1 = x_2 = - c / b
                    if x_1 <= z_curr:
                        if key not in opt_funct_set:
                            list_remove_funct.update({key: funct})

                        del new_funct_set[key]
                    else:
                        list_new_z.append([x_1, key])
            else:
                x_1, x_2 = quadratic_solver(a, b, c)

                if (x_1 is None) and (x_2 is None):

                    if key not in opt_funct_set:
                        list_remove_funct.update({key: funct})

                    del new_funct_set[key]
                else:
                    if x_2 <= z_curr:
                        if key not in opt_funct_set:
                            list_remove_funct.update({key: funct})

                        del new_funct_set[key]
                    elif x_1 <= z_curr < x_2:
                        list_new_z.append([x_2, key])
                    elif z_curr < x_1:
                        list_new_z.append([x_1, key])

        if len(list_new_z) == 0:
            break

        sorted_list_new_z = sorted(list_new_z)

        z_curr = sorted_list_new_z[0][0]
        funct_curr_key = sorted_list_new_z[0][1]
        funct_curr = tempt_f_cost_dict[funct_curr_key]

        list_z.append(z_curr)
        list_opt_funct_corres_to_z.append([funct_curr_key, funct_curr])
        opt_funct_set.update({funct_curr_key: funct_curr})

        tempt_f_cost_dict = new_funct_set.copy()

    return opt_funct_set, list_remove_funct, list_z, list_opt_funct_corres_to_z


def pelt_perturbed_x(x_prime, sg_results, n, beta):

    def ssq(j, i, x_prime_sum, x_prime_sum_sq):
        if j > 0:
            muji = (x_prime_sum[i] - x_prime_sum[j - 1]) / (i - j + 1)
            a = muji[0]
            b = muji[1]
            sji = x_prime_sum_sq[i] - x_prime_sum_sq[j - 1] - (i - j + 1) * np.array([a**2, 2*a*b, b**2])
        else:
            muji = (x_prime_sum[i]) / (i - j + 1)
            a = muji[0]
            b = muji[1]
            sji = x_prime_sum_sq[i] - (i - j + 1) * np.array([a ** 2, 2 * a * b, b ** 2])

        return sji

    opt_funct_set = None
    sum_x_prime = []
    sum_x_prime_sq = []

    global_f_cost_dict = {str([-1]): np.array([0, 0, -beta])}
    T_curr = [[-1]]

    for i in range(n):
        a = x_prime[i][0]
        b = x_prime[i][1]
        
        if i == 0:
            sum_x_prime.append(np.array([a, b]))
            sum_x_prime_sq.append(np.array([a ** 2, 2 * a * b, b ** 2]))
        else:
            sum_x_prime.append(sum_x_prime[i - 1] + np.array([a, b]))
            sum_x_prime_sq.append(sum_x_prime_sq[i - 1] + np.array([a ** 2, 2 * a * b, b ** 2]))

    for t in range(n):
        local_f_cost_dict = {}

        min_quad_term = np.Inf
        min_quad_term_funct_cost = None

        for tau in T_curr:
            if str(tau) not in global_f_cost_dict:
                f_tau = global_f_cost_dict[str(tau[:-1])] + ssq(tau[-2] + 1, tau[-1], sum_x_prime, sum_x_prime_sq) + np.array([0, 0, beta])
                global_f_cost_dict.update({str(tau): f_tau})

            if tau == [-1]:
                f_tau_t = ssq(0, t, sum_x_prime, sum_x_prime_sq)
                f_tau_t = check_coef_zero(f_tau_t)
                local_f_cost_dict.update({str(tau): f_tau_t})

                if check_min_q_term(f_tau_t, min_quad_term, min_quad_term_funct_cost):
                    min_quad_term = f_tau_t[0]
                    min_quad_term_funct_cost = [str(tau), f_tau_t]
            else:
                f_tau_t = global_f_cost_dict[str(tau)] + ssq(tau[-1] + 1, t, sum_x_prime, sum_x_prime_sq) + np.array([0, 0, beta])
                f_tau_t = check_coef_zero(f_tau_t)
                local_f_cost_dict.update({str(tau): f_tau_t})

                if check_min_q_term(f_tau_t, min_quad_term, min_quad_term_funct_cost):
                    min_quad_term = f_tau_t[0]
                    min_quad_term_funct_cost = [str(tau), f_tau_t]

        opt_funct_set, list_remove_funct, list_z, list_opt_funct_corres_to_z = \
            find_opt_funct_set(local_f_cost_dict, min_quad_term_funct_cost)

        T_new = T_curr[:]

        list_remove_key = second_pruning(list_remove_funct, list_z, list_opt_funct_corres_to_z, beta)

        for key in list_remove_key:
            key_list_type = ast.literal_eval(key)
            T_new.remove(key_list_type)

        for key, funct in opt_funct_set.items():
            key_list_type = ast.literal_eval(key)
            key_list_type.append(t)
            new_tau = key_list_type
            T_new.append(new_tau)

        T_curr = T_new[:]

    opt_funct = np.array([0, 0, - beta])
    for i in range(len(sg_results) - 1):
        curr_cp = sg_results[i]
        next_cp = sg_results[i + 1]
        opt_funct = opt_funct + ssq(curr_cp, next_cp - 1, sum_x_prime, sum_x_prime_sq) + np.array([0, 0, beta])

    return opt_funct_set, opt_funct


def union(range_1, range_2):
    lower = max(range_1[0], range_2[0])
    upper = min(range_1[1], range_2[1])

    if upper < lower:
        if range_1[1] < range_2[0]:
            return 2, [range_1, range_2]
        else:
            return 2, [range_2, range_1]
    else:
        return 1, [min(range_1[0], range_2[0]), max(range_1[1], range_2[1])]


def inference(opt_funct_set, opt_funct, eta_vec, eta_T_x, cov, n):
    tn_sigma = np.sqrt(np.dot(eta_vec.T, np.dot(cov, eta_vec))[0][0])
    list_interval = []

    for k, func_cost in opt_funct_set.items():
        a = func_cost[0] - opt_funct[0]
        b = func_cost[1] - opt_funct[1]
        c = func_cost[2] - opt_funct[2]

        a = check_zero(a)
        b = check_zero(b)
        c = check_zero(c)

        if (a == 0) and (b != 0):
            print('a == 0 and b != 0')

        if a == 0:
            continue

        x_1, x_2 = quadratic_solver(a, b, c)
        if (x_1 is None) and (x_2 is None):
            if a < 0:
                print('negative a')
            continue

        elif x_1 == x_2:
            if a > 0:
                list_interval.append([np.NINF, x_1])
            elif a < 0:
                list_interval.append([x_2, np.Inf])

        else:
            if a > 0:
                list_interval.append([x_1, x_2])
            elif a < 0:
                list_interval.append([np.NINF, x_1])
                list_interval.append([x_2, np.Inf])

    sorted_list_interval = sorted(list_interval)

    union_interval = [sorted_list_interval[0]]
    for element in sorted_list_interval:
        no_of_ranges, returned_interval = union(union_interval[-1], element)
        if no_of_ranges == 2:
            union_interval[-1] = returned_interval[0]
            union_interval.append(returned_interval[1])
        else:
            union_interval[-1] = returned_interval

    z_interval = [[np.NINF, union_interval[0][0]]]

    for i in range(len(union_interval) - 1):
        z_interval.append([union_interval[i][1], union_interval[i + 1][0]])

    z_interval.append([union_interval[-1][1], np.Inf])

    negative_eta = - abs(eta_T_x)
    positive_eta = abs(eta_T_x)

    numerator_1 = 0
    numerator_2 = 0
    denominator = 0

    for each_interval in z_interval:
        al = each_interval[0]
        ar = each_interval[1]

        denominator = denominator + (mp.ncdf(ar / tn_sigma) - mp.ncdf(al / tn_sigma))

        if negative_eta >= ar:
            numerator_1 = numerator_1 + (mp.ncdf(ar / tn_sigma) - mp.ncdf(al / tn_sigma))
        elif (negative_eta >= al) and (negative_eta < ar):
            numerator_1 = numerator_1 + (mp.ncdf(negative_eta / tn_sigma) - mp.ncdf(al / tn_sigma))

        if positive_eta >= ar:
            numerator_2 = numerator_2 + (mp.ncdf(ar / tn_sigma) - mp.ncdf(al / tn_sigma))
        elif (positive_eta >= al) and (positive_eta < ar):
            numerator_2 = numerator_2 + (mp.ncdf(positive_eta / tn_sigma) - mp.ncdf(al / tn_sigma))

    if denominator != 0:
        p_value = numerator_1 / denominator + (1 - numerator_2 / denominator)
        return float(p_value)
    else:
        return None