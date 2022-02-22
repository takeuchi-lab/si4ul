import copy
import numpy as np

from si4ul.si.sicore.sicore import SelectiveInferenceNormSE, NaiveInferenceNorm
from si4ul.si.sicore.sicore.utils import OneVec
from si4ul.si.sicore.sicore.intervals import union_all


class MultiDimCusum:
    def __init__(self, data):
        self.data = data
        self.length = data.shape[1]
        self.dim = data.shape[0]
        self.cumsum = np.concatenate(
            [np.zeros((self.dim, 1)), np.cumsum(data, axis=1)], 1
        )

    def cost(self, s, t, e):
        left_length = t - s
        right_length = e - t
        mean_vec = (self.cumsum[:, t] - self.cumsum[:, s]) / left_length - (
            self.cumsum[:, e] - self.cumsum[:, t]
        ) / right_length
        return np.sqrt((t - s) * (e - t) / (e - s)) * mean_vec

    def cost_width(self, s, t, e, width):
        mean_agg_vec = np.zeros((self.dim))
        for w in range(-width, width + 1):
            mean_agg_vec += self.cost(s, t - w, e)
        return mean_agg_vec


# CP detection with Double-CUSUM
class DP_DC:
    def __init__(self, X, K, minseq, width, phi):
        """
        [Parameters]
            X: <ndarray>
                data
            K: int
                num of change point
            minseq: int
                minimum length
            width: int
                same cp width
            phi: double
                Double CUSUM's parameter
                phi=0.5 is optimal asymptotic theory
        [Arguments]
            N: <int>
                length of series(each series fix length)
            d: <int>
                num of series
            F_dict: <dict> (split, length)
                scan value dictionary
            cp_dict: <dict> (split, length)
                change point dictionary
            dim_dict: <dict> (split, length)
                change dimension dictionary
            sort_dict: <dict> (location)
                sort index dictionary
            sign_dict: <dict> (location)
                sign of mean difference dictionary
        """
        self.X = X
        self.K = K
        self.msl = minseq
        self.W = width
        self.phi = phi
        self.N = np.shape(X)[1]
        self.d = np.shape(X)[0]
        self.F_dict = dict()
        self.cp_dict = dict()
        self.dim_dict = dict()
        self.sort_dict = dict()
        self.sign_dict = dict()
        self.cost = MultiDimCusum(X)

    def detection(self):
        """
        [Returns]
            cp: <list>
                change point list
            dim: <multi list>
                change dim list
            cp_dict, dim_dict
        """
        r = self.N - (self.K + 1) * self.msl
        self.initialize()

        # Update DP
        if self.K > 0:
            for k in range(1, self.K):
                for n in range((k + 1) * self.msl, (k + 1) * self.msl + r + 1):
                    self.calc_dp(k, n)
            self.calc_dp(self.K, self.N)

        for cp in self.cp_dict:
            _, n = cp
            self.cp_dict[cp].append(n)

        for dim in self.dim_dict:
            self.dim_dict[dim] = self.dim_dict[dim][1:]

        cp = self.cp_dict[(self.K, self.N)]
        dim = self.dim_dict[(self.K, self.N)]

        return cp, dim, self.cp_dict, self.dim_dict, self.sign_dict, self.sort_dict

    def initialize(self):
        r = self.N - (self.K + 1) * self.msl
        for n in range(self.msl, self.msl + r + 1):
            self.cp_dict[(0, n)] = [0]
            self.dim_dict[(0, n)] = [[]]
            self.F_dict[(0, n)] = 0

    def calc_dp(self, k, n):
        """
        [Parameters]
            k: <int>
                split size
            n: <int>
                subsequence length
        """
        Fmax = np.NINF
        for m in range(k * self.msl, n - self.msl + 1):
            cost_list = self.cost.cost_width(m - self.msl, m, m + self.msl, self.W)
            self.sign_dict[m] = np.sign(cost_list)
            cost_list = abs(cost_list)
            sort_index = np.argsort(-cost_list)
            self.sort_dict[m] = sort_index

            cost_list = np.sort(cost_list)[::-1]
            cumsum_dim = []
            for i in sort_index:
                cumsum_dim.append(i)
                p = len(cumsum_dim)
                const = (p * (2 * self.d - p) / (2 * self.d)) ** self.phi
                cumsum_cost = const * (
                    np.sum(cost_list[:p]) / p - np.sum(cost_list[p:]) / (2 * self.d - p)
                )
                tmp = self.F_dict[(k - 1, m)] + cumsum_cost
                if tmp > Fmax:
                    cp_hat = m
                    dim_hat = copy.deepcopy(cumsum_dim)
                    Fmax = tmp
        self.F_dict[(k, n)] = Fmax
        self.cp_dict[(k, n)] = self.cp_dict[(k - 1, cp_hat)] + [cp_hat]
        self.dim_dict[(k, n)] = self.dim_dict[(k - 1, cp_hat)] + [dim_hat]


# CP detection with scan
class DP_Scan:
    def __init__(self, X, K, minseq, width):
        """
        input
            X: <ndarray>
                data
            K: int
                num of change point
            minseq: int
                minimum length
            width: int
                consider equal cp width (width < minseq)
        arg
            N: <int>
                length of series(each series fix length)
            d: <int>
                num of series
            F_dict: <dict> (split, length)
                scan value dictionary
            cp_dict: <dict> (split, length)
                change point dictionary
            dim_dict: <dict> (split, length)
                change dimension dictionay
        """
        self.X = X
        self.K = K
        self.msl = minseq
        self.W = width
        self.N = np.shape(X)[1]
        self.d = np.shape(X)[0]
        self.F_dict = dict()
        self.cp_dict = dict()
        self.dim_dict = dict()
        self.dim_sort_dict = dict()
        self.cost = MultiDimCusum(X)

    def detection(self):
        """
        CP detection

        [Returns]
            cp: <list>
                change point list
            dim: <multi list>
                change dim list
            cp_dict, dim_dict, dim_sort: <dict>
                selection of table
        """
        r = self.N - (self.K + 1) * self.msl
        self.initialize()

        # Update DP
        if self.K > 0:
            for k in range(1, self.K):
                for n in range((k + 1) * self.msl, (k + 1) * self.msl + r + 1):
                    self.calc_dp(k, n)
            self.calc_dp(self.K, self.N)

        for cp in self.cp_dict:
            _, n = cp
            self.cp_dict[cp].append(n)

        for dim in self.dim_dict:
            self.dim_dict[dim] = self.dim_dict[dim][1:]

        cp = self.cp_dict[(self.K, self.N)]
        dim = self.dim_dict[(self.K, self.N)]

        return cp, dim, self.cp_dict, self.dim_dict, self.dim_sort_dict

    def initialize(self):
        r = self.N - (self.K + 1) * self.msl
        for n in range(self.msl, self.msl + r + 1):
            self.cp_dict[(0, n)] = [0]
            self.dim_dict[(0, n)] = [[]]
            self.F_dict[(0, n)] = 0

    def calc_dp(self, k, n):
        """
        Updata DP
        [Parameters]
        k: <int>
            row of table
        n: <int>
            col of table
        """
        Fmax = np.NINF
        for m in range(k * self.msl, n - self.msl + 1):
            cost_list = self.cost.cost_width(m - self.msl, m, m + self.msl, self.W) ** 2
            sort_index = np.argsort(-cost_list)
            self.dim_sort_dict[(k, n, m)] = sort_index
            cumsum_cost = 0
            cumsum_dim = []
            for i in sort_index:
                cumsum_dim.append(i)
                cumsum_cost += cost_list[i]
                p = len(cumsum_dim)
                tmp = self.F_dict[(k - 1, m)] + (cumsum_cost - p) / np.sqrt(2 * p)
                if tmp > Fmax:
                    cp_hat = m
                    dim_hat = copy.deepcopy(cumsum_dim)
                    Fmax = tmp
        self.F_dict[(k, n)] = Fmax
        self.cp_dict[(k, n)] = self.cp_dict[(k - 1, cp_hat)] + [cp_hat]
        self.dim_dict[(k, n)] = self.dim_dict[(k - 1, cp_hat)] + [dim_hat]


# SI for Double-CUSUM
class DoubleCusumSI:
    def __init__(self, obs_X, K, L, det_cls, tau, theta, cov, result, width, phi):
        self.X = obs_X
        self.vecX = obs_X.flatten()
        self.d = obs_X.shape[0]
        self.length = obs_X.shape[1]
        self.K = K
        self.L = L
        self.det_cls = det_cls
        self.tau = tau
        self.theta = theta
        self.cov = cov
        self.result = result
        self.width = width
        self.phi = phi
        self.eta = self.make_eta()
        self.si = SelectiveInferenceNormSE(self.vecX, self.cov, self.eta)
        self.a = self.si.c
        self.b = self.si.z
        self.intervals = None

    def make_eta(self):
        onevec_n = OneVec(self.length)
        onevec_d = OneVec(self.d)
        left = self.tau - self.L + 1
        right = self.tau + self.L
        center_l = self.tau - self.width
        center_r = self.tau + self.width
        left_vec = onevec_n.get(left, center_l) / (center_l - left + 1)
        right_vec = onevec_n.get(center_r, right) / (right - center_r + 1)
        eta_n = (left_vec - right_vec).reshape(-1, 1)
        e_c = onevec_d.get(self.theta + 1).reshape(-1, 1)
        eta = np.kron(e_c, eta_n).reshape(-1)
        return eta

    def si_oc(self, cp_dict, dim_dict, sign, sort, homotopy=False):
        r = self.length - (self.K + 1) * self.L
        one_d = OneVec(self.d)
        one_n = OneVec(self.length)
        si = SelectiveInferenceNormSE(self.vecX, self.cov, self.eta)

        def sort_event(m):
            mean = make_mean(m)
            sort_list = sort[m]
            sign_list = sign[m]
            for i, j in zip(sort_list, sort_list[1:]):
                bet_i = sign_list[i] * np.kron(one_d.get(i + 1), mean)
                bet_j = sign_list[j] * np.kron(one_d.get(j + 1), mean)
                bet = bet_j - bet_i
                beta, gamma = calc_coeff(bet, self.a, self.b)
                si.cut_interval(0, beta, gamma)

            # 最後の要素についてのイベント
            j = sort_list[-1]
            bet = -(sign_list[j] * np.kron(one_d.get(j + 1), mean))
            beta, gamma = calc_coeff(bet, self.a, self.b)
            si.cut_interval(0, beta, gamma)

        def selection_event(k, n, m):
            tau_n = cp_dict[(k, n)]
            dim_n = dim_dict[(k, n)]
            tau_m = cp_dict[(k - 1, m)]
            dim_m = dim_dict[(k - 1, m)]

            # tau_n > tau_m
            base_bet = 0
            for i in range(1, k + 1):
                sign_list = sign[tau_n[i]]
                mean = make_mean(tau_n[i])
                dims = make_dim(dim_n[i - 1], sign_list)
                base_bet -= np.kron(dims, mean)

            for i in range(1, k):
                sign_list = sign[tau_m[i]]
                mean = make_mean(tau_m[i])
                dims = make_dim(dim_m[i - 1], sign_list)
                base_bet += np.kron(dims, mean)

            sort_list = sort[m]
            sign_list = sign[m]
            mean = make_mean(m)
            for p in range(1, self.d + 1):
                dims = make_dim(sort_list[:p], sign_list)
                bet = base_bet + np.kron(dims, mean)
                beta, gamma = calc_coeff(bet, self.a, self.b)
                si.cut_interval(0, beta, gamma)

        def make_mean(m):
            s = m - self.L
            e = m + self.L
            mean = np.zeros(self.length)
            for w in range(-self.width, self.width + 1):
                const = (m + w - s) * (e - m - w) / (e - s)
                mean_left = one_n.get(s + 1, m + w) / (m + w - s)
                mean_right = one_n.get(m + w + 1, e) / (e - m - w)
                mean += np.sqrt(const) * (mean_left - mean_right)
            return mean

        def make_dim(dim, sign):
            p = len(dim)
            const = (p * (2 * self.d - p) / (2 * self.d)) ** self.phi
            dims = np.zeros(self.d)
            for i in range(self.d):
                if i in dim:
                    dims[i] = 1 / p
                else:
                    dims[i] = -1 / (2 * self.d - p)
            return const * sign * dims

        def calc_coeff(bet, a, b):
            beta = np.dot(bet, a)
            gamma = np.dot(bet, b)
            return beta, gamma

        for k in range(1, self.K):
            for n in range((k + 1) * self.L, (k + 1) * self.L + r + 1):
                for m in range(k * self.L, n - self.L + 1):
                    sort_event(m)
                    selection_event(k, n, m)
        for m in range(self.K * self.L, self.length - self.L + 1):
            sort_event(m)
            selection_event(self.K, self.length, m)

        interval_oc = si.get_intervals()
        if homotopy:
            return interval_oc
        else:
            self.intervals = interval_oc

    def si_homotopy(self, z_min, z_max, epsilon=1e-5):
        z = z_min
        Z = [z]
        result_path = []
        while z < z_max:
            Xz = (self.a * z + self.b).reshape(self.d, self.length)
            dp = self.det_cls(Xz, self.K, self.L, self.width, self.phi)
            cp, dim, cp_dict, dim_dict, sign_dict, sort_dict = dp.detection()

            intervals = self.si_oc(
                cp_dict, dim_dict, sign_dict, sort_dict, homotopy=True
            )

            if len(intervals) == 1:
                z = intervals[0][1]
            else:
                for interval in intervals:
                    if interval[0] < z < interval[1]:
                        z = interval[1]
                        break
            if z < z_max:
                Z.append(z)
            else:
                Z.append(z_max)

            result_path.append([cp, dim])
            z += epsilon

        z_interval = []
        for i, result in enumerate(result_path):
            cp_, dim_ = result
            if self.tau in cp_:
                index_ = cp_.index(self.tau)
                if self.theta in dim_[index_ - 1]:
                    z_interval.append([Z[i], Z[i + 1]])
        z_interval = union_all(z_interval)
        self.intervals = z_interval

    def test(self, homotopy=True, z_min=-10, z_max=10):
        if homotopy:
            self.si_homotopy(z_min, z_max)
        else:
            _, _, cp_dict, dim_dict, sign, sort = self.result
            self.si_oc(cp_dict, dim_dict, sign, sort)
        return self.si.test(intervals=self.intervals)

    def test_naive(self):
        naive = NaiveInferenceNorm(self.vecX, self.cov, self.eta)
        return naive.test()


# SI for scan
class ScanSI:
    def __init__(self, obs_X, K, L, det_cls, tau, theta, cov, result, width):
        self.X = obs_X
        self.vecX = obs_X.flatten()
        self.d = obs_X.shape[0]
        self.n = obs_X.shape[1]
        self.K = K
        self.L = L
        self.width = width
        self.det_cls = det_cls
        self.tau = tau
        self.theta = theta
        self.cov = cov
        self.result = result
        self.width = width
        self.eta = self.make_eta()
        self.si = SelectiveInferenceNormSE(self.vecX, self.cov, self.eta)
        self.a = self.si.c
        self.b = self.si.z
        self.intervals = None

    def make_eta(self):
        onevec_n = OneVec(self.n)
        onevec_d = OneVec(self.d)
        left = self.tau - self.L + 1
        right = self.tau + self.L
        center_l = self.tau - self.width
        center_r = self.tau + self.width
        left_vec = onevec_n.get(left, center_l) / (center_l - left + 1)
        right_vec = onevec_n.get(center_r, right) / (right - center_r + 1)
        eta_n = (left_vec - right_vec).reshape(-1, 1)
        e_c = onevec_d.get(self.theta + 1).reshape(-1, 1)
        eta = np.kron(e_c, eta_n).reshape(-1)
        return eta

    def si_oc(self, cp_dict, dim_dict, dim_sort, homotopy=False):
        onevec = OneVec(self.n)
        r = self.n - (self.K + 1) * self.L
        si = SelectiveInferenceNormSE(self.vecX, self.cov, self.eta)

        # selection event for update dp
        def selection_event(k, n, m, vc):
            tau_n = cp_dict[(k, n)]
            tau_m = cp_dict[(k - 1, m)]
            dim_n = dim_dict[(k, n)]
            dim_m = dim_dict[(k - 1, m)]

            a_minus = []
            b_minus = 0
            a_plus = []
            b_plus = 0

            for i in range(1, k + 1):
                p = len(dim_n[i - 1])
                b_minus -= np.sqrt(2 * p) / 2
                for dim in dim_n[i - 1]:
                    one_dim = np.array(
                        [1 if j == dim else 0 for j in range(self.d)]
                    ).reshape(-1, 1)
                    mean_vec = np.zeros((self.n, 1))
                    for w in range(-self.width, self.width + 1):
                        one_left = onevec.get(
                            tau_n[i] - self.L + 1, tau_n[i] + w
                        ).reshape(-1, 1) / (self.L + w)
                        one_right = onevec.get(
                            tau_n[i] + 1 + w, tau_n[i] + self.L
                        ).reshape(-1, 1) / (self.L - w)
                        one_n = (one_left - one_right) * np.sqrt(
                            (self.L ** 2 - w ** 2) / (2 * self.L)
                        )
                        mean_vec += one_n / np.sqrt(np.sqrt(2 * p))
                    a_minus.append(np.kron(one_dim, mean_vec).reshape(-1))

            for i in range(1, k):
                p = len(dim_m[i - 1])
                b_plus -= np.sqrt(2 * p) / 2
                for dim in dim_m[i - 1]:
                    one_dim = np.array(
                        [1 if j == dim else 0 for j in range(self.d)]
                    ).reshape(-1, 1)
                    mean_vec = np.zeros((self.n, 1))
                    for w in range(-self.width, self.width + 1):
                        one_left = onevec.get(
                            tau_m[i] - self.L + 1, tau_m[i] + w
                        ).reshape(-1, 1) / (self.L + w)
                        one_right = onevec.get(
                            tau_m[i] + 1 + w, tau_m[i] + self.L
                        ).reshape(-1, 1) / (self.L - w)
                        one_m = (one_left - one_right) * np.sqrt(
                            (self.L ** 2 - w ** 2) / (2 * self.L)
                        )
                        mean_vec += one_m / np.sqrt(np.sqrt(2 * p))
                    a_plus.append(np.kron(one_dim, mean_vec).reshape(-1))

            p = len(vc)
            b_plus -= np.sqrt(2 * p) / 2
            for dim in vc:
                one_dim = np.array(
                    [1 if j == dim else 0 for j in range(self.d)]
                ).reshape(-1, 1)
                mean_vec = np.zeros((self.n, 1))
                for w in range(-self.width, self.width + 1):
                    one_left = onevec.get(m - self.L + 1, m + w).reshape(-1, 1) / (
                        self.L + w
                    )
                    one_right = onevec.get(m + 1 + w, m + self.L).reshape(-1, 1) / (
                        self.L - w
                    )
                    one_t = (one_left - one_right) * np.sqrt(
                        (self.L ** 2 - w ** 2) / (2 * self.L)
                    )
                    mean_vec += one_t / np.sqrt(np.sqrt(2 * p))
                a_plus.append(np.kron(one_dim, mean_vec).reshape(-1))

            return a_minus, a_plus, b_minus, b_plus

        # selection event for sort
        def event_sort(k, n, m, ldim, sdim):
            a_minus = []
            one_dim = np.array([1 if j == ldim else 0 for j in range(self.d)]).reshape(
                -1, 1
            )
            mean_vec = np.zeros((self.n, 1))
            for w in range(-self.width, self.width + 1):
                one_left = onevec.get(m - self.L + 1, m + w).reshape(-1, 1) / (
                    self.L + w
                )
                one_right = onevec.get(m + 1 + w, m + self.L).reshape(-1, 1) / (
                    self.L - w
                )
                mean_vec += (one_left - one_right) * np.sqrt(
                    (self.L ** 2 - w ** 2) / (2 * self.L)
                )
            a_minus.append(np.kron(one_dim, mean_vec).reshape(-1))

            a_plus = []
            one_dim = np.array([1 if j == sdim else 0 for j in range(self.d)]).reshape(
                -1, 1
            )
            mean_vec = np.zeros((self.n, 1))
            for w in range(-self.width, self.width + 1):
                one_left = onevec.get(m - self.L + 1, m + w).reshape(-1, 1) / (
                    self.L + w
                )
                one_right = onevec.get(m + 1 + w, m + self.L).reshape(-1, 1) / (
                    self.L - w
                )
                mean_vec += (one_left - one_right) * np.sqrt(
                    (self.L ** 2 - w ** 2) / (2 * self.L)
                )
            a_plus.append(np.kron(one_dim, mean_vec).reshape(-1))

            return a_minus, a_plus

        # calculation of alpha, beta, gamma
        def calc_coeff(a_minus, a_plus, b_minus, b_plus):
            alpha = 0
            beta = 0
            gamma = b_plus - b_minus
            for a in a_minus:
                ca = np.dot(self.a, a)
                za = np.dot(self.b, a)
                alpha -= ca ** 2
                beta -= 2 * ca * za
                gamma -= za ** 2
            for a in a_plus:
                ca = np.dot(self.a, a)
                za = np.dot(self.b, a)
                alpha += ca ** 2
                beta += 2 * ca * za
                gamma += za ** 2

            return alpha, beta, gamma

        # calculation of over-conditioning
        for k in range(1, self.K):
            for n in range((k + 1) * self.L, (k + 1) * self.L + r + 1):
                for m in range(k * self.L, n - self.L + 1):
                    for i in range(len(dim_sort[(k, n, m)])):
                        large = dim_sort[(k, n, m)][i]
                        if i == self.d - 1:
                            small = self.d
                        else:
                            small = dim_sort[(k, n, m)][i + 1]
                        a_minus, a_plus = event_sort(k, n, m, large, small)
                        alp, beta, gam = calc_coeff(a_minus, a_plus, 0, 0)
                        si.cut_interval(alp, beta, gam)
                    for index in range(1, self.d + 1):
                        vc = dim_sort[(k, n, m)][:index]
                        a_m, a_p, b_m, b_p = selection_event(k, n, m, vc)
                        alp, beta, gam = calc_coeff(a_m, a_p, b_m, b_p)
                        si.cut_interval(alp, beta, gam)

        for m in range(self.K * self.L, self.n - self.L + 1):
            for i in range(len(dim_sort[(self.K, self.n, m)])):
                large = dim_sort[(self.K, self.n, m)][i]
                if i == self.d - 1:
                    small = self.d
                else:
                    small = dim_sort[(self.K, self.n, m)][i + 1]
                a_minus, a_plus = event_sort(self.K, self.n, m, large, small)
                alp, beta, gam = calc_coeff(a_minus, a_plus, 0, 0)
                si.cut_interval(alp, beta, gam)

            for index in range(1, self.d + 1):
                vc = dim_sort[(self.K, self.n, m)][:index]
                a_m, a_p, b_m, b_p = selection_event(self.K, self.n, m, vc)
                alp, beta, gam = calc_coeff(a_m, a_p, b_m, b_p)
                si.cut_interval(alp, beta, gam)

        interval_oc = si.get_intervals()

        if homotopy:
            return interval_oc
        else:
            self.intervals = interval_oc

    def si_homotopy(self, z_min, z_max, epsilon=1e-5):
        z = z_min
        Z = [z]
        result_path = []
        while z < z_max:
            Xz = (self.a * z + self.b).reshape(self.d, self.n)
            dp = self.det_cls(Xz, self.K, self.L, self.width)
            cp, dim, cp_dict, dim_dict, sort_dict = dp.detection()

            intervals = self.si_oc(cp_dict, dim_dict, sort_dict, homotopy=True)

            if len(intervals) == 1:
                z = intervals[0][1]
            else:
                for interval in intervals:
                    if interval[0] < z < interval[1]:
                        z = interval[1]
                        break
            if z < z_max:
                Z.append(z)
            else:
                Z.append(z_max)

            result_path.append([cp, dim])
            z += epsilon

        z_interval = []
        for i, result in enumerate(result_path):
            cp_, dim_ = result
            if self.tau in cp_:
                index_ = cp_.index(self.tau)
                if self.theta in dim_[index_ - 1]:
                    z_interval.append([Z[i], Z[i + 1]])
        z_interval = union_all(z_interval)
        self.intervals = z_interval

    def test(self, homotopy=True, z_min=-10, z_max=10):
        if homotopy:
            self.si_homotopy(z_min, z_max)
        else:
            _, _, cp_dict, dim_dict, sort = self.result
            self.si_oc(cp_dict, dim_dict, sort)
        return self.si.test(intervals=self.intervals)

    def test_naive(self):
        naive = NaiveInferenceNorm(self.vecX, self.cov, self.eta)
        return naive.test()
