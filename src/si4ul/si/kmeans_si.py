import numpy as np

from .sicore.sicore import SelectivePCINormSE, NaivePCIChiSquared, NaivePCINorm, SelectivePCIChiSquaredSE
from .sicore.sicore.inference import norm as sicore
from .sicore.sicore.inference import chi_squared as chi_si


# KMeans algorithm
class KMeans:
    def __init__(self, X, n_clusters, max_iter = 1000, random_seed = 0):
        #ここで正規化する場合はHomotopy_PCI_gene内のkmeansにも適用させる必要あり
        # self.X = self.standardization(X)
        self.X = X
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_seed)
        self.random_seed = random_seed


    def fit(self):
        #初期クラスタをデータの中から選ぶ
        self.labels_ = np.full(self.X.shape[0], -1)
        #中心を固定
        np.random.seed(self.random_seed)
        select_center = np.random.choice(range(self.X.shape[0]), self.n_clusters, replace = False)
        give_cluster = 0
        for i in range(self.n_clusters):
            self.labels_[select_center[i]] = give_cluster
            give_cluster += 1
        
        labels_prev = np.zeros(self.X.shape[0])
        #更新回数
        self.count = 0
        #クラスタの中心を保存する
        self.cluster_centers_ = np.zeros((self.n_clusters, self.X.shape[1])) 
        #クラスタに含まれるデータ数のリスト
        self.label_num_list = []
        
            
        #各データポイントが属しているクラスターが変化しなくなった、又は一定回数の繰り返しを越した場合は終了
        while (not (self.labels_ == labels_prev).all() and self.count < self.max_iter):
            #cluster_centers_の初期化
            self.cluster_centers_ = np.zeros((self.n_clusters, self.X.shape[1])) 
            #その時点での各クラスターの重心を計算する
            self.label_num_list=[]
            for i in range(self.n_clusters):
                XX = self.X[self.labels_ == i, :]
                self.cluster_centers_[i, :] = XX.mean(axis = 0)
                self.label_num_list.append(XX.shape[0])
            
            #各データポイントと各クラスターの重心間の距離を総当たりで計算する
            dist = ((self.X[:, :, np.newaxis] - self.cluster_centers_.T[np.newaxis, :, :])**2) .sum(axis = 1)
            #1つ前のクラスターラベルを覚えておく。1つ前のラベルとラベルが変化しなければプログラムは終了する。
            labels_prev = self.labels_
            #再計算した結果、最も距離の近いクラスターのラベルを割り振る
            self.labels_ = dist.argmin(axis = 1)
               
            self.count += 1
        
        self.label_num_list=[]
        for i in range(self.n_clusters):
            XX = self.X[self.labels_ == i, :]
            self.label_num_list.append(XX.shape[0])


    def standardization(self, X):
        x_mean = X.mean(axis=0, keepdims=True)
        x_std = X.std(axis=0, keepdims=True, ddof=1)
        return (X - x_mean) / x_std


#eta
def make_eta(X, labels, cluster_num, comp):
    vec_x = X.T.flatten()
    #eta = np.zeros((X.shape[0], X.shape[1]))
    eta = np.zeros((X.shape[0], 1))
    
    for i in range(X.shape[0]):
        if labels[i] == comp[0]:
                eta[i] = 1/cluster_num[comp[0]]
        elif labels[i] == comp[1]:
                eta[i] = -1/cluster_num[comp[1]]
        
    return eta


#tau_sign
def sign(X, eta, comp):
    vec_x = X.T.flatten()
    e_j = np.zeros((X.shape[1], 1))
    e_j[comp[2], 0] = 1
    ej_eta = np.kron(e_j, eta)
    tau = np.dot(ej_eta.T, vec_x)
    tau_sign = 1
    if 0 > tau:
        tau_sign = -1
        
    return tau_sign


# make interval cluster
def make_interval_cluster(X, cluster_num, labels, labels_prev, label_num, cluster, eta, sigma_hat, m_a_m_b, param_si_z):
    A_list = [[]]
    
    #データがどのクラスに属するかを表現する行列
    one_c = np.zeros((cluster_num, X.shape[0]))
    for i in range(one_c.shape[1]):
        if labels_prev[i] != -1:
            one_c[labels_prev[i],i] = 1
        
    #Aの計算に必要な要素のリストを作成（必要な時に計算する：行列が膨大なため）
    for i in range(X.shape[0]):
        one_x = np.zeros(X.shape[0])
        one_x[i] = 1
        
        a_list = []
        for j in range(cluster_num):
            if labels[i] != j:
                k = np.dot(one_x, X) - cluster[labels[i],:]
                h = np.dot(one_x, X) - cluster[j,:]
                lamda = np.dot(k, k) - np.dot(h, h)
                
                c_k = label_num[labels[i]]
                c_h = label_num[j]
                m_k_one = one_c[labels[i], :]
                m_h_one = one_c[j, :]
                n_i_onehot = one_x
                            
                #alpha,kappa,lamda
                #cov_in_e = np.dot(i_matrix_n, e_onehot)
                C_k = np.dot(eta.T, m_k_one)/c_k
                C_h = np.dot(eta.T, m_h_one)/c_h
                e_eta = np.dot(n_i_onehot.T, eta)
                            
                            
                alp = sigma_hat*((C_k - e_eta)**2 - (C_h - e_eta)**2)/np.dot(eta.T, eta)
                alpha = alp[0][0]
                            
                m_k = cluster[labels[i],:]
                m_h = cluster[j,:]
                x_i = np.dot(n_i_onehot.T, X)
                kappa_a = (C_k - e_eta)*m_k - (C_h - e_eta)*m_h - (C_k - C_h)*x_i
                kappa_b = np.dot(kappa_a.T, m_a_m_b)
                C_ab = np.linalg.norm(m_a_m_b, ord=2)*np.linalg.norm(eta, ord=2)
                            
                kappa = 2*kappa_b/C_ab            
            
                #区間導出
                param_si_z.cut_interval(alpha, kappa, lamda, tau=True)


# make interval gene
def make_interval_gene(X, cluster_num, labels, labels_prev, label_num, SI_original):
    I_d = np.eye(X.shape[1])

    # データがどのクラスに属するかを表現する行列
    one_c = np.zeros((cluster_num, X.shape[0], 1))
    for i in range(one_c.shape[1]):
        if labels_prev[i] != -1:
            one_c[labels_prev[i], i] = 1

    # Aの計算に必要な要素のリストを作成（必要な時に計算する：行列が膨大なため）
    for i in range(X.shape[0]):
        one_x = np.zeros((X.shape[0], 1))
        one_x[i, 0] = 1
        k = labels[i]

        # I_d_one_ck = np.kron(I_d, one_c[k])/label_num[k]
        # m_k = np.dot(I_d_one_ck.T, X.T.flatten())
        # print(m_k)

        for h in range(cluster_num):
            # 自分のクラスでなければ作成
            if k != h:
                one_ck_one_ck_T = np.dot(one_c[k], one_c[k].T)
                one_ch_one_ch_T = np.dot(one_c[h], one_c[h].T)
                one_ck_one_x_T = np.dot(one_c[k], one_x.T)
                one_ch_one_x_T = np.dot(one_c[h], one_x.T)

                a_hik = one_ck_one_ck_T/label_num[k]**2 - one_ch_one_ch_T/label_num[h]**2 - 2*one_ck_one_x_T/label_num[k] + 2*one_ch_one_x_T/label_num[h]

                A_hik = np.kron(I_d, a_hik)

                # 区間導出
                SI_original.add_selection_event(A=A_hik)


#Jaccard(d=2)
def jaccard_2dimension(labels, comp):
    j1 = 0
    j2 = 0
    len_labels = len(labels)
    div_len = int(len_labels/2)
    for i in range(div_len):
        if labels[i] == comp[0]:
            j1 += 1
        elif labels[i] == comp[1]:
            j2 +=1
            
        if labels[i+div_len] == comp[1]:
            j1 += 1
        elif labels[i+div_len] == comp[0]:
            j2 += 1
            
    return max(j1/len_labels, j2/len_labels)


# SI for PCI_gene
class Homotopy_PCI_gene:
    def __init__(self, obs_model, comp_cluster, max_iter, seed=0, Var=1):
        self.max_iter = max_iter
        self.seed = seed
        self.obs_model = obs_model

        self.X = obs_model.X
        self.vec_x = self.X.T.flatten()
        self.eta = make_eta(self.X, obs_model.labels_, obs_model.label_num_list, comp_cluster)
        self.tau_sign = sign(self.X, self.eta, comp_cluster)
        self.comp_cluster = comp_cluster
        self.Var = 1
        self.n_clusters = obs_model.n_clusters

        #δとηを計算
        self.e_onehot = np.zeros((self.X.shape[1], 1))
        self.e_onehot[comp_cluster[2], 0] = 1
        self.delta = self.tau_sign*self.e_onehot
        self.delta_eta = np.kron(self.delta, self.eta).reshape(len(self.vec_x))

        self.param_si = sicore.SelectiveInferenceNormSE(self.vec_x, self.Var, self.delta_eta, init_lower=0)

        self.intervals = []
        self.active_set = []
        self.p_value = 0

    def serch_interval(self, X):

        vec_x = X.T.flatten()
        # 共分散行列を生成
        SI_original = sicore.SelectiveInferenceNormSE(vec_x, self.Var, self.delta_eta, init_lower=0)

        #区間を計算############################
        # 初期クラスタをデータの中から選ぶ
        labels_ = np.full(X.shape[0], -1)
        # 中心を固定
        np.random.seed(self.seed)
        select_center = np.random.choice(
            range(X.shape[0]), self.n_clusters, replace=False)
        give_cluster = 0
        for i in range(self.n_clusters):
            labels_[select_center[i]] = give_cluster
            give_cluster += 1

        labels_prev = np.zeros(X.shape[0])
        # 更新回数
        count = 0
        # クラスタの中心を保存する
        cluster_centers_ = np.zeros((self.n_clusters, X.shape[1]))
        # クラスタに含まれるデータ数のリスト
        label_num_list = []

        # 各データポイントが属しているクラスターが変化しなくなった、又は一定回数の繰り返しを越した場合は終了
        while (not (labels_ == labels_prev).all() and count < self.max_iter):
            # cluster_centers_の初期化
            cluster_centers_ = np.zeros((self.n_clusters, X.shape[1]))
            # その時点での各クラスターの重心を計算する
            label_num_list = []
            for i in range(self.n_clusters):
                XX = X[labels_ == i, :]
                cluster_centers_[i, :] = XX.mean(axis=0)
                label_num_list.append(XX.shape[0])

            # 各データポイントと各クラスターの重心間の距離を総当たりで計算する
            dist = ((X[:, :, np.newaxis] -
                     cluster_centers_.T[np.newaxis, :, :])**2) .sum(axis=1)
            # 1つ前のクラスターラベルを覚えておく。1つ前のラベルとラベルが変化しなければプログラムは終了する。
            labels_prev = labels_
            # 再計算した結果、最も距離の近いクラスターのラベルを割り振る
            labels_ = dist.argmin(axis=1)

            # 区間計算
            make_interval_gene(X, self.n_clusters, labels_, labels_prev, label_num_list, SI_original)
            count += 1

        return SI_original.get_intervals(), labels_

    def fit(self, z_max):
        self.intervals = []
        self.active_set = []
        z = 1e-4
        a = self.param_si.z
        b = self.param_si.c

        while z < z_max:
            vec_x_z = a + b*z
            X_z = vec_x_z.reshape(self.X.shape[1], self.X.shape[0]).T

            #serch interval from X_z
            s_interval, labels_ = self.serch_interval(X_z)

            
            for i in range(len(s_interval)):
                if s_interval[i][0] < z < s_interval[i][1]:
                    interval = s_interval[i]
 
            self.active_set.append(labels_)
            self.intervals.append(interval)
           
            #next z
            z = interval[1] + 1e-4 


    def oc_fit(self):
        self.intervals = []
        self.active_set = []
        s_interval, labels_ = self.serch_interval(self.X)

        for i in range(len(s_interval)):
            self.active_set.append(labels_)
            
        self.intervals = s_interval

    def naive_test(self, popmean=0):
        naive = NaivePCINorm(self.X, self.Var, self.eta, self.delta)
        self.p_value = naive.test(tail='right', popmean=popmean)


    def test(self, tail='double', popmean=0, dps='auto'):
        active_intervals = []
        i = 0
        for i in range(len(self.intervals)):
            if self.active_set[i].tolist() == self.obs_model.labels_.tolist():
                active_intervals.append(self.intervals[i])
    
        self.active_n = len(active_intervals)
        self.p_value = self.param_si.test(intervals=active_intervals, tail=tail, popmean=popmean, dps=dps)


class Homotopy_PCI_cluster:
    def __init__(self, obs_model, comp_cluster, max_iter = 1000, seed = 0, Var = 1):
        self.X = obs_model.X
        self.vec_x = self.X.T.flatten()
        self.comp_cluster = comp_cluster
        self.n_clusters = obs_model.n_clusters
        self.obs_model = obs_model
        self.eta = make_eta(self.X, obs_model.labels_, obs_model.label_num_list, comp_cluster)
        self.max_iter = max_iter
        self.seed = seed
        self.Var = Var
        self.gamma = np.kron(np.eye(self.X.shape[1]), self.eta)
        
        self.param_si = SelectivePCIChiSquaredSE(self.X, self.Var, self.gamma[:,0], 0, init_lower = 0)
        self.sigma_hat = self.param_si.make_sigma_hat(self.X, self.eta)

        # self.param_si = chi_si.SelectiveInferenceChiSquaredSE(
        #     self.vec_x, 1, self.eta, degree=self.X.shape[1], init_lower=0
        # )
        # self.sigma_hat = self.param_si.sigma_hat

        self.intervals = []
        self.active_set = []
        self.p_value = 0
         
    def serch_interval(self, X):
        vec_x = X.T.flatten()

        param_si_z = SelectivePCIChiSquaredSE(X, self.Var, self.gamma[:,0], 0, init_lower = 0)
        #sigma_hat
        sigma_hat = param_si_z.make_sigma_hat(X, self.eta)
        # param_si_z = chi_si.SelectiveInferenceChiSquaredSE(
        #     vec_x, 1, self.eta, degree=self.X.shape[1], init_lower=0
        # )     
        #m_a - m_b
        m_a_m_b = np.dot(self.gamma.T, vec_x)
        
        #区間を計算############################################3
        #初期クラスタをデータの中から選ぶ
        labels_ = np.full(X.shape[0], -1)
        #中心を固定
        np.random.seed(self.seed)
        select_center = np.random.choice(range(X.shape[0]), self.n_clusters, replace = False)
        give_cluster = 0
        for i in range(self.n_clusters):
            labels_[select_center[i]] = give_cluster
            give_cluster += 1
        
        labels_prev = np.zeros(X.shape[0])
        #更新回数
        count = 0
        #クラスタの中心を保存する
        cluster_centers_ = np.zeros((self.n_clusters, X.shape[1])) 
        #クラスタに含まれるデータ数のリスト
        label_num_list = []
        
            
        #各データポイントが属しているクラスターが変化しなくなった、又は一定回数の繰り返しを越した場合は終了
        while (not (labels_ == labels_prev).all() and count < self.max_iter):
            #cluster_centers_の初期化
            cluster_centers_ = np.zeros((self.n_clusters, X.shape[1])) 
            #その時点での各クラスターの重心を計算する
            label_num_list=[]
            for i in range(self.n_clusters):
                XX = X[labels_ == i, :]
                cluster_centers_[i, :] = XX.mean(axis = 0)
                label_num_list.append(XX.shape[0])
            
            #各データポイントと各クラスターの重心間の距離を総当たりで計算する
            dist = ((X[:, :, np.newaxis] - cluster_centers_.T[np.newaxis, :, :])**2) .sum(axis = 1)
            #1つ前のクラスターラベルを覚えておく。1つ前のラベルとラベルが変化しなければプログラムは終了する。
            labels_prev = labels_
            #再計算した結果、最も距離の近いクラスターのラベルを割り振る
            labels_ = dist.argmin(axis = 1)
               
            #区間計算
            make_interval_cluster(X, self.n_clusters, labels_, labels_prev, label_num_list, cluster_centers_, self.eta, self.sigma_hat, m_a_m_b, param_si_z)
            count += 1  
        return param_si_z.get_intervals(), labels_

    def fit(self, z_max):
        """
        homotopy si
        """
        self.intervals = []
        self.active_set = []

        z = 1e-4
        a = self.param_si.z
        b = self.param_si.c
        labels_prev = np.zeros(self.X.shape[0])

        while z<z_max:
            vec_x_z = a + b * self.sigma_hat * z
            X_z = vec_x_z.reshape(self.X.shape[1], self.X.shape[0]).T
            
            s_interval, labels_ = self.serch_interval(X_z)

            interval = []
            for i in range(len(s_interval)):
                if s_interval[i][0] < z < s_interval[i][1]:
                    interval = s_interval[i]

            self.active_set.append(labels_)
            self.intervals.append(interval)

            labels_prev = labels_
            z = interval[1] + 1e-4


    def oc_fit(self):
        """
        over conditioning pci
        """
        self.intervals = []
        self.active_set = []
        
        s_interval, labels_ = self.serch_interval(self.X)

        for i in range(len(s_interval)):
            self.active_set.append(labels_)
            
        self.intervals = s_interval

    def naive_test(self):
        # naive = NaivePCIChiSquared(self.X, 1, self.eta, 0)
        # self.p_value = naive.test(self.X, tail='right')
        naive = chi_si.NaiveInferenceChiSquared(self.vec_x, 1, self.eta, degree=self.X.shape[1])
        self.p_value = naive.test(tail='right')

    def test(self, tail='double', dps='auto'):
        active_intervals = []
        i = 0
        for i in range(len(self.intervals)):
            if self.active_set[i].tolist() == self.obs_model.labels_.tolist():
                active_intervals.append(self.intervals[i])

        self.active_n = len(active_intervals)
        self.p_value = self.param_si.test(active_intervals, tail=tail, dps=dps)
