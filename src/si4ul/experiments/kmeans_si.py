import numpy as np

from si4ul.si import kmeans_si as si
from si4ul.plot import kmeans_si as plot

def kmeans(X, k):
    k_means = si.KMeans(X, k)
    k_means.fit()
    return k_means


def pci_cluster(obs_model, comparison_clusters, sigma, max_iter, random_seed, z_max):
    comp_cluster = [comparison_clusters[0], comparison_clusters[1]]
    pci_cluster = si.Homotopy_PCI_cluster(obs_model, comp_cluster, max_iter=max_iter, seed=random_seed, var=sigma)
    pci_cluster.fit(z_max)

    pci_cluster.test(tail='right')
    hpci_p_value = pci_cluster.p_value
    pci_cluster.naive_test()
    naive_p_value = pci_cluster.p_value

    return pci_cluster.param_si.stat, hpci_p_value, naive_p_value


def pci_gene(obs_model, comparison_clusters, test_gene, sigma, max_iter, random_seed, z_max):
    comp_cluster = [comparison_clusters[0], comparison_clusters[1], test_gene]
    pci_gene = si.Homotopy_PCI_gene(obs_model, comp_cluster, max_iter=max_iter, seed=random_seed, var=sigma)
    pci_gene.fit(z_max)

    pci_gene.test(tail='right')
    hpci_p_value = pci_gene.p_value
    pci_gene.naive_test(popmean=0)
    naive_p_value = pci_gene.p_value

    return pci_gene.param_si.stat, hpci_p_value, naive_p_value


def all_clusters_combination_test(obs_model, test_gene, sigma):
    K = len(obs_model.label_num_list)
    if test_gene is None:
        print("pci_cluster")
    else:
        print("pci_gene for feature No." + str(test_gene))

    stat_matrix = np.full((K, K), np.nan)
    hpci_p_matrix = np.full((K, K), np.nan)
    naive_p_matrix = np.full((K, K), np.nan)
    for c_0 in range(K-1):
        for c_1 in range(c_0+1, K):
            comparison_clusters = [c_0, c_1]
            if test_gene == None:
                stat, hpci_p_value, naive_p_value = pci_cluster(obs_model, comparison_clusters, sigma, max_iter=1000, random_seed=0, z_max=20)
            else:
                stat, hpci_p_value, naive_p_value = pci_gene(obs_model, comparison_clusters, test_gene, sigma, max_iter=1000, random_seed=0, z_max=20)
            stat_matrix[c_0, c_1] = stat
            hpci_p_matrix[c_0, c_1] = hpci_p_value
            naive_p_matrix[c_0, c_1] = naive_p_value

    return stat_matrix, hpci_p_matrix, naive_p_matrix


def plot_histogram(obs_model, comparison_clusters, hist_gene, is_plot_norm):
    plot.histogram(obs_model, comparison_clusters, hist_gene, is_plot_norm)


def plot_scatter(obs_model, comparison_clusters, show_dims):
    plot.scatter(obs_model, comparison_clusters, show_dims)


def plot_violin(obs_model, gene_n):
    plot.violin(obs_model, gene_n)


def plot_p_matrix(matrix, digit=3, alpha=0.05):
    matrix = np.nan_to_num(matrix, nan=1)

    plot.p_matrix(matrix, digit, alpha)


def plot_statistics_matrix(matrix, digit=3):
    matrix = np.nan_to_num(matrix, nan=0)

    plot.statistics_matrix(matrix, digit)
