from si4ul.experiments import kmeans_si as exp


def kmeans(X, n_clusters):
    """
    k-means clustering algorithm.
    
    Args:
        X (array-like of shape (n, d)): data matrix.
        n_clusters (int): number of cluster.
    
    Returns:
        KMeans: reffer to document of KMeans.
    
    Examples:
        >>> print(kmeans_si.kmeans(X, K))
        <si4ul.si.kmeans_si.KMeans at 0x102aef700>

    """

    return exp.kmeans(X, n_clusters)


def pci_cluster(obs_model, comparison_clusters, sigma=1.0, max_iter=1000, random_seed=0, z_max=20):
    """
    post clustering inference for test between clusters.
    
    Args:
        obs_model (KMeans): reffer to document of KMeans.
        comparison_clusters (array-like of shape(2)): set of clusters to compare.
        sigma (float): standard deviation in distribution.
        max_iter (int): upper limit count of iteration in k-means algorithm.
        random_seed (int): seed of random for determine initial cluster.
        z_max (float): upper limit of parameter z on test statisitcs vector.
    
    Returns:
        (float, float, float): test statistics, homotopy PCI p-value and naive p-value.
    
    Examples:
        >>> print(kmeans_si.pci_cluster(obs_model, comparison_clusters))
        (5.245204424402314, 0.018632573868904267, 1.0612270479959528e-06)
    """

    if len(comparison_clusters) != 2:
        raise ValueError("length of comparison_clusters must be 2")
    return exp.pci_cluster(obs_model, comparison_clusters, sigma, max_iter, random_seed, z_max)


def pci_gene(obs_model, comparison_clusters, test_gene, sigma=1.0, max_iter=1000, random_seed=0, z_max=20):
    """
    post clustering inference for test between clusters about a feature.
    
    Args:
        obs_model (KMeans): reffer to document of KMeans.
        comparison_clusters (array-like of shape(2)): set of clusters to compare.
        sigma (float): standard deviation in distribution.
        test_gene (int): feature to compare.
        max_iter (int): upper limit count of iteration in k-means algorithm.
        random_seed (int): seed of random for determine initial cluster.
        z_max (float): upper limit of parameter z on test statisitcs vector.
    
    Returns:
        (float, float, float): test statistics, homotopy PCI p-value and naive p-value.
    
    Examples:
        >>> print(kmeans_si.pci_gene(obs_model, comparison_clusters, gene_id))
        (0.26352301450242993, 0.4212956294190716, 0.20005529456703786)
    """

    if len(comparison_clusters) != 2:
        raise ValueError("length of comparison_clusters must be 2")
    return exp.pci_gene(obs_model, comparison_clusters, test_gene, sigma, max_iter, random_seed, z_max)


def all_clusters_combination_test(obs_model, test_gene=None, sigma=1.0):
    """
    post clustering inference for all cluster combinations.
    If test_gene is set, test is PCI_gene.
    otherwise, test is PCI_cluster.
    
    Args:
        obs_model (KMeans): reffer to document of KMeans.
        test_gene (int): feature to compare.
        sigma (float): standard deviation in distribution.
    
    Returns:
        (array-like of shape(3, cluster_num, cluster_num)): matrix of test statistics, matrix of homotopy PCI p-value and matrix of naive p-value.
    
    Examples:
        >>> print(kmeans_si.all_clusters_combination_test(obs_model))
        pci_cluster
        (array([[nan, 5.24520442, 4.53785448], [nan, nan, 5.38552569], [nan, nan, nan]]), 
        array([[nan, 0.01863257, 0.00028306], [nan, nan, 0.00359755], [nan, nan, nan]]), 
        array([[nan, 1.06122705e-06, 3.37658148e-05], [nan, nan, 5.03368420e-07], [nan, nan, nan]]))
    """
    return exp.all_clusters_combination_test(obs_model, test_gene, sigma)


def plot_histogram(obs_model, comparison_clusters, test_gene, is_plot_norm=False):
    """
    plot histogram of distribution per cluster using test.
    
    Args:
        obs_model (KMeans): reffer to document of KMeans.
        comparison_clusters (array-like of shape(2)): set of clusters to compare.
        test_gene (int): feature to plot.
        is_plot_norm (bool): whether plot normal distribution in background.
    """
    if len(comparison_clusters) != 2:
        raise ValueError("length of comparison_clusters must be 2")

    exp.plot_histogram(obs_model, comparison_clusters, test_gene, is_plot_norm)


def plot_scatter(obs_model, comparison_clusters, show_dims):
    """
    plot scatter data in inputted 2-dims per cluster using test.
    
    Args:
        obs_model (KMeans): reffer to document of KMeans.
        comparison_clusters (array-like of shape(2)): set of clusters to compare.
        show_dims (array-like of shape(2)): set of dims to show.
    """
    if len(comparison_clusters) != 2:
        raise ValueError("length of comparison_clusters must be 2")
    if len(show_dims) != 2:
        raise ValueError("length of show_dims must be 2")

    exp.plot_scatter(obs_model, comparison_clusters, show_dims)


def plot_violin(obs_model, test_gene):
    """
    plot violin per cluster using test and other.
    
    Args:
        obs_model (KMeans): reffer to document of KMeans.
        test_gene (int): feature to plot.
    """
    exp.plot_violin(obs_model, test_gene)


def plot_p_matrix(matrix, digit=3, alpha=0.05):
    """
    plot matrix of p-value that is calculated by each cluster combinations.
    
    Args:
        matrix (array-like of shape(cluster_num, cluster_num)): matrix of p-value.
        digit (int): digit number to display.
        alpha (float): significant level.
    """
    if alpha < 0 or 1 < alpha:
        raise ValueError("alpha must be between 0 and 1")
    exp.plot_p_matrix(matrix, digit, alpha)


def plot_statistics_matrix(matrix, digit=3):
    """
    plot matrix of statistics that is calculated by each cluster combinations.
    
    Args:
        matrix (array-like of shape(cluster_num, cluster_num)): matrix of test statistics.
        digit (int): digit number to display.
    """
    exp.plot_statistics_matrix(matrix, digit)
