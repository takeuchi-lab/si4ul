from si4ul.experiments import multiseq_cp_si as exp


def multiseq_cp_dc_si(X, K, L, Xi=1, Sigma=1, test="hom", width=0, phi=0.5):
    """
    computing valid p-values for changepoint with Double-CUSUM
    by Selective Inference.

    Args:
        X (ndarray of shape (component, location)): multi-dimensional sequence.
        K (int): the number of change locations.
        L (int): minimum segment length.
        Xi (float or ndarray, optional): component's variance. Defaults to 1.
        Sigma (float or ndarray, optional): location's variance. Defaults to 1.
        test (str, optional): Set 'hom' for homotopy search, 'oc' for over-conditioning,
            'naive' for naive test. Defaults to 'homotopy'.
        width (int, optional): same cp width. Defaults to 0
        phi (float, optional): Double-CUSUM's parameter. Defaults to 0.5

    Returns:
        (array-like of shape (changepoint-num)):
            change-location, change-component and p-values of each changepoint.

    Examples:
        >>> result = multiseq_cp_dc_si(X, 1, 5)
        >>> print(result)
        [[10, 0, 1.1237052306791954e-12], [10, 1, 2.050601922239704e-09]]

    """
    return exp.multiseq_cp_dc_si(X, K, L, Xi, Sigma, test, width, phi)


def multiseq_cp_scan_si(X, K, L, Xi=1, Sigma=1, test="hom", width=0):
    """
    computing valid p-values for changepoint with scan statistic
    by Selective Inference.

    Args:
        X (ndarray of shape (component, location)): multi-dimensional sequence.
        K (int): the number of change locations.
        L (int): minimum segment length.
        Xi (float or ndarray, optional): component's variance. Defaults to 1.
        Sigma (float or ndarray, optional): location's variance. Defaults to 1.
        test (str, optional): Set 'hom' for homotopy search, 'oc' for over-conditioning,
            'naive' for naive test. Defaults to 'homotopy'.
        width (int, optional): same cp width. Defaults to 0

    Returns:
        (array-like of shape (changepoint-num)):
            change-location, change-component and p-values of each changepoint.

    Examples:
        >>> result = multiseq_cp_scan_si(X, 1, 5)
        >>> print(result)
        [[10, 0, 1.1237052306791954e-12], [10, 1, 2.050601922239704e-09]]

    """
    return exp.multiseq_cp_scan_si(X, K, L, Xi, Sigma, test, width)


def plot_multiseq(X, true_cp=None, true_mean=None):
    """
    Plot of multi-dimensional sequence.

    Args:
        X (ndarray of shape (component, location)): multi-dimensional sequence.
        true_cp (array-like of shape (components, true-cp-num+2)):
            the true change point position
        true_mean (array-like of shape (components, true-cp-num+1)):
            true population mean
    """
    return exp.plot_multiseq(X, true_cp, true_mean)


def plot_multiseq_si(X, result, alpha=0.05, true_cp=None, true_mean=None):
    """
    the result of multiseq changepoint after SI.

    Args:
        X (ndarray of shape (component, location)): multi-dimensional sequence.
        result (array-like of shape (changepoint-num)):
            change-location, change-component and p-values list
        alpha (float): significance level. Defaults to 0.05
        true_cp (array-like of shape (components, true-cp-num+2)):
            the true change point position
        true_mean (array-like of shape (components, true-cp-num+1)):
            true population mean
    """
    return exp.plot_multiseq_si(X, result, alpha, true_cp, true_mean)
