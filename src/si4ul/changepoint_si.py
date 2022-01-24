from si4ul.experiments import changepoint_si as exp


def optseg_si_oc(x, sigma=1, beta=1.5):
    """
    computing valid p-values for optimal changepoint by over-comditioning Selective Inference.

    Args:
        x (array-like of shape (point-num)): time series data.
        sigma (float): standard deviation in distribution of time series data.
        beta (float): regularization factor.
   
    Returns:
        (array-like of shape (changepoint-num), array-like of shape (changepoint-num)): changepoint list and p-values of each changepoint.
    
    Examples:
        >>> seg, p_list = changepoint_si.optseg_si(x)
        >>> print(seg)
        [25, 40, 59, 75, 80, 100, 125, 140, 159, 175, 180]
        >>> print(p_list)
        [0.013528694810211626, 0.05567852225768205, 0.329173206633249, 0.36014111520463954, 0.4101971284236369, 0.6981932657047393, 0.16996861755494708, 0.3719537702838958, 0.3592986205789473, 0.36014111520464676, 0.1196678816881915]

    """
    return exp.optseg_si_oc(x, sigma, beta)


def optseg_si(x, sigma=1, beta=1.5):
    """
    computing valid p-values for optimal changepoint by Selective Inference using Dynamic Programming.

    Args:
        x (array-like of shape (point-num)): time series data.
        sigma (float): standard deviation in distribution of time series data.
        beta (float): regularization factor.
    
    Returns:
        (array-like of shape (changepoint-num), array-like of shape (changepoint-num)): changepoint list and p-values of each changepoint.

    Examples:
        >>> seg, p_list = changepoint_si.optseg_si(x)
        >>> print(seg)
        [25, 40, 59, 80, 100, 125, 140, 159, 180]
        >>> print(p_list)
        [6.439051311970013e-06, 5.451462826100586e-07, 3.589542368255502e-07, 7.74132481356728e-05, 0.11719543493338598, 6.4387982662212394e-06, 1.1776732917248327e-06, 3.589542368255725e-07, 1.8180641069769048e-09]

    """
    return exp.optseg_si(x, sigma, beta)


def plot_changepoint_detection(x, sg_results, p_value_list, alpha, underlying=None, segment_size=0, title='OptSeg-SI'):
    """
    the result of optimal changepoint after SI.

    Args:
        x (array-like of shape (point-num)): time series data.
        sg_results (array-like of shape (changepoint-num)): changepoint list.
        p_value_list (array-like of shape (changepoint-num)): p-values of each changepoint.
        alpha (float): significance level.
        underlying (array-like of shape (true changepoint-num)): mean values of true segment.
        segment_size (float): true segment size.
        title (string): plot title.
    
    """
    if (underlying is None) ^ (segment_size == 0):
        raise AssertionError("If you want to plot underlying mechanism, you need input both of \"underlying\" and \"segmentation_size\"")

    return exp.plot_changepoint_detection(x, underlying, segment_size, alpha, sg_results, p_value_list, title)