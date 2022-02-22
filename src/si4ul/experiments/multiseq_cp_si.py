import numpy as np
from tqdm import tqdm

from si4ul.si import multiseq_cp_si as si
from si4ul.plot import multiseq_cp_si as plot
from si4ul.si.sicore.sicore.utils import is_int_or_float


def multiseq_cp_dc_si(X, K, L, Xi, Sigma, test, width, phi):
    if is_int_or_float(Xi):
        Xi = Xi * np.identity(X.shape[0])
    if is_int_or_float(Sigma):
        Sigma = Sigma * np.identity(X.shape[1])
    cov = np.kron(Xi, Sigma)
    cp_result = si.DP_DC(X, K, L, width, phi).detection()
    t, dim = cp_result[:2]
    cp_list = [[tau, theta] for i, tau in enumerate(t[1:-1]) for theta in dim[i]]
    result = []
    for tau, theta in tqdm(cp_list):
        dc_si = si.DoubleCusumSI(
            X, K, L, si.DP_DC, tau, theta, cov, cp_result, width, phi
        )
        if test == "hom":
            pvalue = dc_si.test()
        elif test == "oc":
            pvalue = dc_si.test(homotopy=False)
        elif test == "naive":
            pvalue = dc_si.test_naive()
        else:
            print(f"Not implement {test}")
            return None
        result.append([tau, theta, pvalue])
    return result


def multiseq_cp_scan_si(X, K, L, Xi, Sigma, test, width):
    if is_int_or_float(Xi):
        Xi = Xi * np.identity(X.shape[0])
    if is_int_or_float(Sigma):
        Sigma = Sigma * np.identity(X.shape[1])
    cov = np.kron(Xi, Sigma)
    cp_result = si.DP_Scan(X, K, L, width).detection()
    t, dim = cp_result[:2]
    cp_list = [[tau, theta] for i, tau in enumerate(t[1:-1]) for theta in dim[i]]
    result = []
    for tau, theta in tqdm(cp_list):
        scan_si = si.ScanSI(X, K, L, si.DP_Scan, tau, theta, cov, cp_result, width)
        if test == "hom":
            pvalue = scan_si.test()
        elif test == "oc":
            pvalue = scan_si.test(homotopy=False)
        elif test == "naive":
            pvalue = scan_si.test_naive()
        else:
            print(f"Not implement {test}")
            return None
        result.append([tau, theta, pvalue])
    return result


def plot_multiseq(X, true_cp, true_mean):
    plot.plot(X, true_cp=true_cp, true_mean=true_mean)


def plot_multiseq_si(X, result, alpha, true_cp, true_mean):
    for res in result:
        pvalue = res[2]
        if pvalue < (alpha / len(result)):
            res[2] = True
        else:
            res[2] = False
    plot.plot_si(X, result, true_cp=true_cp, true_mean=true_mean)
