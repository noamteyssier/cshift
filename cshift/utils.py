import numpy as np
from scipy.stats import chi2_contingency

def chisquare_test(ref: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """
    Multivariate chi-square test for each categorical variable abundance.

    Parameters
    ----------
    ref : np.ndarray
        Reference abundance of each categorical variable.
    obs : np.ndarray
        Observed abundance of each categorical variable.
    """ 
    x_max = ref.sum()
    y_max = obs.sum()
    pvalues = np.zeros(ref.size)

    for i in np.arange(pvalues.size):

        if ref[i] == 0 and obs[i] == 0:
            pvalues[i] = 1.
            continue

        m = np.array([
            [ref[i], x_max - ref[i]],
            [obs[i], y_max - obs[i]]
        ])

        _, pvalues[i], _, _ = chi2_contingency(m)

    return pvalues

def aggregate_chisquare_test(ref: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """
    Aggregate chi-square test for all categorical variables using multiple
    reference distributions. Aggregated using a geometric mean on each
    categorical p-value.

    Parameters
    ----------
    ref : np.ndarray
        Reference abundance of each categorical variable (2D matrix).
    obs : np.ndarray
        Observed abundance of each categorical variable (1D array).
    """ 
    pvalues = np.stack([
        chisquare_test(ref[i], obs) for i in np.arange(ref.shape[0])
    ])
    return np.exp(np.log(pvalues).sum(axis=0))

def percent_change(
        ref: np.ndarray,
        obs: np.ndarray) -> np.ndarray:
    """
    calculates the percent change between a reference group
    and a test group. Will first normalize the vectors so that
    their total will sum to 1
    """
    assert ref.size == obs.size
    r_norm = ref / ref.sum()
    t_norm = obs / obs.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        pcc = (t_norm - r_norm) / r_norm
    pcc[np.isinf(pcc)] = 1.
    pcc[np.isnan(pcc)] = 0.
    return pcc


def false_discovery_rate(
        pval: np.ndarray) -> np.ndarray:
    """
    converts the pvalues into false discovery rate q-values
    """
    dim = pval.shape
    qval = p_adjust_bh(pval.ravel())
    return qval.reshape(dim)


def p_adjust_bh(
        p: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg p-value correction for multiple hypothesis testing.
    https://stackoverflow.com/a/33532498
    """
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

