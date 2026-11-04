import numpy as np
from scipy.special import gammaln, logsumexp


def ln_omega(N: int, k: int | np.ndarray) -> float | np.ndarray:
    """
    Calculate ln(Omega(N, k)) in log-space using the log-gamma function:
    ln(N!) - ln(k!) - ln((N-k)!) = gammaln(N+1) - gammaln(k+1) - gammaln(N-k+1).
    """
    k_arr = np.asarray(k)
    if np.any(k_arr < 0) or np.any(k_arr > N):
        raise ValueError(f"k must be between 0 and {N}")

    res = gammaln(N + 1) - gammaln(k_arr + 1) - gammaln(N - k_arr + 1)
    if np.ndim(k) == 0:
        return float(res)
    return res


def prob_k(N: int, k: int | np.ndarray) -> float | np.ndarray:
    """
    Calculate macrostate probability P_k = Omega(N, k) / 2^N
    using log-space subtraction: exp(ln_omega(N, k) - N * ln(2)).
    """
    ln_p = ln_omega(N, k) - N * np.log(2.0)
    return np.exp(ln_p)


def get_distribution(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return arrays of k, ln(Omega(N, k)), and P_k for all k = 0, ..., N.
    """
    k_vals = np.arange(0, N + 1)
    ln_w = ln_omega(N, k_vals)
    p_vals = np.exp(ln_w - N * np.log(2.0))
    return k_vals, ln_w, p_vals


def cumulative_prob(N: int, k1: int, k2: int) -> float:
    """
    Calculate the cumulative probability sum_{k=k1}^{k2} P_k.
    Uses logsumexp to avoid underflow when summing very small probabilities.
    """
    if k1 > k2:
        k1, k2 = k2, k1
    k_range = np.arange(k1, k2 + 1)
    ln_p = ln_omega(N, k_range) - N * np.log(2.0)
    return float(np.exp(logsumexp(ln_p)))