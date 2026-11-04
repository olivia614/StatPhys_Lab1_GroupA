import numpy as np
from src.combinatorics import ln_omega

k_B = 1.380649e-23  # Boltzmann constant (J/K)
SEC_PER_YEAR = 365.25 * 24 * 3600  # Seconds in a Julian year


def get_team_N(nias: list[int]) -> tuple[float, int, int]:
    """
    Calculate M as the harmonic mean of the last two digits of the team's NIAs,
    and return (M_float, M_rounded, N = 100 + M).
    """
    digits = [n % 100 for n in nias]
    if any(d == 0 for d in digits):
        raise ValueError("Last two digits of NIA cannot be 00 for harmonic mean.")

    inv_sum = sum(1.0 / d for d in digits)
    M_val = len(digits) / inv_sum
    M_int = round(M_val)
    return M_val, M_int, 100 + M_int


def delta_entropy(N: int, k1: int, k2: int) -> tuple[float, float]:
    """
    Compute entropy change Delta S = S(k1) - S(k2) = k_B * ln(Omega(N, k1) / Omega(N, k2)).
    Returns (Delta S / k_B, Delta S in J/K).
    """
    ds_over_kb = float(ln_omega(N, k1) - ln_omega(N, k2))
    return ds_over_kb, ds_over_kb * k_B


def waiting_time(N: int, k: int, rate_trials_per_s: float = 1.0) -> tuple[float, float]:
    """
    Expected waiting time <t> = 1 / (rate * P_k) to observe k heads.
    At N coins tossed per second, 1 full trial of N coins takes 1 second.
    Returns (time in seconds, time in years).
    """
    ln_p = float(ln_omega(N, k)) - N * np.log(2.0)
    t_sec = float(np.exp(-ln_p)) / rate_trials_per_s
    t_years = t_sec / SEC_PER_YEAR
    return t_sec, t_years


def likelihood_ratio(N: int, k_ref: int, k_target: int) -> float:
    """
    Ratio of probabilities P(k_ref) / P(k_target) = Omega(N, k_ref) / Omega(N, k_target).
    """
    diff_ln = float(ln_omega(N, k_ref) - ln_omega(N, k_target))
    return float(np.exp(diff_ln))