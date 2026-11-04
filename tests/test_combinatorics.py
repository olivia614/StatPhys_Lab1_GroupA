import math
import numpy as np
from src.combinatorics import ln_omega, prob_k, get_distribution, cumulative_prob
from src.thermodynamics import get_team_N, delta_entropy, likelihood_ratio


def test_ln_omega_matches_exact_comb():
    # Compare log-gamma calculation against Python's exact math.comb
    # The log-gamma formula must agree with the log of the exact integer
    for N in [20, 50, 100, 150]:
        for k in [0, 1, N // 4, N // 2, N]:
            exact_ln = math.log(math.comb(N, k))
            calc_ln = ln_omega(N, k)
            assert math.isclose(calc_ln, exact_ln, rel_tol=1e-12)


def test_probabilities_sum_to_one():
    # Ensure that the probability distribution sums to 1 and is symmetric
    for N in [20, 50, 80, 100]:
        _, _, probs = get_distribution(N)
        assert math.isclose(np.sum(probs), 1.0, rel_tol=1e-12)
        # Check symmetry P_k == P_{N-k}
        assert np.allclose(probs, probs[::-1])


def test_cumulative_prob():
    # Test that summing over every macrostate is 1 and that a partial range matches teh exact integer sum / 2^N
    N = 150
    # Full range must sum to 1
    assert math.isclose(cumulative_prob(N, 0, N), 1.0, rel_tol=1e-12)

    # Check range [65, 85] against exact integer sum
    exact_p = sum(math.comb(N, k) for k in range(65, 86)) / (2**N)
    assert math.isclose(cumulative_prob(N, 65, 85), exact_p, rel_tol=1e-12)


def test_entropy_and_likelihood():
    # Check that entropy change and the likelihood ratio are consistent
    _, M, N = get_team_N([100400036, 100400045, 100400060, 100400075])
    assert M == 50
    assert N == 150

    ds_kb, _ = delta_entropy(N, 85, 75)
    ratio = likelihood_ratio(N, 75, 85)
    # Since Delta S / k_B = ln(Omega_85 / Omega_75), exp(-ds_kb) must equal Omega_75 / Omega_85
    assert math.isclose(math.exp(-ds_kb), ratio, rel_tol=1e-12)
    assert math.isclose(prob_k(N, 75) / prob_k(N, 85), ratio, rel_tol=1e-12)