import math
import numpy as np
from src.microstates import iter_microstates, enumerate_coins, fast_bitwise_counts


def test_iter_microstates_count():
    # Check that the generator yields exactly 2^N items
    for N in [3, 5, 7, 10]:
        states = list(iter_microstates(N))
        assert len(states) == (1 << N)


def test_enumerate_n7():
    # Verify N = 7 table has 128 states and matches binomial coefficients
    table = enumerate_coins(7)
    total_outcomes = sum(len(outcomes) for outcomes in table.values())
    assert total_outcomes == 128

    for k, outcomes in table.items():
        assert len(outcomes) == math.comb(7, k)
        for s in outcomes:
            assert s.count("H") == k
            assert s.count("T") == 7 - k


def test_fast_bitwise_counts():
    counts = fast_bitwise_counts(12, chunk_bits=8)
    expected = np.array([math.comb(12, k) for k in range(13)])
    assert np.array_equal(counts, expected)