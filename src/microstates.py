import time
import numpy as np


def iter_microstates(N: int):
    """
    Generator that yields all 2^N microstates as integers from 0 to (1 << N) - 1.
    Uses O(1) memory since states are generated on the fly.
    """
    total_states = 1 << N
    for state in range(total_states):
        yield state


def state_to_str(state: int, N: int) -> str:
    """
    Convert an integer bitmask into a string of 'H' (1) and 'T' (0) of length N.
    Reads from the most significant bit down to bit 0.
    """
    chars = []
    for bit in range(N - 1, -1, -1):
        if (state >> bit) & 1:
            chars.append("H")
        else:
            chars.append("T")
    return "".join(chars)


def enumerate_coins(N: int) -> dict[int, list[str]]:
    """
    Enumerate all 2^N outcomes for N coins and group them by the number of heads k.
    Ordered from k = N down to k = 0 to match Table 1 in the lab guide.
    """
    macrostates = {k: [] for k in range(N, -1, -1)}
    mask = (1 << N) - 1

    for state in iter_microstates(N):
        # Flip bits with XOR so HHHHHHH (all 1s) comes first
        flipped = state ^ mask
        k = flipped.bit_count()
        macrostates[k].append(state_to_str(flipped, N))

    return macrostates


def fast_bitwise_counts(N: int, chunk_bits: int = 24) -> np.ndarray:
    """
    Count microstates per macrostate k by streaming blocks of integers
    and summing bits via bitwise right-shifts. Avoids RAM overflow for large N.
    """
    counts = np.zeros(N + 1, dtype=np.int64)
    total_states = 1 << N
    step = min(total_states, 1 << chunk_bits)

    for start in range(0, total_states, step):
        stop = min(start + step, total_states)
        chunk = np.arange(start, stop, dtype=np.uint64)

        # Count 1-bits (heads) for each number in the chunk using bit shifting
        heads = np.zeros(stop - start, dtype=np.int64)
        for bit in range(N):
            heads += (chunk >> bit) & 1

        counts += np.bincount(heads, minlength=N + 1)

    return counts


def time_generator(N: int) -> tuple[int, float]:
    """Measure the wall-clock time (in seconds) needed to iterate through 2^N states."""
    t0 = time.perf_counter()
    count = 0
    for _ in iter_microstates(N):
        count += 1
    dt = time.perf_counter() - t0
    return count, dt