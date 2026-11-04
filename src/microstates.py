import time
import numpy as np
import tracemalloc

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
    one = np.uint64(1)

    for start in range(0, total_states, step):
        stop = min(start + step, total_states)
        chunk = np.arange(start, stop, dtype=np.uint64)

        # Count 1-bits (heads) for each number in the chunk using bit shifting.
        # Everything stays uint64: mixing uint64 with a plain Python int or with
        # int64 makes NumPy promote to float64, which cannot be added into int64.
        heads = np.zeros(stop - start, dtype=np.uint64)
        for bit in range(N):
            heads += (chunk >> np.uint64(bit)) & one

        # bincount needs integer indices of type intp, so convert once per block.
        counts += np.bincount(heads.astype(np.intp), minlength=N + 1)

    return counts


def time_generator(N: int) -> tuple[int, float]:
    """Measure the wall-clock time (in seconds) needed to iterate through 2^N states."""
    t0 = time.perf_counter()
    count = 0
    for _ in iter_microstates(N):
        count += 1
    dt = time.perf_counter() - t0
    return count, dt


def histogram_generator(N: int) -> np.ndarray:
    """Full enumeration with the generator, tallying heads per microstate.

    Returns an array counts[k] = Omega(N, k). Memory is O(1) apart from the
    N + 1 counters, because only one state exists at a time."""
    counts = [0] * (N + 1) # one count per macrostate k = 0...N
    for s in iter_microstates(N): # takes each state from generator
        counts[s.bit_count()] += 1 # counts every head
    return np.array(counts, dtype=np.int64)


# Lookup table: _POP16[i] = number of 1-bits (heads) in the 16-bit integer i.
# Built once at import time (2**16 = 65,536 entries). 
# Lets the vectorised enumeration count heads in 16-bit pieces by table lookup instead of bit by bit.
_POP16 = np.array([i.bit_count() for i in range(1 << 16)], dtype=np.uint8) # uint8 since max is 16

def histogram_vectorised(N: int, chunk_bits: int = 22) -> np.ndarray:
    """Full enumeration in blocks, using a 16-bit popcount lookup table (N <= 63).
    
    Heads are counted in 16-bit pieces with the _POP16 lookup table.
    Requires N <= 63 because states are stored as uint64."""
    counts = np.zeros(N + 1, dtype=np.int64) # histogram, counts[k] = Omega(N, k)
    total = 1 << N # total number of microstates, 2**N
    step = min(total, 1 << chunk_bits) # block size; a block of 2**22 states fits in RAM
    base = np.arange(step, dtype=np.uint64) # 0, 1, ..., step-1, reused for every block
    mask = np.uint64(0xFFFF) # 16 one-bits: keeps only the lowest 16 bits
    for start in range(0, total, step):
        chunk = base + np.uint64(start) # the actual state numbers of this block

        # Heads in the lowest 16 coins of every state in the block, via table lookup.
        # astype(np.intp) is needed because NumPy indexes arrays with intp integers.
        heads = _POP16[(chunk & mask).astype(np.intp)].astype(np.int64)

        # For N > 16: shift the next 16 bits down to the bottom, isolate them with
        # the mask, look them up and add, a state's heads = sum over its 16-bit pieces.
        for shift in range(16, N, 16):
            heads += _POP16[((chunk >> np.uint64(shift)) & mask).astype(np.intp)]

        # Count how many states in this block have 0, 1, ..., N heads
        counts += np.bincount(heads, minlength=N + 1)
    return counts


def benchmark(func, N: int, repeats: int = 3) -> tuple[float, float]:
    """Return (mean, sample standard deviation) of wall-clock time in seconds
    over `repeats` runs of func(N)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()  # perf_counter: high-resolution wall-clock timer
        func(N)
        times.append(time.perf_counter() - t0)
    # ddof=1 gives the sample s.d. (divide by n - 1), appropriate for few repeats
    return float(np.mean(times)), float(np.std(times, ddof=1))


def peak_memory_mb(func, N: int) -> float:
    """Peak Python/NumPy memory allocation (MB) while running func(N).

    Measured with tracemalloc, so it excludes memory used outside Python's
    allocator and is slowed down by tracing: use a small N.
    """
    tracemalloc.start()
    func(N)
    _, peak = tracemalloc.get_traced_memory()  # returns (current, peak) in bytes
    tracemalloc.stop()
    return peak / 1e6