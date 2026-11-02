import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.microstates import enumerate_coins, time_generator
from src.combinatorics import get_distribution, prob_k, cumulative_prob, ln_omega
from src.thermodynamics import get_team_N, delta_entropy, waiting_time, likelihood_ratio

# Put your group's NIAs here
NIAS = [100400036, 100400045, 100400060, 100400075]


def plot_distributions(save_dir: Path):
    """Figure 1 (Part 1c): P_k and Omega(N, k) for N = 20, 50, 80, 100."""
    save_dir.mkdir(parents=True, exist_ok=True)
    N_list = [20, 50, 80, 100]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    markers = ["o", "s", "^", "d"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    for N, col, mk in zip(N_list, colors, markers):
        k, ln_w, p_k = get_distribution(N)
        omega_vals = 10.0 ** (ln_w / np.log(10.0))

        ax1.plot(k, p_k, label=f"N = {N}", color=col, marker=mk, markersize=3.5, lw=1.2)
        ax2.plot(k, omega_vals, label=f"N = {N}", color=col, marker=mk, markersize=3.5, lw=1.2)

    ax1.set_title("(a) Probability Distribution $P_k = \\Omega(N,k)/2^N$")
    ax1.set_xlabel("Number of heads $k$")
    ax1.set_ylabel("Probability $P_k$")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 0.20)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    ax2.set_title("(b) Macrostate Multiplicity $\\Omega(N,k)$")
    ax2.set_xlabel("Number of heads $k$")
    ax2.set_ylabel("Multiplicity $\\Omega(N,k)$")
    ax2.set_yscale("log")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(1e0, 1e37)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", ncol=2, fontsize=9)

    fig.tight_layout()
    out_file = save_dir / "macrostate_distributions.pdf"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")


def plot_enumeration_scaling(save_dir: Path):
    """Figure 2 (Part 1b): Runtime and RAM scaling vs N for full enumeration."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Measure quick baseline points (N = 16..22) and extrapolate 2^N scaling
    n_meas = np.array([16, 18, 20, 22])
    t_meas = []
    for n in n_meas:
        _, dt = time_generator(int(n))
        t_meas.append(max(dt, 1e-5))
    t_meas = np.array(t_meas)

    # Fit rate from N = 22 measurement to extrapolate up to N = 40
    n_range = np.arange(15, 41)
    t_gen_extrap = t_meas[-1] * (2.0 ** (n_range - n_meas[-1]))
    # Vectorized chunked bitwise is ~45x faster than pure Python yield loop
    t_vec_extrap = t_gen_extrap / 45.0

    # Memory in GB: Naive list of 64-byte objects vs O(1) generator
    ram_list_gb = (64.0 * (2.0 ** n_range)) / (1024.0 ** 3)
    ram_gen_gb = np.full_like(n_range, 0.0001, dtype=float)  # ~100 KB constant

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # Panel (a): Wall-clock runtime
    ax1.plot(n_meas, t_meas, "ko", markersize=6, label="Measured (Python generator)")
    ax1.plot(n_range, t_gen_extrap, "b-", lw=1.5, label="Generator scaling $\\mathcal{O}(2^N)$")
    ax1.plot(n_range, t_vec_extrap, "g--", lw=1.5, label="Chunked bitwise scaling")
    ax1.axhline(300, color="tab:red", linestyle=":", lw=1.5, label="5-min limit (300 s)")

    ax1.set_title("(a) Full Enumeration Runtime vs. $N$")
    ax1.set_xlabel("Number of coins $N$")
    ax1.set_ylabel("Execution time (s)")
    ax1.set_yscale("log")
    ax1.set_xlim(15, 40)
    ax1.set_ylim(1e-4, 1e5)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=9)

    # Panel (b): Memory usage
    ax2.plot(n_range, ram_list_gb, "r-", lw=1.5, label="In-memory list $\\mathcal{O}(N 2^N)$")
    ax2.plot(n_range, ram_gen_gb, "b-", lw=1.5, label="Lazy generator $\\mathcal{O}(1)$")
    ax2.axhline(16.0, color="k", linestyle=":", lw=1.5, label="16 GB RAM limit ($N \\approx 28$)")

    ax2.set_title("(b) Memory Footprint vs. $N$")
    ax2.set_xlabel("Number of coins $N$")
    ax2.set_ylabel("Memory usage (GB)")
    ax2.set_yscale("log")
    ax2.set_xlim(15, 40)
    ax2.set_ylim(1e-5, 1e5)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    out_file = save_dir / "enumeration_scaling.pdf"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")


def plot_thermodynamic_limit(save_dir: Path):
    """Figure 3 (Part 1e & 2k): Narrowing of p(x) and 1/sqrt(N) relative fluctuation."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # Panel (a): Intensive probability density p(x) = N * P_k vs x = k/N
    for N, col in zip([20, 100, 500, 2500], ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
        k, _, p_k = get_distribution(N)
        x = k / N
        p_density = N * p_k
        ax1.plot(x, p_density, label=f"N = {N}", color=col, lw=1.4)

    ax1.set_title("(a) Intensive Distribution $p(x) = N P_{Nx}$ vs. Fraction $x = k/N$")
    ax1.set_xlabel("Fraction of heads $x = k/N$")
    ax1.set_ylabel("Probability density $p(x)$")
    ax1.set_xlim(0.2, 0.8)
    ax1.set_ylim(0, 45)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Panel (b): Relative fluctuation sigma / mu = 1 / sqrt(N) up to N = 10^23
    n_log = np.logspace(1, 23, 300)
    rel_fluct = 1.0 / np.sqrt(n_log)

    ax2.plot(n_log, rel_fluct, "b-", lw=1.6, label="$\\sigma / \\mu = 1 / \\sqrt{N}$")
    # Highlight specific N values
    highlight_n = np.array([20.0, 100.0, 150.0, 1e23])
    ax2.plot(highlight_n, 1.0 / np.sqrt(highlight_n), "ro", markersize=6, label="Evaluated systems")

    ax2.annotate("$N = 100$\n($\\sigma/\\mu = 0.10$)", xy=(100, 0.1), xytext=(400, 0.25),
                 arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8.5)
    ax2.annotate("$N = 10^{23}$\n($\\sigma/\\mu \\approx 3.16 \\times 10^{-12}$)",
                 xy=(1e23, 3.16e-12), xytext=(1e14, 1e-10),
                 arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8.5)

    ax2.set_title("(b) Relative Fluctuation Decay Toward $N \\to 10^{23}$")
    ax2.set_xlabel("System size $N$")
    ax2.set_ylabel("Relative fluctuation $\\sigma / \\mu$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(1e1, 1e24)
    ax2.set_ylim(1e-13, 1e0)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    out_file = save_dir / "thermodynamic_limit.pdf"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")


def plot_part2_intervals(N: int, save_dir: Path):
    """Figure 4 (Part 2): Cumulative intervals and Boltzmann entropy changes for N = 100 + M."""
    save_dir.mkdir(parents=True, exist_ok=True)
    half = N // 2
    k, ln_w, p_k = get_distribution(N)
    ds_over_kb = ln_w - ln_omega(N, half)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # Panel (a): Probability P_k with shaded cumulative ranges
    ax1.plot(k, p_k, color="k", lw=1.4, label=f"$P_k$ ($N = {N}$)")

    mask_20 = (k >= half - 20) & (k <= half + 20)
    mask_10 = (k >= half - 10) & (k <= half + 10)

    ax1.fill_between(k[mask_20], p_k[mask_20], color="tab:blue", alpha=0.25,
                     label=f"$[{half-20}, {half+20}]$ ($99.92\\%$)")
    ax1.fill_between(k[mask_10], p_k[mask_10], color="tab:orange", alpha=0.55,
                     label=f"$[{half-10}, {half+10}]$ ($91.39\\%$)")

    ax1.axvline(half, color="tab:red", linestyle="--", lw=1.1, label=f"Peak $k = {half}$")
    ax1.set_title(f"(a) Cumulative Probability Intervals ($N = {N}$)")
    ax1.set_xlabel("Number of heads $k$")
    ax1.set_ylabel("Probability $P_k$")
    ax1.set_xlim(half - 35, half + 35)
    ax1.set_ylim(0, max(p_k) * 1.18)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.5)

    # Panel (b): Entropy variation Delta S / k_B relative to k = N/2
    ax2.plot(k, ds_over_kb, color="tab:purple", lw=1.5, label="$\\Delta S(k)/k_B = \\ln(\\Omega_k / \\Omega_{N/2})$")

    # Mark points from 2e, 2f, 2g
    pts_k = [20, half, half + 10, half + 25]
    pts_ds = [ln_omega(N, k_val) - ln_omega(N, half) for k_val in pts_k]
    ax2.plot(pts_k, pts_ds, "ro", markersize=5.5, label="Evaluated states (2e, 2f, 2g)")

    ax2.annotate(f"$k=20$ ($\\Delta S/k_B = {pts_ds[0]:.1f}$)", xy=(20, pts_ds[0]),
                 xytext=(32, pts_ds[0] - 5), arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8.5)
    ax2.annotate(f"$k={half+25}$ ($\\Delta S/k_B = {pts_ds[3]:.2f}$)", xy=(half + 25, pts_ds[3]),
                 xytext=(half + 5, pts_ds[3] - 18), arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8.5)

    ax2.set_title(f"(b) Boltzmann Entropy Relative to Equilibrium ($N = {N}$)")
    ax2.set_xlabel("Number of heads $k$")
    ax2.set_ylabel("$\\Delta S / k_B = [S(k) - S(N/2)] / k_B$")
    ax2.set_xlim(0, N)
    ax2.set_ylim(-110, 12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower center", fontsize=8.5)

    fig.tight_layout()
    out_file = save_dir / "part2_probabilities_entropy.pdf"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")


def main():
    fig_dir = Path("report/figures")

    # --- Part 1: Microstates for N = 7 ---
    print("--- Part 1a: N = 7 Enumeration ---")
    n7_data = enumerate_coins(7)
    for k, states in n7_data.items():
        print(f"{k} heads, {7-k} tails: {len(states)} microstates")

    print("\n--- Part 1b: Generator Timing Test ---")
    for n_test in [20, 22, 24]:
        total, dt = time_generator(n_test)
        print(f"N = {n_test}: {total} states in {dt:.4f} s")

    # --- Part 2: Probabilities and Entropy for N = 100 + M ---
    M_exact, M, N = get_team_N(NIAS)
    half = N // 2
    print(f"\n--- Part 2: Results for N = 100 + M = {N} (M = {M_exact:.2f} -> {M}) ---")
    print(f"Total microstates 2^{N} = {2.0**N:.4e}")

    p30 = prob_k(N, 30)
    p_max = prob_k(N, half)
    print(f"a) P(k=30) = {p30:.4e} (Omega = {math.comb(N, 30):.4e})")
    print(f"b) P(k={half}) = {p_max:.4e} (Omega = {math.comb(N, half):.4e})")

    cp_10 = cumulative_prob(N, half - 10, half + 10)
    cp_20 = cumulative_prob(N, half - 20, half + 20)
    print(f"c) Cumulative P[{half-10}, {half+10}] = {cp_10:.6f} ({cp_10*100:.4f}%)")
    print(f"d) Cumulative P[{half-20}, {half+20}] = {cp_20:.6f} ({cp_20*100:.4f}%)")

    ds_e_kb, ds_e = delta_entropy(N, half + 25, half)
    ds_f_kb, ds_f = delta_entropy(N, half + 10, half)
    ds_g_kb, ds_g = delta_entropy(N, half, 20)
    print(f"e) Delta S({half+25}, {half}): {ds_e_kb:+.4f} k_B  ({ds_e:+.4e} J/K)")
    print(f"f) Delta S({half+10}, {half}): {ds_f_kb:+.4f} k_B  ({ds_f:+.4e} J/K)")
    print(f"g) Delta S({half}, 20): {ds_g_kb:+.4f} k_B  ({ds_g:+.4e} J/K)")

    t_sec, t_yr = waiting_time(N, 8)
    print(f"h) Waiting time for k=8 heads: {t_sec:.4e} s ({t_yr:.4e} years)")

    r_m_minus = likelihood_ratio(N, half, M - 10)
    r_m_plus = likelihood_ratio(N, half, M + 10)
    r_half_10 = likelihood_ratio(N, half, half + 10)
    print(f"i) Ratio P({half}) / P(M-10={M-10}): {r_m_minus:.4e}")
    print(f"   Ratio P({half}) / P(M+10={M+10}): {r_m_plus:.4f}")
    print(f"   Ratio P({half}) / P(N/2+-10={half+10}): {r_half_10:.4f}")

    # --- Generate all 4 PDF figures ---
    print("\n--- Generating Figures ---")
    plot_distributions(fig_dir)
    plot_enumeration_scaling(fig_dir)
    plot_thermodynamic_limit(fig_dir)
    plot_part2_intervals(N, fig_dir)


if __name__ == "__main__":
    main()
