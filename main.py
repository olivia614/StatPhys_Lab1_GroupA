import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.microstates import enumerate_coins, time_generator
from src.combinatorics import get_distribution, prob_k, cumulative_prob, ln_omega
from src.thermodynamics import get_team_N, delta_entropy, waiting_time, likelihood_ratio

NIAS = [100496704, 100599060, 100523129, 100534604]

def write_n7_summary_table(path: Path = Path("report/tables/n7_summary.tex")) -> None:
    """Table 1 (Part 1a): Write the N = 7 summary rows (k, tails, example microstate, Omega, P_k) like the example."""
    path.parent.mkdir(parents=True, exist_ok=True)
    N = 7
    data = enumerate_coins(N)  # ordered from k = 7 down to k = 0
    lines = []
    for k, states in data.items():
        omega = len(states)
        assert omega == math.comb(N, k)  # sanity check against the exact binomial
        example = states[0]
        lines.append(
            f"{k} & {N - k} & \\texttt{{{example}}} & {omega} & "
            f"${omega}/{2**N} = {omega / 2**N:.4f}$ \\\\"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")

def write_n7_full_table(path: Path = Path("report/tables/n7_full.tex")) -> None:
    """Table 1 (Part 1a - Appendix C): Write all 128 microstates for N = 7."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = enumerate_coins(7)  # ordered from k = 7 down to k = 0
    lines = []
    for k, states in data.items():
        label = f"{k}H\\,{7 - k}T ({len(states)})"
        strings = ", ".join(states)
        lines.append(f"{label} & \\texttt{{{strings}}} \\\\")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")

def plot_enumeration_scaling(save_dir: Path) -> None:
    """Figure 2 (Part 1b): Measured enumeration times vs N with 2^N fits and the 5-min limit

    The numbers are the measured mean +/- sample standard deviation of 3 runs (single direct runs for N = 33, 34), taken from our benchmark runs on a single core.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Python generator: N, mean time (s), sample standard deviation (s)
    Ng = [16, 18, 20, 22, 24, 26]
    tg = [0.004357, 0.020913, 0.075226, 0.296597, 1.245301, 5.284663]
    sg = [0.000156, 0.002405, 0.003418, 0.013058, 0.107880, 0.702673]
    # Chunked NumPy (vectorized)
    Nv = [24, 26, 28, 30]
    tv = [0.330541, 1.281124, 5.142856, 22.143300]
    sv = [0.008448, 0.008635, 0.082801, 0.942634]

    n = np.arange(15, 37)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(Ng, tg, yerr=sg, fmt="o", color="tab:blue", capsize=3,
                label="Python generator (1 core)")
    ax.plot(n, 73.0e-9 * 2.0 ** n, "--", color="tab:blue")      # t = c 2^N, c = 73.0 ns
    ax.errorbar(Nv, tv, yerr=sv, fmt="o", color="tab:green", capsize=3,
                label="Chunked NumPy (1 core)")
    ax.plot(n, 19.4e-9 * 2.0 ** n, "--", color="tab:green")     # c = 19.4 ns
    ax.plot([33, 34], [243.9, 468.4], "k*", markersize=12, label="Single direct run")
    ax.axhline(300, color="tab:red", linestyle=":", label="5-min limit")

    ax.set_yscale("log")
    ax.set_xlabel("Number of coins $N$")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_xlim(15, 36)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    out_file = save_dir / "enumeration_scaling.pdf"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")

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


def plot_thermodynamic_limit(save_dir: Path):
    """Figure 3 (Part 1e & 2k): Narrowing of p(x) and 1/sqrt(N) relative fluctuation."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # Panel (a): Intensive probability density p(x) = N * P_k vs x = k/N
    for N, col in zip([20, 100, 500, 2500], ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
        k, _, p_k = get_distribution(N)
        x = k / N
        p_density = N * p_k
        ax1.plot(x, p_density, label=f"N = {N}", color=col, lw=1.5)

    ax1.set_title("(a) Intensive Distribution $p(x) = N P_{Nx}$ vs. Fraction $x = k/N$")
    ax1.set_xlabel("Fraction of heads $x = k/N$")
    ax1.set_ylabel("Probability density $p(x) = N P_{Nx}$")
    ax1.set_xlim(0.2, 0.8)
    ax1.set_ylim(0, 45)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)
    
    # Panel (b): Relative fluctuation sigma / mu = 1 / sqrt(N) up to N = 10^23
    n_log = np.logspace(1, 23, 300)
    rel_fluct = 1.0 / np.sqrt(n_log)
    ax2.plot(n_log, rel_fluct, "k-", lw=1.5, label="$\\sigma / \\mu = 1 / \\sqrt{N}$")
    
    # Highlight specific N values
    systems = [(20, "tab:blue"), (107, "tab:orange"), (1e23, "tab:green")]
    for n_val, col in systems:
        val = 1.0 / np.sqrt(n_val)
        name = f"$N = 10^{{23}}$:  $\\sigma/\\mu = {val:.2e}$" if n_val == 1e23 \
            else f"$N = {int(n_val)}$:  $\\sigma/\\mu = {val:.3f}$"
        ax2.plot([n_val], [val], "o", color=col, markersize=7, label=name)

    ax2.set_title("(b) Relative Fluctuation Decay Toward $N \\to 10^{23}$")
    ax2.set_xlabel("System size $N$")
    ax2.set_ylabel("Relative fluctuation $\\sigma / \\mu$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(1e1, 1e24)
    ax2.set_ylim(1e-13, 1e0)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    out_file = save_dir / "thermodynamic_limit.pdf"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")


def plot_probabilities_entropy(N: int, save_dir: Path):
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
    ax2.plot(k, ds_over_kb, color="k", lw=1.5, label="$\\Delta S(k)/k_B = \\ln(\\Omega_k / \\Omega_{N/2})$")

    # Mark points from 2e, 2f, 2g
    for kv, col in zip([20, half, half + 10, half + 25],
                       ["tab:red", "tab:green", "tab:orange", "tab:blue"]):
        ds = ln_omega(N, kv) - ln_omega(N, half)
        ax2.plot([kv], [ds], "o", color=col, markersize=6,
                 label=f"$k={kv}$: $\\Delta S/k_B={ds:.2f}$")

    ax2.set_title(f"(b) Boltzmann Entropy Relative to Equilibrium ($N = {N}$)")
    ax2.set_xlabel("Number of heads $k$")
    ax2.set_ylabel("$\\Delta S / k_B = [S(k) - S(N/2)] / k_B$")
    ax2.set_xlim(0, N)
    ax2.set_ylim(-110, 12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower center", fontsize=8.5)

    fig.tight_layout()
    out_file = save_dir / "probabilities_entropy.pdf"
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
    
    write_n7_summary_table()
    write_n7_full_table()

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

    # r_m_minus = likelihood_ratio(N, half, M - 10) # this is a negative number since our M is 7
    r_m_plus = likelihood_ratio(N, half, M + 10)
    r_minus_10 = likelihood_ratio(N, half, half - 10)
    r_plus_10 = likelihood_ratio(N, half, half + 10)
    # print(f"i) Ratio P({half}) / P(M-10={M-10}): {r_m_minus:.4e}")
    print(f"i) Ratio P({half}) / P(M-10={M-10}): k = M - 10 = {M-10} is not a valid macrostate (k must be >= 0), so no ratio exists.")
    print(f"   Ratio P({half}) / P(M+10={M+10}): {r_m_plus:.4f}")
    print(f"i) Ratio P({half}) / P({half-10}): {r_minus_10:.4f}")
    print(f"   Ratio P({half}) / P({half+10}): {r_plus_10:.4f}")

    # Mean and standard deviation of the P_k distribution (N = 100 + M), from the computed P_k
    k_all, _, p_all = get_distribution(N)
    mean_k = float(np.sum(k_all * p_all))                          # <k> = sum_k k P_k
    std_k = float(np.sqrt(np.sum((k_all - mean_k) ** 2 * p_all)))  # sigma = sqrt(sum_k (k - <k>)^2 P_k)
    print(f"Distribution of P_k for N = {N}: mean <k> = {mean_k:.4f} (N/2 = {N/2}), "
          f"std sigma = {std_k:.4f} (sqrt(N)/2 = {np.sqrt(N)/2:.4f}), sum P_k = {np.sum(p_all):.6f}")
    # --- Generate all 4 PDF figures ---
    print("\n--- Generating Figures ---")
    plot_distributions(fig_dir)
    plot_enumeration_scaling(fig_dir)
    plot_thermodynamic_limit(fig_dir)
    plot_probabilities_entropy(N, fig_dir)


if __name__ == "__main__":
    main()
