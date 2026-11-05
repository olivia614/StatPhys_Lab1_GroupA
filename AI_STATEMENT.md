# AI Usage Statement — Statistical Physics Lab 1

This file is required by **Phase 3 (Responsible AI Collaboration & Transparency Protocol)** of the
*Student Project Guidelines: Lab 1* (UC3M, 2026/27). It logs every AI query, what the AI produced,
what the team changed, and how the output was verified.

### Session 1
**Prompt 1.1**
> Check the guide folder, there you will find three files, one contains the report as transcribed
> latex in the main.tex fle. The slides are the old version of the lab guide, which we use for
> reference and understanding of the problem but contains not updated requisites for the report.
> The other filie is the updated guide. Note that I will need a report of all AI usage as
> specified in the guide.

**AI output:** The AI read the three PDFs in `guide/` and the repository. It elaborated a first code as included in implementation1 branch

**Prompt 1.2**
> Now, I want you to make a separate file where you explain what you did in each part overall and
> guide us on what we should be revising and understanding in each part. Do that in a new branch.

**AI output.** The AI created the branch `docs/review-guide` (from `feature/lab1-implementation`) and
wrote [`docs/REVIEW_GUIDE.md`](docs/REVIEW_GUIDE.md). For each part of the project the guide gives a
summary of the work, the concepts to understand, the checks to make (including the AI's judgement calls
on how to read questions 4, 7 and 9), and likely defense questions.

**Human modifications:** The code was finally discarded due to it being too complex to be understood fully by students. It served as a starting point and source of inspiration for the final human-made code.

**Verification:** 

1. **Unit tests** (`python -m pytest -q`): 68 passed. The tests compare every routine against
   independent references:
   - enumeration histograms vs. `math.comb` for N = 0 … 16 (generator) and N ≤ 20 (vectorised, 1 and 2 processes);
   - `log_binom` vs. `log(math.comb)` to relative 1e-12 for N up to 1000, and vs. Stirling at N = 1e23;
   - Σ Ω = 2^N, symmetry, normalisation of probabilities, ΔS sign conventions, waiting time = 2^N / Ω;
   - known value Ω(100, 50) = 1.0089e29 and P(100, 50) = 0.07959 (slides 10 and 12 of the presentation).
2. **Monte Carlo cross-check**: 100,000 simulated tosses of 100 coins against the exact distribution,
   χ² = 41.3 for 42 degrees of freedom.
3. **Benchmark verification run**: N = 36 (6.9×10¹⁰ microstates) was fully enumerated in 99 s on
   16 processes, and the histogram matched `math.comb` exactly (`report/data/benchmark.json`).
   An earlier run under heavier machine load took 322 s for N = 37, just over the 5-minute limit.
4. **LaTeX build**: the report compiles without errors or warnings (checked with the Tectonic engine;
   9 pages, body = pages 1–7, appendices from page 8). The rendered pages were inspected visually.
5. **To do by the team**: re-derive the formulas in Sections 3–4 of the report by hand, and check
   the physics discussion before submission.

**AI failures and how they were resolved**

| Problem | Detected by | Resolution |
|---------|-------------|------------|
| `float_overflow_limits()` first compared logarithms and returned N = 1025 for overflow of 2^N. The true value is 1024: `1024·ln 2` and `ln(float max)` agree to rounding error. | `test_float_overflow_limits` | Switched to exact integer comparisons (`2**n > int(float_max)`). |
| `numpy.polyfit` and every `numpy.linalg` call (also used internally by Matplotlib) crashed Python with Windows fatal error `0xc06d007f`. The threaded MKL backend in the conda-forge environment failed to load its OpenMP runtime. | Running the tests / `main.py` | Set `MKL_THREADING_LAYER=SEQUENTIAL` before importing NumPy (`main.py`, `tests/conftest.py`). Also replaced `polyfit` with a closed-form least-squares fit. The Conda environment itself was not modified. |
| Design risk: a naive `list(range(1 << N))` or storing all microstates needs 8·2^N bytes (8 GB at N = 30). | Planning | Generators (`yield`) and fixed-size chunks of 2^22 states keep memory constant (≈ 100 MB per process). |
| The first version of `benchmark()` timed each run with `tracemalloc` active. Tracing slows pure-Python loops several-fold, so the generator timings would have been inflated. | AI self-review during the first benchmark run (run aborted) | Timing runs now execute without tracing; peak memory is measured in a separate probe run. |
| Fitting a free slope to noisy timings (other load on the laptop, thermal throttling) gave slopes far from 1 and absurd extrapolations: N_max = 49 in one run, which would have taken days to verify. Process-pool start-up (~3 s on Windows) also dominated the small-N vectorised runs. | Checking benchmark progress | Reuse one warmed-up process pool. Fit only the 4 largest N. Estimate N_max from the theoretical law t = c·2^N with c = median(t/2^N) (`per_state_cost`, unit-tested with an outlier). Cap verification at 3 coins beyond the largest timed N, and step down if a run exceeds 300 s. |
| `\input` of generated table rows inside `tabular` failed (`Misplaced \noalign`): LaTeX's `\input` is not expandable. | LaTeX compilation | Added `\tabinput`, which uses the expandable TeX primitive `\@@input`. |
| The AI first wrote an invented GitHub handle for one member in `team.json`. | AI self-review | Replaced with a placeholder. Real handles must be entered by the team. |
| The first draft printed P(k = 8) at N = 1e23 with a 23-digit exponent, which double precision cannot resolve. | AI self-review | Exponent now printed to 3 significant figures (10^(−3.01×10²²)). |


### Session 2  
**Prompt 2.1**
> I have been asked to do the work described in these files (one is a general explanation, another a more
> concrete example without the GitHub part, and the third is specifically what I have to do in my
> laboratory, taking the other two into account). Taking into account the GitHub repository we have already
> created, what would be the next step? Do we have to create new documents, or is the only thing left to
> improve the code?

**AI output.** No new documents are needed: the structure required by the project guidelines already exists.
The remaining work is: agree on a base branch (`docs/review-guide` and `feature/lab1-implementation` have
diverged); fix two errors in the existing code (M computed with the arithmetic instead of the harmonic mean,
giving N = 124 instead of 107, and the sign of ΔS in questions 5–7 reversed with respect to the handout);
regenerate the results and the report; fill in `team.json`; improve and understand the code each member
will defend; keep logging AI use in this file; merge through reviewed Pull Requests, tag `v1.0` and submit.
It also pointed out the contradiction between the two documents about AI-written code (see the open issue above).

**Prompt 2.2**
> I have reviewed the different files and you are right that we have to use the harmonic mean, which is what
> they really ask for in the specific handout of this lab. And explain the sign of S to me better.

**AI output.** An explanation of the sign. The handout defines ΔS(k₁k₂) ≡ S(k₁) − S(k₂). In questions 5 and 6,
k₁ is further from N/2 than k₂ and has fewer microstates, so ΔS is negative; in question 7, k₁ = N/2 is the
most disordered macrostate, so ΔS is positive. The code computed S(k₂) − S(k₁), which only differs in sign.

**Prompt 2.3**
> Ok, now redo the whole repository fixing these two errors everywhere it is needed, and add to the AI
> statement the conversation we have just had (since I said we were starting again). It has to be in English.
> Also, if I have missed anything I have to do now (beyond the steps you already gave me), tell me too.

**AI output.** The changes below, prepared on the local branch `feature/harmonic-mean-and-entropy-sign`:

| File | Change |
|------|--------|
| `src/thermodynamics.py` | New `harmonic_mean`; `m_from_nias` uses it (clear error if a NIA ends in 00, where it is undefined). `entropy_change(n, k1, k2)` and `entropy_change_si` now return S(k1) − S(k2), as in the handout |
| `tests/test_thermodynamics.py` | Harmonic-mean tests (team case M = 7, N = 107, zero digit); entropy-sign tests (questions 5–7) and an antisymmetry test. 72 tests in total |
| `main.py` | Harmonic mean exported; odd-N note mentions the tie between k = 53 and 54; question 9 uses N/2 ± 10; entropy differences use the handout definition; magnitude shown for the N → 10²³ entropy |
| `notebooks/exploration.ipynb` | Entropy cell uses the handout's ΔS definition (comment updated, old output cleared) |
| `report/main.tex` | Text on M (harmonic), definition ΔS(k₁k₂) ≡ S(k₁) − S(k₂) with its sign convention, entropy table and discussion, question 9 as N/2 ± 10; test count updated to 72 |
| `report/results.tex`, `report/figures/*.pdf`, `report/main.pdf` | Regenerated with `python main.py` for N = 107 and recompiled |
| `docs/REVIEW_GUIDE.md` | Section 7 and related notes updated: harmonic mean, N = 107, sign convention, new results table |
| `AI_STATEMENT.md` | This entry, the open-issue note and the new tools row. All Session 1 entries are kept |

**Prompt 2.4**
> Now, with the new branch in which the changes were made, do this part: "Audit & Test: All AI-generated logic
> must be backed by unit tests", and tell me if there is any error that should be changed or improved. Also
> document all of this in this new branch, and give me again a zip with this new branch and all the changes
> included.

**AI output.** An audit of the code and of the existing tests, recorded in full in
[`docs/AUDIT.md`](docs/AUDIT.md). Summary:

- **Coverage** went from 81% (`src/`) and 0% (`main.py`) to 100% and 99%; the suite went from 72 to 147 tests
  (145 checks and 2 `xfail` tests that document known limitations).
- **Errors found:** (F2) the report claimed that the log-gamma formula "has no limit at all" and agrees with
  Stirling's formula at N = 10²³; in fact probabilities and entropy differences are reliable only up to
  N ≈ 10¹⁰ (measured table in `docs/AUDIT.md`), so the report text, three docstrings and the review guide were
  corrected; (F1) `sci()` mis-formatted negative numbers just below a power of ten; (F3, F4, F5) three tests that
  did not test what they claimed (the NumPy 1.x popcount fallback was never executed, a tolerance of 10⁻¹² hid
  errors of 10¹⁰, and one test compared a formula with itself); (F6) `main.py` had no tests.
- **Decision left to the team (F7):** the convention for "N/2 ± 10" when N = 107 is odd.
- **Mutation analysis:** 11 bugs were injected (including the two errors corrected in prompt 2.3); 10 were
  detected, and the 11th turned out to be an equivalent change (tails and heads windows have the same
  probability for a fair coin).

| File | Change |
|------|--------|
| `docs/AUDIT.md` | New: method, findings F1–F12, measured precision table, coverage, test-to-logic map, mutation analysis, open decisions |
| `tests/test_main.py` | New: 35 tests of `main.py` (formatting, team files, each part of the report against exact arithmetic, whole-program run, check that `report/results.tex` is not stale) |
| `tests/test_microstates.py`, `tests/test_combinatorics.py`, `tests/test_thermodynamics.py` | Weak tests replaced (F3, F4, F5); tests added for every uncovered function and edge case; two `xfail(strict)` tests documenting F2 |
| `main.py` | `sci()` fix (F1) |
| `src/combinatorics.py`, `src/thermodynamics.py` | Docstrings only: valid numerical range of `log_binom`; `log10_prob_fraction` is leading order. No change in behaviour |
| `report/main.tex` | Corrected the log-gamma paragraph and the conclusion (F2); test count and reference to the audit |
| `docs/REVIEW_GUIDE.md` | Sections 0, 4 and 11 corrected and updated |
| `.gitignore` | Ignores `.coverage` and `htmlcov/` |

### Verification performed by the AI

1. `python -m pytest -q`: 72 passed after prompt 2.3; **145 passed and 2 expected failures (`xfail`) after prompt 2.4**,
   with 100% line coverage of `src/` and 99% of `main.py` (`pytest --cov`).
2. The N = 107 results were recomputed independently with exact integer arithmetic (`math.comb`,
   `fractions`) and match `report/results.tex`: P(30) = 1.963e-6, P(53) = P(54) = 0.07660,
   P(43 ≤ k ≤ 63) = 0.9572, P(33 ≤ tails ≤ 73) = 0.999934, ΔS = −11.527, −1.675 and +22.371 (units of k_B),
   P(53)/P(63) = 5.34, P(53)/P(43) = 7.77, waiting time for 8 heads 10^20.70 s.
3. The report compiles (two `pdflatex` passes, 0 errors, 0 overfull boxes, 9 pages).
4. Mutation analysis of prompt 2.4: 10 of 10 real injected bugs are detected by the suite (see `docs/AUDIT.md`, section 4).
5. **To do by the team:** run everything on a personal laptop, re-derive the formulas by hand, and decide the
   judgement calls listed in `docs/REVIEW_GUIDE.md` (rounding of M, odd N, question 9).

### AI failures and how they were resolved (Session 2)

| Problem | Detected by | Resolution |
|---------|-------------|------------|
| Session 1 used the arithmetic mean for M (taken from the old slides), giving N = 124, while the updated handout and the template say harmonic (N = 107). All part-2 numbers were affected. | Comparing the code with the updated handout | `m_from_nias` fixed, tests updated, everything regenerated |
| `entropy_change` computed S(k2) − S(k1), the opposite of the handout's ΔS(k1k2) ≡ S(k1) − S(k2): the signs of questions 5–7 were reversed. The Session 1 tests checked the code against itself, not against the handout. | Re-reading the handout's definition against the code | Function, tests, report and review guide changed; the convention is now stated in the report |
| With M = 7 the reading "k = M ± 10" of question 9 gives k = −3, which is impossible. | Running the code with the new N | Question 9 read as N/2 ± 10, documented in the report and the review guide. **To be confirmed with the instructor.** |
| The entropy table wrapped its rows and one limit value was printed with a sign inside an absolute value. | Reading the compiled PDF | Ratios removed from the table; the magnitude is printed |
| Session 1 wrote, and the report stated, that the log-gamma formula "has no limit at all" and is validated at N = 10²³. The validating test used a relative tolerance of 10⁻¹² on a value of 7·10²², so it could not see the missing −27 correction. | Audit (prompt 2.4): a numerical experiment at N = 10¹⁵ and 10²³ | Report, docstrings and review guide corrected with the measured limit (N ≈ 10¹⁰); test replaced; the limit is documented by `xfail` tests. Code unchanged |
| `sci()` printed `-10.00\times10^{-5}` for −9.9996·10⁻⁵. | Audit (prompt 2.4), edge-case experiment | Fixed and tested |
| `test_popcount_fallback_matches` never executed the fallback it was named after (NumPy ≥ 2 uses `np.bitwise_count`). | Coverage report | Test rewritten to force the fallback |
| *Audit's own mistakes.* (a) A new test assumed the N = 7 table is listed from 0 to 7 heads; the code lists 7 down to 0, as in the handout. (b) One injected bug looked like an undetected gap in the tests, but it was an equivalent change. | Test run; investigating the surviving mutant | (a) The test assumption was corrected, not the code. (b) Documented in `docs/AUDIT.md` with a test of the symmetry |

**Human modifications:** Although some mistakes detected by the AI in this code were taken as prevention for the new code, the code was still doscarded due to high complexity and used obly as inspiration.

### Verification of the final code

The final code in `src/` and `main.py` was written by the team; the AI-generated code of
Sessions 1–2 was used only as a reference and is not part of the submission.

1. **Unit tests** (`python -m pytest -q`, 8 tests, all passing). They compare the core routines
   with Python's exact integer arithmetic (`math.comb`):
   - enumeration: the generator yields exactly 2^N states; the N = 7 table contains 128 microstates
     with Ω(7, k) = C(7, k); the generator, bitwise and vectorised histograms reproduce C(N, k)
     exactly for N = 5–20;
   - combinatorics: `ln_omega` agrees with log C(N, k) to relative 1e-12 for N ≤ 150; probabilities
     are normalised (Σ_k Ω = 2^N) and symmetric (P_k = P_{N−k}); `cumulative_prob` agrees with an
     exact integer sum;
   - thermodynamics: the harmonic mean is checked on a known input, and ΔS and the likelihood
     ratio are checked to be consistent with each other (P(k1)/P(k2) = exp(ΔS/k_B)).
2. **Cross-check of Table 3.** During review (with AI assistance), every value for N = 107 was
   recomputed with exact rational arithmetic (`fractions.Fraction`, `math.comb`). Probabilities 2a–2d
   and ΔS/k_B for 2e–2g matched; the values for 2h, 2i and the SI entropies were corrected in the report.
3. **Analytical re-derivation** by the team of the formulas in Sections 3–4.