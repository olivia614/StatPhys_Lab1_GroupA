Following Phase 3 of the course guidelines, AI assistance was used for modular code scaffolding,
plot styling, and report formatting. Every prompt, the generated code, and the team’s modifica-
tions are logged in AI_STATEMENT.md at the repository root. Verification consisted of:
1. Automated pytest unit tests comparing every routine in src/ against exact integer arith-
metic (math.comb), normalization identities (∑︁ k Ω = 2N , symmetry), and analytical values
(Ω(100, 50) ≈ 1.009 × 1029).
2. Independent verification of all numerical values in Table 3 against exact integer fractions.
3. Analytical re-derivation by the team of all formulas in Sections 3–4.
