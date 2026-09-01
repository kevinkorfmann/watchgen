# Codex verification ledger

Last reconciled: 2026-09-01  
Baseline: GitHub `main` at `d0d99a3`  
Scope: the current source-grounded, chapter-by-chapter Codex review campaign

This ledger records **Codex verification**, not independent expert review or a
formal proof of correctness. A chapter is `VERIFIED` only after its corrections,
tests, and documentation build have completed and its chapter-scoped commit is on
GitHub `main`. Worktree-only changes do not count.

## Status meanings

| Status | Meaning |
|---|---|
| `VERIFIED` | Source/paper audit completed, validation passed, chapter-scoped commit pushed. |
| `FOLLOW-UP` | A review commit exists, but later findings or uncommitted fixes still require closure. |
| `IN PROGRESS` | An active Codex task owns the chapter; do not duplicate its work. |
| `NOT STARTED` | No completed chapter-scoped review from this campaign is recorded. |

## Prerequisites

All eight prerequisite chapters are complete. Their combined final gate was **281
tests passed**, including direct execution of published RST code, plus a clean full
Sphinx build with warnings treated as errors.

| Chapter | Status | Commit | Principal validation |
|---|---|---:|---|
| Probabilistic inference | `VERIFIED` | `e0b8c06` | Analytic likelihood/posterior oracles, optimization and invalid-input checks, RST parity. |
| Coalescent theory | `VERIFIED` | `b5c602c` | Exact waiting-time/rate formulas, simulation moments, Kingman invariants, RST parity. |
| Ancestral recombination graphs | `VERIFIED` | `84c4abd` | Graph/segment invariants, recombination bookkeeping, executable examples. |
| Hidden Markov models | `VERIFIED` | `cdafdbd` | Brute-force likelihood/path comparisons, forward-backward/Viterbi invariants, scaling checks. |
| Sequentially Markov coalescent | `VERIFIED` | `9ca5143` | Primary SMC/SMC' definitions, transition and survival oracles, simulation checks. |
| Diffusion approximation | `VERIFIED` | `8e0885c` | Wright-Fisher convergence, boundary behavior, mass conservation, numerical PDE checks. |
| Ordinary differential equations | `VERIFIED` | `72c0e93` | Closed-form solutions, convergence order, stability, exact Kingman count system. |
| Markov chain Monte Carlo | `VERIFIED` | `f6eab42` | Detailed balance, exact Beta posterior moments/quantiles, asymmetric MH, Gibbs covariance, AR(1) ESS. |

## Timepieces

Only pushed commits are treated as completed below. Test counts refer to the
chapter review evidence, not necessarily the repository-wide total.

| # | Chapter | Status | Commit / owner | Evidence or remaining closure |
|---:|---|---|---|---|
| I | PSMC | `FOLLOW-UP` | `e7336e0` | Review commit exists. A later mini audit corrected infinite-tail normalization in the shared worktree; the fix and focused PSMC suite still need a dedicated pushed commit and clean completion. |
| II | SMC++ | `VERIFIED` | `49a01f0` | 37 focused tests; upstream C++ CSFS parity within `2e-12`, transition parity within `2.4e-16`; 2,475-test broad run and strict Sphinx build passed. |
| III | Li & Stephens HMM | `IN PROGRESS` | task `Review and fix book chapters` | Paper and `lshmm` parity audit completed substantial fixes; validation/commit packaging is still active, so the earlier date in the book table is not counted as campaign closure. |
| IV | msprime | `FOLLOW-UP` | `60025dc` | Review commit exists. Later mini review fixed gap recombination and per-site/recurrent mutation-state handling in the shared worktree; those fixes need a dedicated pushed commit and final focused validation. |
| V | ARGweaver | `NOT STARTED` | — | Requires original ARGweaver source/paper parity audit and chapter-scoped validation. |
| VI | tsinfer | `NOT STARTED` | — | Requires upstream `tsinfer` parity fixtures and chapter-scoped validation. |
| VII | SINGER | `IN PROGRESS` | task `Review watchgen chapters backward` | Active audit against the original SINGER source and paper; teaching code is being bounded explicitly against the production C++ ARG engine. |
| VIII | Threads | `NOT STARTED` | — | Requires paper/software parity audit and chapter-scoped validation. |
| IX | tsdate | `NOT STARTED` | — | Requires upstream `tsdate` parity, dating-prior checks, and chapter-scoped validation. |
| X | moments | `NOT STARTED` | — | Requires original software/paper parity and SFS moment-equation fixtures. |
| XI | dadi | `NOT STARTED` | — | Requires upstream `dadi` parity, PDE stability/convergence checks, and chapter-scoped validation. |
| XII | momi2 | `NOT STARTED` | — | Requires original tensor/coalescent implementation parity and chapter-scoped validation. |
| XIII | Gamma-SMC | `NOT STARTED` | — | Requires paper/software parity and gamma-posterior numerical fixtures. |
| XIV | PHLASH | `NOT STARTED` | — | Requires original implementation/paper parity, composite-likelihood and SVGD checks. |
| XV | CLUES | `NOT STARTED` | — | Requires original implementation/paper parity and selection-likelihood fixtures. |
| XVI | SLiM | `NOT STARTED` | — | Requires upstream SLiM behavior/recipe parity and forward-simulation validation. |
| XVII | Relate | `VERIFIED` | `d0d99a3` | 37 focused checks, source-guided teaching kernels, official Relate build/example execution, output-structure validation, strict Sphinx build. |
| XVIII | discoal | `VERIFIED` | `0a00529` | 32 focused checks, primary sweep/structured-coalescent audit, regenerated figures, strict build. |

## Remaining campaign gates

1. Push and validate the post-review PSMC and msprime mini fixes.
2. Finish and push the active Li & Stephens HMM and SINGER reviews.
3. Review the eleven `NOT STARTED` Timepieces one chapter at a time.
4. Reconcile `docs/timepieces/verification.rst` from this ledger after each pushed
   chapter, while preserving its independent-domain-expert disclaimer.
5. After all chapters are closed, run the entire repository test suite, execute all
   published examples, build HTML and PDF with warnings as errors, and perform a
   final cross-chapter terminology/equation/citation audit.

## Update protocol for Codex tasks

- Claim a chapter by changing only its status to `IN PROGRESS` and adding the task
  title; avoid editing another active task's row.
- Do not mark `VERIFIED` until the commit is on GitHub `main` and its validation
  evidence is recorded here.
- Record known limitations and failed/blocked validation honestly; a plausible
  figure or passing self-referential unit test is not sufficient evidence.
- Prefer independent analytic or brute-force oracles, official executable output,
  upstream fixtures, stochastic calibration across seeds, invalid-input tests, and
  direct execution of published code.
