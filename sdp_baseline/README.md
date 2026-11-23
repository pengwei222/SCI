# Same-candidate SDP comparison for multisensor SCI

We compare the manuscript's trace certificates with an independently formulated semidefinite program for exactly the same trace minimization. The SDP solves the full weight-selection problem and yields a lower bound reusable for every candidate on the same covariance input. A Hessian certificate assesses the supplied candidate with less computation. This comparison uses the standard Schur-complement epigraph and a general SDP solver. It does not claim to reproduce the exact numerical implementation of Pedroso et al.

## Sample and procedure

The prespecified sample comprises all 32 trace inputs in the archived grid with state dimensions 2 or 4, sensor counts 3 or 6, both covariance condition numbers, both sensor scale regimes and both replicates per cell. Each saved candidate uses η = 0.001. Both covariance components are strictly positive definite. File loading, reference retrieval and package loading occur outside the clock. After an unrelated 1D warmup, each input receives three separately constructed cold-start solves for each of the SDP, vertex and inexpensive methods, in reproducibly randomized order. Time includes input preprocessing for every method, plus the lift where required. Runs use single-thread settings. The SDP is solved from scratch without the candidate or the reference. We use the reference only in the offline comparison with the true objective error. Table summaries use the median over repetitions for each input before forming paired ratios. The optional exact check of the SDP certificate is timed separately.

## Primal and dual

Stack one d-dimensional block per sensor, let C vertically stack s identity matrices, let B be the block diagonal of the residual covariances, and let E select the first d coordinates. Define the SPD affine matrix

    M(w) = [[C' B^{-1} C, C' B^{-1}],
            [B^{-1} C, B^{-1} + diag_i(w_i A_i^{-1})]].

For positive weights its leading inverse block E' M(w)^{-1} E equals the SCI covariance by the Woodbury identity. Positive definiteness persists at all simplex vertices, as follows from the quadratic form sum_i ||x+y_i||²_{B_i^{-1}} + sum_i w_i ||y_i||²_{A_i^{-1}}. Continuity extends the identity to boundary weights. We solve

    minimize tr(X)/d
    subject to simplex weights w,  [[M(w), E], [E', X]] >= 0.

The solver uses a first-block congruence that whitens M at the uniform weight for numerical conditioning; the returned factors are mapped back into original coordinates. The solve uses CVXPY/Clarabel with at most 500 iterations, absolute-gap, relative-gap and feasibility tolerances of 1e-11, and no warm start. We retain every instance, including reported lower-accuracy terminations.

Every real rectangular matrix K of the same shape as E supplies a feasible dual witness

    Y(K) = [[KK', -K], [-K', I]]/d >= 0,
    L(K) = [2 tr(K'E) - max_i tr(K' M(e_i) K)]/d <= f*.

Indeed the inner product of Y(K) with each feasible primal PSD block is nonnegative; M(w) is a convex combination of the vertex matrices. Equivalently the witness completes the square for matrix inversion. This construction enforces dual positive semidefiniteness and the trace-objective stationarity equation exactly. Take K from (i) the off-diagonal block of the returned PSD dual multiplier, or (ii) M(ŵ)^{-1}E from the solver's normalized nonnegative primal weights. The latter is the first-order supporting-plane lower bound at the SDP solution in exact arithmetic. Both use the result of the same complete SDP solve. We keep the larger genuinely feasible lower bound, and report both individually. The raw solver dual objective is never used as a guaranteed lower bound. Recovering a certificate this way charges the SDP for its entire solve.

We verify both K factors as exact binary rationals. The verifier checks symmetry and all leading principal minors of every original binary covariance component, then takes exact small-block inverses. With K written in d-row blocks K_0,K_1,…,K_s, the vertex trace entering the bound is

    sum_j tr[(K_0+K_j)' B_j^{-1} (K_0+K_j)] + tr[K_i' A_i^{-1} K_i].

The verifier also reconstructs the supplied candidate by exact rational normalization of its stored nonnegative binary weights, and computes its objective f(c) rationally. Reported upper bounds are widened one float64 ULP upward; the lower bound for f* is widened downward. This establishes the displayed upper bound for the exact archived binary matrices and rationally normalized candidate. The vertex and inexpensive certificates in this comparison are the archive's float64 implementations. As an offline diagnostic, the feasible reference and its exact rational Frank–Wolfe gap bracket the candidate's objective error. Reference computations are excluded from all method timings. Eigenvalue and stationarity residuals of the raw solver returns are recorded as numerical diagnostics only.

## Reproducing the comparison

Use the revised experiment package and run the commands below from its root, the directory containing both `sdp_baseline/` and `synthetic/`. The timed comparison imports the copy of `synthetic/code/benchmark_core.py` in this package. The Python and package versions used for the distributed run are listed in [environment.json](results/environment.json). Choose a new output directory for each run.

    python sdp_baseline/compare_sdp.py --archive . --output fresh_sdp_results --repeats 3

For a two-input smoke check use a separate output directory.

    python sdp_baseline/compare_sdp.py --archive . --output fresh_sdp_smoke --limit 2 --repeats 1
 The output includes individual trials, one row per covariance input, summaries by optimal-weight regime, raw numerical residuals, original-coordinate dual witness factors and environment versions. In cases.csv, sdp_total_s includes numerical model assembly, cold CVXPY compilation and cone solve, dual recovery and direct objective evaluation. solver_internal_s is the solver-reported subset, while exact_factor_verification_s is an additional optional online cost. All three methods' numerical total timings exclude the offline checks. A single SDP lower bound can subsequently assess any additional candidate without another SDP solve. Machine-dependent cold comparisons do not establish an amortized advantage for repeated evaluations.

## Archived run

| Reference group | Inputs | Median vertex bound / SDP bound | Median inexpensive bound / SDP bound | Median FW bound / SDP bound |
|---|---:|---:|---:|---:|
| All | 32 | 2.329 | 4.059 | 3892.3 |
| Interior | 17 | 3.463 | 5.261 | 6156.4 |
| Boundary | 15 | 1.003 | 1.882 | 2.428 |

The SDP bound was tighter than both local certificates in each of these 32 cases. The exact reference bracket bounds the ratio of the SDP upper certificate to the actual objective error; the empirical median lies between 1.00000351 and 1.00000354, and every case is below 1.003515. The factor built from the solver dual was selected in all 32 cases. Six SDP solves were marked optimal and 26 optimal_inaccurate at the requested tolerance. All 32 independently reconstructed lower bounds passed the exact rational checks.

The [summary table](results/summary.csv) gives the current median cold times for all three methods, the paired time ratios, and the additional exact verification time. Time columns use seconds. The [per-trial timings](results/cold_times.csv) record each cold-start repetition and the [per-input results](results/cases.csv) give its cost breakdown. These measurements correspond to one candidate per input on the [recorded host](results/environment.json). A computed SDP lower bound can be reused when the same input is assessed at several candidate weights.

To recheck all recorded exact certificates from their saved dual factors without rerunning any optimizer, use:

    python sdp_baseline/verify_witnesses.py --archive . --results sdp_baseline/results

The recheck requires only NumPy and SymPy. It evaluates the original binary matrices using exact rational operations and checks the outward rounding of the recorded limits against the exact rational values.
