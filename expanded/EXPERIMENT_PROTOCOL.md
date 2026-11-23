# Expanded SCI benchmark protocol

Frozen before the formal timing runs, 2026-11-16. This extends the manuscript's fixed-dimension, three-tolerance study. Algorithms retain the preceding validated source versions.

## Comparisons

Four configurations: original matrix oracle + SLSQP + FW; batched oracle + SLSQP + FW; batched oracle + Newton + FW; batched oracle + Newton + inexpensive quadratic stopping. Gate factor is fixed at 4, each candidate-solver phase cap at 150 (Newton may invoke the existing SLSQP fallback, whose full cost is included), and every online start is uniform. Both objective and all derivative/bound quantities are normalized consistently. No benchmark output is used to retune a solver or the input generator.

## Input families

- Scaling: d in {8,16,25,64,128} at s=5; s in {2,5,10,20} at d=25. The shared (25,5) anchor is counted once. Twenty independently generated covariance inputs per setting; both normalized full trace and logdet/d. Total160 covariance sets and320 objective problems. Independently rotated SPD A andB components; no outcome filtering.
- Geometry: interior, strict boundary, nonunique flat optimal set, and ill-conditioned interior. Twenty independent common-congruence constructions in each regime, d=16,s=5; both criteria. Total80 covariance sets and160 objective problems. Analytic optima and Hessian nullity are predetermined; see GENERATOR_PROTOCOL.md. Uniform starts are nonoptimal by construction.
- Existing recorded data: exactly the prior1,246 input instances, D2=364 andD3=294 epochs under three initializations (882). These initializations share the same recording.
- Additional recorded data: D4 andD5, seed42, exact frozen replay/model configuration and calibration. Select epoch divisible by5 and time>=30s. Raw sequence provenance and hashes retained separately. No outcome-based subsampling.

## Sweeps and timing

Synthetic tolerances are {1e-2,1e-3,1e-5}. Recorded-data tolerances are {1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5}. Each case/method/tolerance is measured three times in randomized order, with single-threaded BLAS. Formal timing suites run sequentially after sequence export; no replay or another benchmark runs concurrently.

End-to-end time includes a fresh model constructor, solver and every certificate call, and the fused mean/covariance. File I/O and independent verification are outside the timer. The same small oracle-profiling wrapper is used for all methods. Oracle times are inclusive of oracle calls inside certificates and are not additive with certificate time. The actual complete wall time remains the main metric.

Reduce timing repetitions to a per-input median first. Compute paired speedup ratios perinput before summarizing. Report median andP10/P90 across inputs, plusP95/P99 times, accuracy failures, solver failures and complete-timing counts. Exception attempts retain their elapsed wall time and are identified; a failed attempt is never presented as a successful solve. Repeated timing is not an independent statistical replication; recorded time points are correlated. No unqualified confidence intervals treating all epochs as IID are reported.

## Independent numerical checks

The audit computes each branch through direct Cholesky solves of A_i+w_iB_i, independent of the timed spectral oracle. Feasible reference values and direct FW lower bounds bracket the optimum numerically. Reference interval width must be <= the smallest requested tolerance/20; unresolved cases remain flagged. Known analytic weights are supplied only to this offline reference step. Saved reference weights are never online warmstarts.

Returned candidates are checked against the independent objective interval, simplex feasibility, original covariance and fused mean. A success requires the solver's stopping test, the independent upper objective-error check, a resolved reference and output agreement. An explicit small floating-point allowance is recorded by the harness; this is not interval arithmetic or an exact rational certificate. All attempted cases, exceptions and failures remain in the raw outputs. Success-only paired speed summaries state their denominators.

## Scope

All timed paths require positive definite A_i; B_i can be semidefinite. The general singular-A theory is not implemented by this speed study and remains covered by the manuscript's earlier exact/synthetic checks. The new geometrical mechanisms deliberately isolate weight structure through common congruences; noncommuting scaling cases provide a separate broader numerical stress test.
