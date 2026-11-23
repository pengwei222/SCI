# Predetermined synthetic input protocol

The generator is `synthetic_inputs.py`. No seed is selected, rejected, or
modified according to a computed optimum, speedup, or certificate result.
Default `make_cases()` returns 480 objective cases from 240 covariance inputs.
Trace and logdet share each covariance input and local mean array.

## Scaling panel

- State-dimension sweep: `d = 8,16,25,64,128`, `s = 5`.
- Estimate-count sweep: `s = 2,5,10,20`, `d = 25`.
- The common `(25,5)` setting is generated once: eight unique settings.
- Twenty independent fixed seed sequences per setting: 160 covariance inputs,
  320 objective cases.
- Every A_i and B_i has its own dense random orthogonal eigenvectors, with
  prescribed individual spectral conditions 30 and 10. Thus the components are
  generally noncommuting. Their mean eigenvalues vary across sensors over
  predetermined heterogeneous ranges, with small continuous random changes.
- Geometry is not imposed or filtered in this panel; classify it from the
  independently computed reference solution.

The `panel` value is `scaling`; `metadata.scaling_axes` specifies which sweep(s)
use an input. Both sweep names occur at the common anchor. Selection by
`d == 25` or `s == 5` also gives the intended sweeps.

## Mechanism panel and exact geometry

All mechanism inputs have d=16, s=5, with twenty fixed seeds for each of four
regimes. Let C be a random dense SPD matrix, and put A_i=a_i C, B_i=b_i C.
Then J(w)=q(w) C^{-1} and P(w)=C/q(w), where

    q(w) = sum_i w_i/(a_i+b_i w_i).

Both positive trace and logdet criteria therefore maximize q. These inputs
are structural mechanism tests, not a claim of representative noncommuting
covariance statistics; the separate scaling panel supplies that setting.

1. **Unique interior optimum.** Every a_i=1 and b_i>0. The exact optimum is
   w_i*=(1/b_i)/sum_j(1/b_j). At that point all branch derivatives
   1/(1+b_i w_i*)^2 are equal. Strict concavity of q makes the optimum unique.
   Predetermined heterogeneous b_i keep the optimum away from the uniform
   cold start. The condition of C is 30.

2. **Strict boundary optimum.** Choose one branch j and require
   1/a_i < a_j/(a_j+b_j)^2 for every i != j. These are the strict simplex
   KKT inequalities for maximizing q at e_j; all b_i are positive, so e_j is
   the unique optimum. The generator records the strict scalar KKT margin.
   The condition of C is 30.

3. **Flat nonunique optimum.** Four branches have B_i=0 and the same scalar
   a_affine. A fifth branch has positive a,b. A target v in [0.40,0.65] is
   drawn before solving, and a_affine=(a+bv)^2/a. Thus the complete optimum
   set fixes the nonlinear branch's weight to v and leaves the other four
   nonnegative weights with sum 1-v. Its dimension and tangent Hessian
   nullity are three. The provided w_star is only one representative.
   The condition of C is 30.

4. **Ill-conditioned interior optimum.** The interior construction is used
   with condition(C)=10^6 and a b_i spread of approximately 1,500. This creates
   ill-conditioned covariance matrices and strongly heterogeneous curvature
   in weight space while retaining an exact unique interior optimum. It is
   separate from the moderate interior regime.

For every mechanism case, metadata includes a_i, b_i, the expected geometry,
one analytic optimal weight, and analytic objective values. These formulae
specify exact structural properties. Their float64 evaluation must not be
treated as interval arithmetic: independently evaluate the supplied optimum
and its first-order gap before accepting a numerical reference interval.

## Normalization and checks

Use the supplied constructor defaults with heading_length=1 and no pose-only
restriction. Trace is divided by its value at uniform weights, and logdet by
d, matching the tested implementations. Absolute objective tolerances are
interpreted within each objective family.

`generator_validation.json` records untimed checks of all 160 mechanism
objective cases and a one-seed scaling construction check. Maximum first-order
gap at an analytic mechanism optimum was 1.57e-12; maximum discrepancy from
the scalar objective formula was 1.58e-11. All expected tangent nullities
matched, all A_i were positive definite, and every uniform cold start had
objective error at least 0.0445. This validation is a numerical check, not
an exact rational certification or a timing experiment.

`save_inputs(cases, directory)` writes one NPZ per covariance input and an
objective-case manifest, and refuses to overwrite existing input files.
