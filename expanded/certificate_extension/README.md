# Seven-point recorded certificate diagnostic

This extension evaluates the same-candidate certificate comparison at seven
approach parameters: `0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001`.
Every plotted value comes from a new numerical evaluation. No curve smoothing
or interpolation adds data points.

## Fixed inputs and comparison

The 1,246 input instances are exactly the existing D2/D3 panel: 364 from D2,
and 294 times under three initializations from D3 (882). The seeds share one
recording. All matrices, stored starting references, and local models are
unchanged. The old reference candidate is reconstructed by the original
SLSQP/FW solver from each saved `ref_w`, with tolerance `1e-10` and phase limit
200. Its objective and `min(w_ref)<1e-5` classification are checked against
the archived references. These references are feasible numerical points, not
assumed exact optima.

For each eta, the candidate is the normalized convex combination
`(1-eta)*w_ref + eta*uniform`. Both vertex and inexpensive quadratic scales use
the archived original auxiliary QP and its feasible-point dual gap. The
reported effective gap is `min(FW, GQ)`; a failed certificate retains FW. An
exact floating-point `0/0` gain is represented by the no-improvement value 1,
as in the archived normalized diagnostic. No failures are excluded.

`archived_three_point.csv` and `archived_references.csv` preserve the earlier
result snapshots used to verify the three old eta anchors. The run retains
all reconstructed reference weights, raw certificate evaluations, anchor
comparisons, numerical failures, summaries, source hashes and input hashes.
The old float64 error tolerance `2e-9` is retained for observed-bound checks.
No outward rounding or exact arithmetic is introduced.

## Reproduce

Place this folder inside `SCI_TSP_Expanded_Benchmarks/`, beside its existing
`inputs/` directory. Then run:

```bash
python run_certificate_extension.py --output rerun
```

Alternatively specify the saved per-case input directory with `--inputs`.
The default input path is `../inputs`; existing result files are never
overwritten. A pilot can use `--limit 3 --output pilot_new`. Requirements are
the parent package's NumPy, SciPy, pandas and threadpoolctl dependencies.
Only D2/D3 case filenames are selected, and a complete run requires exactly
1,246 input files. The source module `original_core.py` is an unchanged copy
of the archived implementation.

This is a certificate diagnostic, not a fresh runtime benchmark. Its wall
clock is recorded only for execution provenance. Near the approximate
reference, the feasible objective-error brackets may be too wide to resolve
the exact objective error; the plot compares two evaluated certificates,
not their ratios to an assumed exact optimum.

## Completed results

- 1,246 inputs, 8,722 fixed candidates, 17,444 certificate evaluations.
- All 7,476 old-anchor evaluations match the archived numerical results.
  Maximum old summary-median discrepancy: 4.44e-16.
- Reference geometry remains D2: 94 interior / 270 boundary; D3: 165 / 717.
- Fourteen vertex-scale spectral failures (D3 seed 42 and seed 43, epoch 875,
  seven eta values each) remain in the results with FW fallback. The three
  old eta anchors contain the original six failures. No observed bound
  violation exceeds the retained 2e-9 numerical tolerance.
- The old reference maximum retained gap is 4.51e-8. Reference brackets do
  not resolve arbitrarily small true objective errors near the reference.

The seven-point curve is not monotone: interior median gains peak near
eta=1e-3 and decrease at smaller eta for the archived auxiliary QP. These
values are retained unchanged in the primary plot data.

## Ancillary diagnosis of the near-reference downturn

`diagnose_inner_accuracy.py` re-evaluates 259 interior input instances at
eta=1e-3 and 1e-4 using the previously existing active-set QP routine. The
candidate weights, gradients, Hessians and spectral scales stay fixed. This
is an after-run mechanism diagnosis, not a new primary protocol or timing
claim. Neither `same_point_tightness.csv` nor its plotted medians is changed.

At eta=1e-4, the original auxiliary duality residual accounts for 95.4--98.2%
of the evaluated quadratic bound at the median (calculated over successful
spectral evaluations). Its median residual is approximately 6--9e-8; the
active-set routine reduces that residual to 2.22e-16. Paired re-evaluation of
the same inputs yields the following median certificate gains:

| Dataset | Scale | Original QP | Active-set QP |
| --- | --- | ---: | ---: |
| D2 | Inexpensive | 164.63 | 8260.44 |
| D2 | Vertex | 162.88 | 10795.90 |
| D3 | Inexpensive | 192.19 | 6649.66 |
| D3 | Vertex | 220.08 | 8431.45 |

These medians include the original spectral failures through unchanged FW
fallback. All 1,036 ancillary evaluations retain the original candidate and
scales, and show no bound violation beyond the recorded numerical tolerance.
The result identifies auxiliary-solve accuracy as the dominant source of the
observed downturn. It does not convert these float64 diagnostics into exact
certification or establish a runtime improvement.

Reproduce after the main extension:

```bash
python diagnose_inner_accuracy.py
```

`diagnostic_active_core.py` is an unchanged copy of the archived active-set
implementation. `auxiliary_accuracy_provenance.json` records its source hash
and the strict separation from primary plotted results.
