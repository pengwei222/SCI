# Stress-test vertex scales

The revised stress experiment tests 32 inputs, two objectives and two candidate weights, giving 128 vertex scales. Each scale passed two separate exact-arithmetic checks of all its vertex inequalities. The two checks cover 576 inequalities each. All 128 accepted their first spectral proposal, and none invoked the analytic or doubled-scale fallback. The remaining 128 inexpensive evaluations completed using the original implementation.

## What is verified

The file `data/stress_factor_inputs.npz` stores the floating-point factors and split eigenvalues actually used. Exact checks interpret each binary number as its rational value. They interpret the floating center in `data/stress_vertex_audit.csv` as its exactly normalized rational vector. The stored hexadecimal floating-point scale names the exact binary number used to construct the quadratic. The verification proves the global upper **matrix comparison** for these factors, this center and this scale. Objective derivatives, auxiliary solves, final gap estimates, numerical references and envelope probes in these stress tables use floating-point arithmetic. The separate rational end-to-end experiment remains the source of exact objective-gap certificates.

The unchanged seed protocol can give slightly different binary factors and reference solutions on another platform. Independently checking the shipped factors and centers avoids this dependency. The legacy archive contains seeds and scalar results but omits the original stress factors and candidate weights. Evaluating its archived scales on the reproduced centers gave 111 passes and nine failures among the 120 returned numbers, as well as eight originally reported numerical failures. Six of the nine failures occur at exactly uniform rational centers. The three others occur at regenerated near-optimal centers. These counts test the sufficient matrix inequality and do not directly establish a false reported objective gap.

## How the scale is obtained

The ordinary vertex certificate automatically switches to a stable, verified computation when a mixed split eigenvalue is within the square root of machine epsilon of either endpoint. The stress driver explicitly requests verification for every vertex evaluation. The ordinary floating-point path remains available on inputs outside this trigger, with its numerical status recorded. An explicit `SCI.scales(c, verified=True)` or `certificate(model, c, mode, verify_scale=True)` requests the exact upper-scale check for other factored inputs too.

The proposal block-diagonalizes the lift at the center by Schur elimination, whitens the resulting blocks, and forms the symmetric changes induced by each vertex. The positive margin on the computed eigenvalue is a proposal. To check a number $b>1$, let $k=b-1$ and $u_i=b c_i-(e_j)_i$ for vertex $j$. For a mixed split mode with eigenvalue $a$ and factor column $z$, the auxiliary block of the difference pencil is $k a+(1-a)u_i$. Its contribution to the Schur complement is $[k u_i/(k a+(1-a)u_i)]zz^T$.

The coefficient is $k$ for a constant mode and $u_i$ for a linear mode. The implementation requires every mixed-mode diagonal to be positive and checks positive definiteness of the summed Schur matrix by integer fraction-free elimination. An unsuccessful proposal uses the analytical scale $1/\min_i c_i$, rounded upward, in the interior. At the boundary it doubles the scale and verifies each candidate exactly. If a representable certified scale cannot be found it raises an error. For `both=True` it verifies the lower comparison in reverse and halves the candidate lower scale as needed.

An independent regression program reconstructs the original arrowhead blocks after a rational congruence. It forms the original top-left terms and the separate Schur corrections exactly, then checks positive semidefiniteness by rational pivoted elimination. Its all-vertex result is saved in `stress_vertex_verification.csv`. Public-entry-point tests cover cases 3, 7 and 15 at uniform and boundary centers, both objectives, both scales, and injected unsafe proposals exercising both fallback branches.

## Reproduction

The runner requires NumPy, SciPy, pandas and threadpoolctl. The independently frozen-input checks require NumPy and SciPy for the entry-point test, and NumPy for the full archival comparison.

```sh
python synthetic/code/run_stress.py
python synthetic/code/check_stress_scales.py
python synthetic/code/test_scale_route.py
```

Run just the second and third commands to verify the shipped artifacts without regenerating the inputs. The independent archival check writes its own results file. The full runner reads the unmodified baseline copy `data/stress_results_original.csv` and regenerates `stress_results.csv`, `stress_envelopes.csv`, `stress_vertex_audit.csv`, and `stress_factor_inputs.npz`. The envelope table contains matched first-order-plus-curvature probes for both inexpensive and vertex scales. A maximum positive signed violation is a floating-point consistency diagnostic, with normalization by $\max(1,|f(w)|)$.
