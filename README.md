# SCI reproducibility package

Download this repository using **Code → Download ZIP**, extract it, and run the following from the directory containing the parts. Python 3 and about 450 MiB of free disk space are sufficient for restoration.

    python3 restore.py
    cd sci-repro
    python3 plotting/summarize_thresholds.py

The restoration script verifies the SHA-256 digest of the complete archive before extracting it. Each part belongs to a single ZIP stream. To reproduce the other results, follow the instructions below from the restored sci-repro directory.

## Inspect the reported results

    python plotting/summarize_thresholds.py
    python plotting/summarize_runtime.py
    python diagnostics/summarize_existing_experiments.py .

The first command pairs certificates at the same candidate and tolerance. For eta = 0.001 and tolerance = 0.00001, the counts over all 1,246 inputs are 584 (Frank–Wolfe), 844 (inexpensive quadratic) and 887 (vertex quadratic). For the 259 interior-reference inputs the counts are 3, 252 and 252. The second command summarizes per-input timing ratios across the four robot datasets. Original timings are archived; new timings depend on hardware and numerical libraries.

A second run on arm64 macOS, with NumPy 2.3.5, SciPy 1.17.0 and pandas 2.2.3, recomputed all 17,444 robot certificate evaluations. The acceptance counts in the main robot table agreed with the archived Linux run. Of 7,476 individual comparisons at the three historical candidate settings, 6,801 met the prespecified numerical matching tolerances. For example, the Dataset 3 interior vertex median at eta = 0.001 was 486.63 in this run and approximately 483 in the archived run. Both runs recorded 14 fallback evaluations and no violations of the numerical lower-gap check at its stated tolerance. The paper reports the archived run.

## Install and reproduce

    python -m venv .venv
    . .venv/bin/activate
    python -m pip install -r requirements.txt

The default requirements pin the original synthetic and robot numerical libraries. The archived SDP timings used a separate numerical environment documented in `sdp_baseline/results/environment.json`. The robot reference optimizer is rerun from each stored warm start. Different numerical libraries can change the final floating-point weights and near-optimum gaps. The pilot writes `anchor_comparison.csv` and `quality.json` with per-anchor comparisons and bound checks. Archived counts and fresh optimizer runs are separate outputs.

| Paper result | Source and reproduction entry |
| --- | --- |
| Optimal-weight family | `python plotting/plot_weight_family.py --output redrawn` |
| Three main experimental figures | `python plotting/redraw.py --output redrawn` |
| Appendix runtime, scaling and geometry figures | `python expanded/plot_expanded.py --only runtime scaling geometry --output redrawn` |
| Synthetic trace and log-determinant sweep | `synthetic/code/run_main.py` and `synthetic/data/certificate_results.csv` |
| Exact, high-precision and stress panels | `synthetic/code/run_exact.py`, `run_high_precision.py`, `run_stress.py` and `check_stress_scales.py` in the same directory |
| SDP comparison | `sdp_baseline/compare_sdp.py`, `verify_witnesses.py`, archived results and protocol in `sdp_baseline/` |
| Robot fixed-candidate curves and inner-solver comparison | `expanded/certificate_extension/run_certificate_extension.py` and `diagnose_inner_accuracy.py` |
| Timings and ablations | `expanded/run_expanded.py`, `expanded/EXPERIMENT_PROTOCOL.md` and `expanded/results/` |
| Robot position and coverage diagnostics | `python diagnostics/summarize.py` |

Recheck all 32 archived SDP witnesses, run the scale regression check and start small independent solver runs with:

    python sdp_baseline/verify_witnesses.py --archive . --results sdp_baseline/results
    python synthetic/code/test_scale_route.py
    python sdp_baseline/compare_sdp.py --archive . --output rerun/sdp-smoke --limit 2 --repeats 1
    python expanded/certificate_extension/run_certificate_extension.py --limit 3 --output rerun/robot-smoke

The synthetic runners write into their own module. To repeat the synthetic stress and exact checks without replacing archived results:

    mkdir -p rerun
    cp -R synthetic rerun/synthetic-copy
    python rerun/synthetic-copy/code/run_stress.py
    python rerun/synthetic-copy/code/check_stress_scales.py
    python rerun/synthetic-copy/code/run_exact.py

For the full fixed-candidate robot experiment omit `--limit` and use a fresh output directory. For the full SDP sweep omit `--limit` and use three timing repeats. To rerun timing experiments, use a separate working copy and follow the sweep, single-threaded execution and pairing protocol in `expanded/EXPERIMENT_PROTOCOL.md`. The per-case inputs are already present in `expanded/inputs/`; `expanded/run_expanded.py --suite saved --banks DIR` accepts a directory of selected per-case NPZ files. Use distinct output directories for different suites.

## Data attribution

The recorded-data experiments use UTIAS MRCLAM by K. Y. K. Leung, Y. Halpern, T. D. Barfoot and H. H. T. Liu, *The UTIAS Multi-Robot Cooperative Localization and Mapping Dataset*, International Journal of Robotics Research, 2011, DOI [10.1177/0278364911398404](https://doi.org/10.1177/0278364911398404). The repository includes derived fusion inputs, replay diagnostics (including sampled reference positions) and benchmark results. Full original sensor recordings can be obtained via the [dataset homepage](https://asrl.utias.utoronto.ca/datasets/mrclam/). The frozen replay implementation, calibration, pinned data-source provenance and file hashes for the additional recordings are in `expanded/sequence_extension/`. Fixed-input experiments do not download any recordings.
