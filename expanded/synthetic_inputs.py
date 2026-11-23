"""Predetermined SCI scaling and mechanism inputs, with no outcome filtering.

This module generates covariance components and local means only.  It does not
run a timed solver and does not change any implementation under comparison.
Scaling inputs have independently rotated, noncommuting SPD A_i and B_i.
Mechanism inputs use A_i=a_i C and B_i=b_i C, so their optimal-weight geometry
can be stated analytically.  Singular A_i are intentionally outside this fast
implementation's scope; the flat mechanism has semidefinite B_i=0 branches.
"""
from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
import numpy as np

BASE_SEED = 20261116
DIMENSIONS = (8, 16, 25, 64, 128)
SENSOR_COUNTS = (2, 5, 10, 20)
REGIMES = ("interior", "strict_boundary", "flat_nonunique", "ill_conditioned")


def _rng(tag: int, d: int, s: int, seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, tag, d, s, seed]))


def _spd(rng: np.random.Generator, d: int, condition: float,
         mean_eigenvalue: float = 1.0) -> np.ndarray:
    """Dense SPD matrix with prescribed spectral condition, up to rounding."""
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    q *= np.where(np.diag(r) < 0.0, -1.0, 1.0)[None, :]
    eigenvalues = np.geomspace(1.0, condition, d)
    eigenvalues *= mean_eigenvalue / eigenvalues.mean()
    c = (q * eigenvalues) @ q.T
    return (c + c.T) * 0.5


def _scaling_input(d: int, s: int, seed: int) -> dict:
    rng = _rng(101, d, s, seed)
    # Both fixed heterogeneous ranges and continuous random perturbations are
    # specified before solving.  No reference optimum or timing affects input
    # generation.  Each sensor uses separate eigenvectors for A_i and B_i.
    a_scales = rng.permutation(np.geomspace(0.55, 1.80, s))
    b_scales = rng.permutation(np.geomspace(0.30, 3.00, s))
    a_scales *= np.exp(rng.uniform(-0.10, 0.10, s))
    b_scales *= np.exp(rng.uniform(-0.10, 0.10, s))
    a = np.array([_spd(rng, d, 30.0, scale) for scale in a_scales])
    b = np.array([_spd(rng, d, 10.0, scale) for scale in b_scales])
    axes = []
    if s == 5 and d in DIMENSIONS:
        axes.append("dimension")
    if d == 25 and s in SENSOR_COUNTS:
        axes.append("sensors")
    return dict(
        covariance_id=f"scale_d{d:03d}_s{s:02d}_seed{seed:02d}",
        panel="scaling", regime="noncommuting_random_spd", d=d, s=s,
        seed=seed, A=a, B=b, means=rng.standard_normal((s, d)),
        w_star=None, geometric_expected="classify_from_reference",
        metadata={
            "seed_sequence": [BASE_SEED, 101, d, s, seed],
            "scaling_axes": axes,
            "condition_A": 30.0, "condition_B": 10.0,
            "A_mean_eigenvalues": a_scales.tolist(),
            "B_mean_eigenvalues": b_scales.tolist(),
            "construction": "independent dense SPD rotations per branch/component",
            "reference_kind": "independent_numerical_reference_required",
            "selection": "all predetermined seeds retained",
        },
    )


def _mechanism_input(regime: str, seed: int, d: int = 16, s: int = 5) -> dict:
    if s != 5:
        raise ValueError("The preregistered mechanism panel has five estimates.")
    tag = 201 + REGIMES.index(regime)
    rng = _rng(tag, d, s, seed)
    condition = 1.0e6 if regime == "ill_conditioned" else 30.0
    c = _spd(rng, d, condition)
    metadata = {
        "seed_sequence": [BASE_SEED, tag, d, s, seed],
        "construction": "A_i=a_i C, B_i=b_i C with random common SPD C",
        "condition_C": condition,
        "reference_kind": "analytic_scalar_information_geometry",
        "selection": "all predetermined seeds retained",
    }
    if regime in ("interior", "ill_conditioned"):
        a = np.ones(s)
        values = ([0.35, 0.65, 1.15, 2.20, 4.00] if regime == "interior"
                  else [0.02, 0.10, 0.70, 5.00, 30.00])
        b = rng.permutation(np.array(values, dtype=float))
        b *= np.exp(rng.uniform(-0.07, 0.07, s))
        w_star = (1.0 / b) / np.sum(1.0 / b)
        geometric_expected = "unique_strict_interior"
        metadata.update(
            tangent_nullity=0,
            optimality_formula="w_i*=(1/b_i)/sum_j(1/b_j), since a_i=1",
            b_ratio=float(b.max() / b.min()),
        )
    elif regime == "strict_boundary":
        best = int(rng.integers(s))
        a_best = float(rng.uniform(0.8, 1.2))
        b_best = float(rng.uniform(0.10, 0.20))
        lambda_best = a_best / (a_best + b_best) ** 2
        factors = rng.permutation(np.array([1.5, 2.0, 3.0, 4.0]))
        factors *= np.exp(rng.uniform(-0.05, 0.05, s - 1))
        a = np.empty(s)
        others = np.array([i for i in range(s) if i != best])
        a[others] = factors / lambda_best
        a[best] = a_best
        b = rng.uniform(0.5, 3.0, s)
        b[best] = b_best
        w_star = np.eye(s)[best]
        geometric_expected = "unique_strict_vertex"
        metadata.update(
            tangent_nullity=0, active_branch=best,
            optimality_formula="1/a_i < a_best/(a_best+b_best)^2 for every inactive i",
            scalar_kkt_margin=float(lambda_best - np.max(1.0 / a[others])),
        )
    elif regime == "flat_nonunique":
        nonlinear = int(rng.integers(s))
        target = float(rng.uniform(0.40, 0.65))
        a_nonlinear = float(rng.uniform(0.8, 1.2))
        b_nonlinear = float(rng.uniform(1.2, 2.8))
        # The four B_i=0 branches share one affine information coefficient.
        # Set it to the nonlinear branch's derivative at the desired optimum.
        a_affine = (a_nonlinear + b_nonlinear * target) ** 2 / a_nonlinear
        a = np.full(s, a_affine)
        b = np.zeros(s)
        a[nonlinear] = a_nonlinear
        b[nonlinear] = b_nonlinear
        w_star = np.full(s, (1.0 - target) / (s - 1))
        w_star[nonlinear] = target
        geometric_expected = "nonunique_affine_optimal_face"
        metadata.update(
            tangent_nullity=s - 2, nonlinear_branch=nonlinear,
            nonlinear_optimal_weight=target,
            optimality_formula="w_nonlinear*=target; remaining weights nonnegative with sum 1-target",
            affine_A=a_affine,
        )
    else:
        raise ValueError(regime)

    uniform = np.full(s, 1.0 / s)
    scalar_information = lambda w: float(np.sum(w / (a + b * w)))
    q_star = scalar_information(w_star)
    q_uniform = scalar_information(uniform)
    sign, logdet_c = np.linalg.slogdet(c)
    assert sign == 1
    metadata.update(
        a=a.tolist(), b=b.tolist(), scalar_information_at_optimum=q_star,
        scalar_information_at_uniform=q_uniform,
        analytic_f_star_trace=q_uniform / q_star,
        analytic_f_star_logdet=float(logdet_c / d - np.log(q_star)),
        analytic_uniform_trace_gap=float(1.0 - q_uniform / q_star),
        analytic_uniform_logdet_gap=float(np.log(q_star / q_uniform)),
        minimum_optimal_weight=float(w_star.min()),
        reference_caution="Analytic geometry is exact; computed float64 f values still require independent numerical checks.",
    )
    return dict(
        covariance_id=f"geometry_{regime}_seed{seed:02d}",
        panel="geometry", regime=regime, d=d, s=s, seed=seed,
        A=a[:, None, None] * c, B=b[:, None, None] * c,
        means=rng.standard_normal((s, d)), w_star=w_star,
        geometric_expected=geometric_expected, metadata=metadata,
    )


def make_cases(n_seeds: int = 20, panels: Iterable[str] | None = None) -> list[dict]:
    """Return 480 objective cases by default, sharing 240 covariance inputs.

    ``panels`` may contain ``scaling`` and/or ``geometry``.  The common (25,5)
    scaling anchor is generated once and belongs to both sweep axes via its
    metadata.  The same arrays are intentionally shared by trace/logdet entries;
    solvers must not modify input matrices in place.
    """
    if n_seeds < 1:
        raise ValueError("n_seeds must be positive")
    chosen = set(("scaling", "geometry") if panels is None else panels)
    if chosen - {"scaling", "geometry"}:
        raise ValueError(f"Unknown panels: {chosen}")
    inputs = []
    if "scaling" in chosen:
        settings = [(d, 5) for d in DIMENSIONS]
        settings += [(25, s) for s in SENSOR_COUNTS if s != 5]
        for d, s in settings:
            inputs.extend(_scaling_input(d, s, seed) for seed in range(n_seeds))
    if "geometry" in chosen:
        for regime in REGIMES:
            inputs.extend(_mechanism_input(regime, seed) for seed in range(n_seeds))
    cases = []
    for item in inputs:
        for mode in ("trace", "logdet"):
            entry = dict(item)
            entry["mode"] = mode
            entry["case_id"] = item["covariance_id"] + "_" + mode
            entry["metadata"] = dict(item["metadata"])
            if item["w_star"] is not None:
                entry["analytic_f_star"] = item["metadata"]["analytic_f_star_" + mode]
            cases.append(entry)
    return cases


def save_inputs(cases: list[dict], destination: str | Path) -> None:
    """Save each covariance input once, plus its objective-case metadata."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifests = []
    seen = set()
    for case in cases:
        identity = case["covariance_id"]
        if identity not in seen:
            target = destination / (identity + ".npz")
            if target.exists():
                raise FileExistsError(target)
            arrays = {name: case[name] for name in ("A", "B", "means")}
            if case["w_star"] is not None:
                arrays["w_star"] = case["w_star"]
            np.savez_compressed(target, **arrays)
            seen.add(identity)
        entry = {k: v for k, v in case.items() if k not in ("A", "B", "means", "w_star")}
        entry["input_file"] = identity + ".npz"
        entry["has_analytic_weight"] = case["w_star"] is not None
        manifests.append(entry)
    target = destination / "cases.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(manifests, indent=2))
