"""Independent direct-matrix checks for the expanded SCI experiments.

All branch matrices are evaluated by Cholesky solves of A_i + w_i B_i.
The spectral whitening used by the timed implementations is not reused.
Returned intervals are float64 diagnostics, NOT outward-rounded certificates.
No solver status alone is treated as evidence of the requested accuracy.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def _sym(x):
    return (x + x.T) * .5


def _inverse(x):
    factor = cho_factor(_sym(x), lower=True, check_finite=True)
    return _sym(cho_solve(factor, np.eye(x.shape[0]), check_finite=False))


def _weights(case, d):
    if 'W' in case:
        W = _sym(np.asarray(case['W'], dtype=float))
        if W.shape != (d, d):
            raise ValueError('W has the wrong shape')
        return W
    W = np.eye(d)
    pose = case.get('pose_dimension')
    length = float(case.get('heading_length', 1.))
    if pose is not None:
        W[int(pose):, int(pose):] = 0.
        use = int(pose)
    else:
        use = d
    idx = np.arange(2, use, 3)
    W[idx, idx] = length ** 2
    return W


def _inputs(case):
    A, B = np.asarray(case['A'], dtype=float), np.asarray(case['B'], dtype=float)
    if A.ndim != 3 or A.shape != B.shape or A.shape[1] != A.shape[2]:
        raise ValueError('A and B must have shape (s,d,d)')
    mode = case.get('mode', 'trace')
    if mode not in ('trace', 'logdet'):
        raise ValueError('Unknown objective mode')
    return A, B, mode


def _raw(case, w, with_hessian=False):
    A, B, mode = _inputs(case)
    s, d, _ = A.shape
    w = np.asarray(w, dtype=float)
    if w.shape != (s,) or not np.isfinite(w).all():
        raise ValueError('Invalid weight vector')
    C = np.array([_inverse(a + t*b) for a, b, t in zip(A, B, w)])
    Ji = w[:, None, None] * C
    P = _inverse(Ji.sum(axis=0))
    D = np.array([_sym(c @ a @ c) for c, a in zip(C, A)])
    if mode == 'trace':
        W = _weights(case, d)
        f = float(np.sum(W.T * P))
        G = P @ W @ P
    else:
        W = None
        lp = np.linalg.cholesky(P)
        f = float(2*np.log(np.diag(lp)).sum())
        G = P
    g = -np.einsum('ij,sji->s', G, D)
    result = dict(f=f, g=g, P=P, Ji=Ji)
    if with_hessian:
        # J_i'' = -C_i B_i D_i - D_i B_i C_i; no spectral factors.
        F = np.array([-_sym(c @ b @ di + di @ b @ c)
                      for c, b, di in zip(C, B, D)])
        L = np.linalg.cholesky(P)
        if mode == 'trace':
            vals, vectors = np.linalg.eigh(W)
            if vals.min() < -1e-12 * max(1., abs(vals).max()):
                raise ValueError('Trace weight is not positive semidefinite')
            R = vectors[:, vals > 0] * np.sqrt(vals[vals > 0])
            Y = np.array([L.T @ di @ P @ R for di in D]).reshape(s, -1)
            H = 2 * (Y @ Y.T)
        else:
            Y = np.array([L.T @ di @ L for di in D]).reshape(s, -1)
            H = Y @ Y.T
        H -= np.diag(np.einsum('ij,sji->s', G, F))
        result['H'] = _sym(H)
    return result


def normalization(case):
    """Uniform full/pose-weighted trace, or state dimension for logdet."""
    A, _, mode = _inputs(case)
    s, d, _ = A.shape
    scale = _raw(case, np.full(s, 1/s), False)['f'] if mode == 'trace' else float(d)
    if not np.isfinite(scale) or scale <= 0:
        raise ArithmeticError('Invalid objective normalization')
    return float(scale)


def evaluate(case, w, scale=None, with_hessian=False):
    """Compute direct f, gradient, covariance, and a simplex FW gap.

    ``w`` is not projected or otherwise altered: callers can inspect
    ``simplex_error`` separately. There is no retained cache or solver state.
    ``Ji`` and ``P`` are raw covariance quantities; f, g, H are normalized.
    """
    if scale is None:
        scale = normalization(case)
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be finite and positive')
    r = _raw(case, w, with_hessian)
    r['f'] /= scale
    r['g'] /= scale
    if with_hessian:
        r['H'] /= scale
    w = np.asarray(w, dtype=float)
    r['scale'] = scale
    r['raw_fw_gap'] = float(r['g'] @ w - r['g'].min())
    r['fw_gap'] = max(0., r['raw_fw_gap'])
    r['simplex_error'] = float(max(abs(w.sum()-1.), max(0., -w.min())))
    if 'means' in case:
        means = np.asarray(case['means'], dtype=float)
        if means.shape != (len(w), len(r['P'])):
            raise ValueError('means has the wrong shape')
        r['fused'] = r['P'] @ np.einsum('sij,sj->i', r['Ji'], means)
    return r


def _simplex(w):
    w = np.maximum(np.asarray(w, dtype=float), 0.)
    if not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError('No finite feasible candidate')
    return w / w.sum()


def reference(case, min_tol=1e-5, target_gap=1e-9, maxiter=300):
    """Construct and independently assess a numerical reference interval.

    Uniform, supplied analytic/reference weights, and a high-accuracy fast
    Newton candidate are checked by the direct oracle. If needed, direct
    SLSQP and independent active-face KKT polishing refine the candidate.
    ``verified`` means the numerical interval width is <= min_tol/20; it does
    not mean exact/interval-arithmetic certification. Failures remain in the
    returned status and diagnostics.
    """
    if min_tol <= 0 or target_gap <= 0:
        raise ValueError('Positive tolerances required')
    t0 = time.perf_counter()
    A, B, mode = _inputs(case)
    s = len(A)
    scale = normalization(case)
    desired = min(float(target_gap), float(min_tol)/20.)
    required = float(min_tol)/20.
    best_f, best_lower, best_w = np.inf, -np.inf, None
    best_direct_fw = np.inf
    observations, errors, statuses = [], [], []
    evals = 0

    def direct(w, hessian=False):
        nonlocal evals
        evals += 1
        return evaluate(case, w, scale, hessian)

    def record(w, origin):
        nonlocal best_f, best_lower, best_w, best_direct_fw
        w = _simplex(w)
        r = direct(w)
        lower = r['f'] - r['fw_gap']
        if r['f'] < best_f or (abs(r['f']-best_f) <= 2e-15*max(1., abs(best_f))
                              and r['fw_gap'] < best_direct_fw):
            # Retain a near-identical objective with the sharper KKT residual.
            best_f, best_w, best_direct_fw = r['f'], w.copy(), r['fw_gap']
        best_lower = max(best_lower, lower)
        observations.append(dict(origin=origin, f=r['f'], fw_gap=r['fw_gap']))
        return r

    record(np.full(s, 1/s), 'uniform')
    for key in ('w_star', 'ref_w'):
        if key in case and case[key] is not None:
            try:
                record(case[key], key)
            except Exception as exc:
                errors.append(f'{key}: {type(exc).__name__}: {exc}')
    if best_f - best_lower > desired and 'W' not in case:
        try:
            from fast_core import FastSCI
            from newton_solver import solve
            m = FastSCI(A, B, mode, case.get('heading_length', 1.),
                        pose_dimension=case.get('pose_dimension'))
            r = solve(m, tol=min(1e-11, desired*.01), rule='fw', maxiter=maxiter)
            record(r['w'], 'fast_candidate_direct_checked')
            statuses.append('fast:' + str(r.get('status', '')))
        except Exception as exc:
            errors.append(f'fast_candidate: {type(exc).__name__}: {exc}')

    if best_f - best_lower > desired:
        starts = [best_w.copy(), np.full(s, 1/s)]
        for start_id, start in enumerate(starts):
            try:
                res = minimize(lambda w: direct(w)['f'], start,
                               jac=lambda w: direct(w)['g'], method='SLSQP',
                               bounds=[(0., 1.)]*s,
                               constraints={'type':'eq', 'fun':lambda w:w.sum()-1.,
                                            'jac':lambda w:np.ones_like(w)},
                               options={'ftol':1e-15, 'maxiter':maxiter, 'disp':False})
                record(res.x, f'direct_slsqp_{start_id}')
                statuses.append('slsqp:' + str(res.message))
            except Exception as exc:
                errors.append(f'direct_slsqp: {type(exc).__name__}: {exc}')
            if best_f - best_lower <= desired:
                break

    # Polish the KKT residual, which can remain larger than the objective
    # termination criterion of SLSQP even when its primal value is accurate.
    w = best_w.copy()
    for k in range(30):
        if best_f - best_lower <= desired:
            break
        try:
            r = direct(w, True)
            f, g, H = r['f'], r['g'], r['H']
            support = np.flatnonzero(w > 1e-12)
            j = int(np.argmin(g))
            if j not in support:
                support = np.unique(np.r_[support, j])
            K = np.zeros((len(support)+1, len(support)+1))
            K[:-1, :-1] = H[np.ix_(support, support)]
            K[:-1, -1] = K[-1, :-1] = 1.
            rhs = np.r_[-g[support], 0.]
            sol = np.linalg.lstsq(K, rhs, rcond=1e-13)[0][:-1]
            p = np.zeros(s); p[support] = sol
            slope = float(g @ p)
            if not np.isfinite(p).all() or slope >= 0:
                p = -w.copy(); p[j] += 1.; slope = float(g @ p)
            bad = p < 0
            alpha = min(1., float(np.min(-w[bad]/p[bad]))) if bad.any() else 1.
            if alpha <= 0:
                p = -w.copy(); p[j] += 1.; slope = float(g @ p); alpha = 1.
            accepted = False
            slack = 8*np.finfo(float).eps*max(1., abs(f))
            for _ in range(35):
                trial = _simplex(w + alpha*p)
                tr = direct(trial)
                if tr['f'] <= f + 1e-4*alpha*slope + slack:
                    w = trial; record(w, f'direct_kkt_{k}'); accepted = True; break
                alpha *= .5
            if not accepted:
                statuses.append('direct_kkt_line_search_stall'); break
        except Exception as exc:
            errors.append(f'direct_kkt: {type(exc).__name__}: {exc}'); break

    final = direct(best_w)
    raw_lower = float(best_lower)
    crossing = max(0., raw_lower - best_f)
    # Explicit engineering allowance: this is not a proved roundoff enclosure.
    margin = 64*np.finfo(float).eps*max(1., abs(best_f), np.linalg.norm(final['g'], 1))
    lower = min(best_f, raw_lower) - margin
    gap = max(0., best_f - lower)
    verified = bool(np.isfinite(gap) and gap <= required and crossing <= margin)
    return dict(ref_f=float(best_f), lower=float(lower), gap=float(gap),
                raw_lower=raw_lower, roundoff_crossing=float(crossing),
                engineering_margin=float(margin), w=best_w.copy(), scale=scale,
                direct_fw_gap=float(final['fw_gap']), target_gap=desired,
                required_gap=required, verified=verified,
                status=('reference_verified' if verified else 'reference_unresolved'),
                target_met=bool(gap <= desired), evaluations=evals,
                seconds=time.perf_counter()-t0, solver_statuses=statuses,
                errors=errors, observations=observations,
                geometry_min_weight=float(best_w.min()),
                geometry_active_count=int(np.count_nonzero(best_w > 1e-8)),
                exact_arithmetic=False)
