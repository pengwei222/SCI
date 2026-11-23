"""Damped active-set Newton on the SCI weight simplex.

A candidate generator, not a new statistical fusion model. Both stopping rules
use the same Newton/Armijo updates. `cheap` only adds the original global lower
model at the fixed gate. All returned success requires a computed bound <= tol.
Float64 diagnostics are not interval-arithmetic certificates.
"""
import time
import numpy as np
import original_core as original
from optimized_core import active_qp


def solve(model, tol=1e-3, rule='fw', maxiter=150, w0=None, history=False,
          gate_factor=4.0):
    if rule not in ('fw', 'cheap', 'vertex'):
        raise ValueError('Unknown certificate rule')
    if tol <= 0 or not np.isfinite(tol):
        raise ValueError('tol must be finite and positive')
    t0 = time.perf_counter()
    w = np.full(model.s, 1. / model.s) if w0 is None else original.simplex(w0)
    best_lower = -np.inf
    calls = checks = failures = hessian_calls = 0
    cert_time = 0.
    hist = []
    cache_w = cache = None
    fallback_used = False
    latest = None

    def evaluate(v, hessian=False):
        nonlocal cache_w, cache, calls, hessian_calls
        if cache_w is None or not np.array_equal(v, cache_w):
            cache_w = v.copy()
            cache = model.oracle(v, hessian)
            calls += 1
            hessian_calls += int(hessian)
        elif hessian and cache[2] is None:
            cache = model.oracle(v, True)
            hessian_calls += 1
        return cache

    status = 'maxiter_or_stall'
    for iteration in range(maxiter + 1):
        f, g, H, P = evaluate(w)
        checks += 1
        fw = max(0., float(g @ w - g.min()))
        lower = f - fw
        b = inner_gap = gq = np.nan
        ct = time.perf_counter()
        if rule != 'fw' and tol < f - max(best_lower, lower) <= gate_factor * tol:
            try:
                if rule == 'cheap':
                    if w.min() <= 0:
                        raise ArithmeticError('Cheap scale needs interior weights')
                    b = 1. / float(w.min())
                else:
                    b = model.vertex_b(w)
                H = evaluate(w, True)[2]
                eigenvalues = np.linalg.eigvalsh(H)
                if eigenvalues[0] < -1e-8 * max(1., abs(eigenvalues[-1])):
                    raise ArithmeticError('Negative numerical curvature')
                _, dual, inner_gap = active_qp(w, g, H / b)
                gq = max(0., -float(dual))
                if not np.isfinite(gq):
                    raise ArithmeticError('Nonfinite quadratic bound')
                lower = max(lower, f - gq)
            except (ArithmeticError, np.linalg.LinAlgError, ValueError):
                failures += 1
        cert_time += time.perf_counter() - ct
        best_lower = max(best_lower, lower)
        gap = max(0., float(f - best_lower))
        latest = dict(w=w.copy(), f=f, gap=gap, fw=fw, gq=gq, b=b,
                      inner_gap=inner_gap, P=P)
        if history:
            hist.append({k: v for k, v in latest.items() if k not in ('w', 'P')})
        if gap <= tol:
            status = 'bound_met'
            break
        if iteration == maxiter:
            break
        if H is None:
            H = evaluate(w, True)[2]
        # Regularization is used only to generate a descent candidate. The
        # original unregularized Hessian above defines the quadratic certificate.
        eigenvalues = np.linalg.eigvalsh(H)
        floor = max(1., float(np.max(np.abs(eigenvalues)))) * 1e-12
        ridge = max(0., floor - float(eigenvalues[0]))
        target, _, _ = active_qp(w, g, H + ridge * np.eye(model.s))
        direction = target - w
        slope = float(g @ direction)
        if not np.isfinite(slope) or slope >= -1e-14 * max(1., abs(f)):
            target = np.zeros(model.s)
            target[int(np.argmin(g))] = 1.
            direction = target - w
            slope = float(g @ direction)
        alpha = 1.
        accepted = False
        for _ in range(30):
            trial = original.simplex(w + alpha * direction)
            tf = evaluate(trial)[0]
            if np.isfinite(tf) and tf <= f + 1e-4 * alpha * slope:
                w = trial
                accepted = True
                break
            alpha *= .5
        if not accepted:
            status = 'line_search_stall'
            break

    # Hard-to-condition or singular/flat cases retain the existing solver as a
    # safe fallback. It starts at the current feasible point, pays all costs,
    # and independently checks its numerical bound at the identical tolerance.
    if latest['gap'] > tol:
        from optimized_core import solve as slsqp_solve
        remaining = max(20, maxiter)
        fallback_used = True
        extra = slsqp_solve(model, tol=tol, rule=rule, maxiter=remaining,
                            w0=latest['w'], history=history,
                            gate_factor=gate_factor)
        calls += extra['oracle_calls']
        checks += extra['checks']
        failures += extra['certificate_failures']
        cert_time += extra['certificate_seconds']
        if history:
            hist += extra['history']
        latest = {k: extra[k] for k in ('w','f','gap','fw','gq','b','inner_gap','P')}
        status = 'fallback_' + extra['status']
    latest.update(seconds=time.perf_counter() - t0, certificate_seconds=cert_time,
                  oracle_calls=calls, checks=checks, certificate_failures=failures,
                  converged=latest['gap'] <= tol, status=status, history=hist,
                  hessian_calls=hessian_calls, fallback_used=fallback_used)
    return latest
