"""Reproduce the stress panel and freeze the inputs used in its scale audit."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from benchmark_core import certificate, feasible, generate, reference

DATA = Path(__file__).resolve().parent.parent / 'data'


def collect_probes(model, mode, common, vertex_scale):
    center = np.ones(model.s) / model.s
    value, gradient, hessian, _ = model.oracle(center, mode, 2)
    rng = np.random.default_rng(377 + common['case'])
    weights = [*np.eye(model.s), *rng.dirichlet(np.full(model.s, .5), size=12)]
    probes = []
    for weight in weights:
        v = weight - center
        linear = value + gradient @ v
        curvature = .5 * (v @ hessian @ v)
        actual = model.oracle(weight, mode, 0)
        unit = max(1., abs(actual))
        cheap_violation = linear + curvature/model.s - actual
        vertex_violation = (linear + curvature/vertex_scale - actual
                            if vertex_scale is not None else np.nan)
        probes.append(dict(**common, mode=mode, lower_violation=cheap_violation,
                           relative_violation=cheap_violation/unit,
                           vertex_relative_violation=vertex_violation/unit))
    return probes


def evaluate(model, common, mode, legacy, rows, audit):
    try:
        ref = reference(model, mode)
    except Exception as error:
        rows.append(dict(**common, mode=mode, stage='reference', status='failure',
                         error=repr(error)))
        return None
    vertex_uniform = None
    for eta in (1., .001):
        center = feasible((1-eta)*ref['w'] + eta*np.ones(model.s)/model.s)
        for stage in ('cheap', 'vertex'):
            started = time.perf_counter()
            parameters = dict(**common, mode=mode, eta=eta, stage=stage)
            try:
                result = certificate(model, center, mode, stage, tol=1e-13,
                                     verify_scale=(stage == 'vertex'))
            except Exception as error:
                rows.append(dict(**parameters, status='failure', error=repr(error),
                                 seconds=time.perf_counter()-started))
                continue
            elapsed = time.perf_counter()-started
            diagnostics = dict(gfw=result['gfw'], gq=result['gq'], b=result['b'],
                               qp_gap=result['qp_gap'], reference_fw=ref['gfw'],
                               true_gap_estimate=result['f']-ref['f'])
            rows.append(dict(**parameters, status='success', **diagnostics,
                             scale_status=result['scale_status'], seconds=elapsed))
            if stage == 'vertex':
                if eta == 1.:
                    vertex_uniform = result['b']
                old = legacy.loc[(common['case'],mode,eta,stage)]
                audit.append(dict(**parameters, **diagnostics,
                                  weights=json.dumps(center.tolist()),
                                  anchor=json.dumps(result['z'].tolist()),
                                  b_hex=float(result['b']).hex(),
                                  scale_status=result['scale_status'],
                                  spectral_proposal=result['spectral_proposal'],
                                  legacy_status=old['status'], legacy_b=old['b'],
                                  legacy_gq=old['gq']))
    return vertex_uniform


def main():
    threadpool_limits(1)
    legacy = pd.read_csv(DATA/'stress_results_original.csv', float_precision='round_trip')
    legacy = legacy.set_index(['case','mode','eta','stage'])
    settings = [(d,s,cond,kind) for d,s in [(2,3),(4,6),(8,3),(8,6)]
                for cond in [1e4,1e8] for kind in
                ['singular','affine','near_independent','near_ci']]
    rows, audit, envelopes, frozen = [], [], [], {}
    for case, (d,s,cond,kind) in enumerate(settings):
        model = generate(d,s,cond,6173+case,kind)
        if kind in ('near_independent','near_ci'):
            alpha = 1e-14 if kind == 'near_independent' else 1-1e-14
            model.aas = [np.full(d,alpha) for _ in range(s)]
        common = dict(case=case,d=d,s=s,cond=cond,kind=kind)
        frozen[f'Z_{case}'] = np.stack(model.Zs)
        frozen[f'a_{case}'] = np.stack(model.aas)
        for mode in ('trace','logdet'):
            scale = evaluate(model,common,mode,legacy,rows,audit)
            envelopes.extend(collect_probes(model,mode,common,scale))
    pd.DataFrame(rows).to_csv(DATA/'stress_results.csv',index=False)
    pd.DataFrame(envelopes).to_csv(DATA/'stress_envelopes.csv',index=False)
    pd.DataFrame(audit).to_csv(DATA/'stress_vertex_audit.csv',index=False)
    np.savez_compressed(DATA/'stress_factor_inputs.npz',**frozen)
    print('Stress panel:',len(settings),'inputs,',len(rows),'evaluations,',
          len(envelopes),'paired lower-model probes.')
    print(pd.DataFrame(rows).groupby(['stage','status']).size())


if __name__ == '__main__':
    main()
