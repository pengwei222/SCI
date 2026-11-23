"""Read-only post-run audit; never invoke while formal timing is active.

Reconstructs coverage, success flags, repetition reduction and paired statistics.
Selected fixed candidate weights are reevaluated by direct dense matrix solves.
No optimization or timing is run. Float64 checks are not interval verification.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from fast_core import FastSCI
import original_core

threadpool_limits(1)
ROOT = Path(__file__).resolve().parent
EXPECTED = {'synthetic': (480, 17280), 'recorded_existing': (1246, 104664),
            'recorded_additional': (724, 60816)}
SOURCES = ('original_core.py', 'optimized_core.py', 'fast_core.py', 'newton_solver.py',
           'synthetic_inputs.py', 'audit_reference.py', 'run_expanded.py')
CONTRASTS = {'matrix': ('blas-fw', 'original-fw'), 'newton': ('newton-fw', 'blas-fw'),
             'certificate': ('newton-cheap', 'newton-fw'), 'complete': ('newton-cheap', 'original-fw')}


def read_json(path):
    return json.loads(Path(path).read_text())


def json_safe(value):
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def booleans(column):
    if column.dtype == bool:
        return column
    values = column.astype(str).str.lower().str.strip()
    assert values.isin(['true', 'false']).all(), ('invalid boolean', column.name)
    return values.eq('true')


def close(a, b, label, rtol=1e-11, atol=1e-13):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True), (label, float(np.nanmax(abs(a-b))))


def direct_fixed_weight(case, weights):
    """Independent numpy.solve formula; no spectral whitening or reference helper."""
    A, B = case['A'], case['B']
    s, d, _ = A.shape
    identity = np.eye(d)
    inverse = lambda matrix: np.linalg.solve(matrix, identity)
    sym = lambda matrix: (matrix + matrix.T) / 2
    C = np.array([sym(inverse(a+w*b)) for a, b, w in zip(A, B, weights)])
    information = np.sum(weights[:, None, None] * C, axis=0)
    P = sym(inverse(information))
    D = np.array([sym(c@a@c) for c, a in zip(C, A)])
    if case['mode'] == 'logdet':
        raw = np.linalg.slogdet(P)
        assert raw[0] > 0
        f = raw[1]/d
        gradient = -np.einsum('ij,sji->s', P, D)/d
        scale = float(d)
    else:
        W = np.eye(d)
        pose = case.get('pose_dimension')
        use = d if pose is None else int(pose)
        if pose is not None:
            W[use:, use:] = 0
        ids = np.arange(2, use, 3)
        W[ids, ids] = float(case.get('heading_length', 1.))**2
        if 'W' in case:
            W = case['W']
        uniform_C = np.array([inverse(a+b/s) for a, b in zip(A, B)])
        uniform_P = inverse(np.sum(uniform_C, axis=0)/s)
        # Normalize by the same full/pose-weighted trace at uniform weights.
        scale = float(np.einsum('ij,ji', W, uniform_P))
        f = float(np.einsum('ij,ji', W, P)/scale)
        gradient = -np.einsum('ij,sji->s', P@W@P, D)/scale
    fused = P@np.sum((weights[:, None, None]*C)@case['means'][..., None], axis=0)[:, 0]
    return dict(f=f, g=gradient, P=P, fused=fused, scale=scale,
                fw_gap=float(gradient@weights-gradient.min()))


def load_input(case_id):
    with np.load(ROOT/'inputs'/(case_id+'.npz'), allow_pickle=False) as data:
        case = json.loads(str(data['metadata']))
        case.update({key: data[key].copy() for key in data.files if key != 'metadata'})
    return case


def audit_one(suite, expected_inputs, expected_rows):
    folder = ROOT/'results'/suite
    for filename in ['completion.json', 'manifest.json', 'references.json', 'timing.csv',
                     'per_case.csv', 'summary.csv', 'paired.csv', 'paired_summary.csv', 'quality.json']:
        assert (folder/filename).is_file(), ('missing completed output', suite, filename)
    manifest, completion = read_json(folder/'manifest.json'), read_json(folder/'completion.json')
    assert completion['sources_unchanged']
    assert set(manifest['source_hashes']) == set(SOURCES)
    for name, digest in manifest['source_hashes'].items():
        assert sha256(ROOT/name) == digest, ('source hash', suite, name)
    for case_id, digest in manifest['input_sha256'].items():
        assert sha256(ROOT/'inputs'/(case_id+'.npz')) == digest, ('input hash', suite, case_id)
    frame = pd.read_csv(folder/'timing.csv')
    references = pd.DataFrame(read_json(folder/'references.json')).set_index('case_id')
    assert references.index.is_unique
    assert set(references.index) == set(manifest['input_sha256'])
    assert len(frame) == expected_rows == completion['rows']
    assert frame.case_id.nunique() == expected_inputs == manifest['inputs'] == completion['inputs']
    assert manifest['repeats'] == 3
    assert set(manifest['methods']) == {'original-fw', 'blas-fw', 'newton-fw', 'newton-cheap'}
    index_names = ['case_id', 'method', 'tolerance', 'repeat']
    assert not frame.duplicated(index_names).any()
    expected_index = pd.MultiIndex.from_product([sorted(manifest['input_sha256']), manifest['methods'],
        manifest['tolerances'], range(manifest['repeats'])], names=index_names)
    actual_index = pd.MultiIndex.from_frame(frame[index_names])
    assert not len(expected_index.difference(actual_index))
    assert not len(actual_index.difference(expected_index))
    flags = ['success', 'converged', 'reference_resolved', 'accuracy_upper_confirmed',
             'bound_violation', 'timed_solve_completed']
    for flag in flags:
        frame[flag] = booleans(frame[flag])
    reference_resolved = (references.verified.astype(bool)
        & pd.to_numeric(references.gap, errors='coerce').le(min(manifest['tolerances'])/20))
    assert np.array_equal(frame.reference_resolved, frame.case_id.map(reference_resolved))
    assert (references.loc[reference_resolved, 'gap'] >= 0).all()
    tolerance = frame.tolerance.to_numpy()
    slack = np.maximum(1e-10, 1e-7*tolerance)
    lower = np.maximum(0., frame.direct_objective-frame.case_id.map(references.ref_f))
    upper = np.maximum(0., frame.direct_objective-frame.case_id.map(references.lower))
    present = frame.direct_objective.notna() & frame.audit_error_upper.notna()
    close(frame.loc[present, 'audit_error_lower'], lower[present], 'error lower')
    close(frame.loc[present, 'audit_error_upper'], upper[present], 'error upper')
    close(frame.loc[present, 'oracle_f_error'],
          abs(frame.loc[present, 'objective']-frame.loc[present, 'direct_objective']), 'oracle f error')
    assert np.array_equal(frame.loc[present, 'accuracy_upper_confirmed'],
        (upper <= tolerance+slack)[present])
    assert np.array_equal(frame.loc[present, 'bound_violation'],
        (frame.reported_gap+slack < lower)[present])
    fresh_success = (frame.converged & frame.reference_resolved & frame.timed_solve_completed
        & (frame.audit_error_upper <= tolerance+slack)
        & (frame.oracle_f_error <= np.maximum(slack, 1e-8))
        & (frame.covariance_relative_error <= 1e-6)
        & (frame.fused_relative_error <= 1e-6) & (frame.simplex_error <= 1e-10)
        & (frame.reported_gap+slack >= frame.audit_error_lower))
    assert np.array_equal(frame.success, fresh_success)
    assert (frame.loc[frame.success, 'reported_gap'] <=
        (tolerance+slack)[frame.success]).all()
    # Feasibility is recomputed from every returned vector, not trusted from the flag.
    weight_errors = []
    for row in frame[['weights', 's', 'simplex_error']].itertuples(index=False):
        weights = json.loads(row.weights)
        if not weights:
            weight_errors.append(np.nan)
            continue
        assert len(weights) == row.s and all(np.isfinite(weights))
        weight_errors.append(max(abs(sum(weights)-1), max(0., -min(weights))))
    mask = frame.simplex_error.notna()
    close(frame.loc[mask, 'simplex_error'], np.asarray(weight_errors)[mask], 'simplex error', atol=1e-15)
    keys = ['case_id', 'method', 'tolerance']
    time_metrics = ['end_to_end_seconds','construction_seconds','solver_seconds','fusion_seconds',
                    'oracle_solver_seconds','certificate_seconds','checks','oracle_calls',
                    'certificate_attempts','hessian_requests','audit_error_upper']
    fresh = frame.groupby(keys)[time_metrics].median()
    fresh['success'] = frame.groupby(keys).success.all()
    fresh['reference_resolved'] = frame.groupby(keys).reference_resolved.all()
    fresh['timing_complete'] = frame.groupby(keys).timed_solve_completed.all()
    fresh['repeats'] = frame.groupby(keys)['repeat'].size()
    saved = pd.read_csv(folder/'per_case.csv').set_index(keys)
    assert saved.index.is_unique and set(saved.index) == set(fresh.index)
    saved = saved.loc[fresh.index]
    for metric in time_metrics+['repeats']:
        close(saved[metric], fresh[metric], 'per-case '+metric)
    for metric in ['success','reference_resolved','timing_complete']:
        assert np.array_equal(booleans(saved[metric]), fresh[metric])
    attributes = ['panel','regime','d','s','seed','mode','dataset','epoch']
    for attr in attributes:
        assert frame.groupby('case_id')[attr].nunique(dropna=False).max() == 1
        from_raw = frame.drop_duplicates('case_id').set_index('case_id')[attr]
        assert np.array_equal(saved[attr].to_numpy(),
            saved.reset_index().case_id.map(from_raw).to_numpy()), ('case metadata', attr)
    with_attributes = fresh.reset_index().merge(
        frame[['case_id']+attributes].drop_duplicates('case_id'), on='case_id', validate='many_to_one')
    aggregate_keys = ['panel','regime','d','s','mode','dataset','method','tolerance']
    aggregate = pd.read_csv(folder/'summary.csv').set_index(aggregate_keys)
    assert aggregate.index.is_unique
    assert len(aggregate) == with_attributes.groupby(aggregate_keys, dropna=False).ngroups
    for key, group in with_attributes.groupby(aggregate_keys, dropna=False):
        row = aggregate.loc[key]
        assert row.cases == len(group)
        assert row.successes == int(group.success.sum())
        assert row.complete_timings == int(group.timing_complete.sum())
        assert row.reference_unresolved == int((~group.reference_resolved).sum())
        for metric in time_metrics:
            close(row['median_'+metric], group[metric].median(), 'summary '+metric)
        for quantile in (.1,.9,.95,.99):
            close(row[f'p{round(quantile*100)}_seconds'], group.end_to_end_seconds.quantile(quantile),
                  'summary quantile')
    paired = pd.read_csv(folder/'paired.csv')
    paired.paired_success = booleans(paired.paired_success)
    assert not paired.duplicated(['case_id','tolerance','contrast']).any()
    assert len(paired) == expected_inputs*len(manifest['tolerances'])*4
    pairs_report = []
    for contrast, (new, base) in CONTRASTS.items():
        a, b = fresh.xs(new,level='method'), fresh.xs(base,level='method')
        pp = paired[paired.contrast.eq(contrast)].set_index(['case_id','tolerance'])
        assert set(pp.index) == set(a.index) == set(b.index)
        pp = pp.loc[a.index]
        ratios = b.end_to_end_seconds/a.end_to_end_seconds
        valid = a.success & b.success
        close(pp.paired_speedup, ratios, 'paired speedup')
        close(pp.checks_saved, b.checks-a.checks, 'checks saved')
        assert np.array_equal(pp.paired_success, valid)
        pairs_report.append(dict(contrast=contrast, attempted_pairs=len(valid),
            successful_pairs=int(valid.sum()), median_successful_speedup=float(ratios[valid].median())))
    summary_keys = ['panel','regime','d','s','mode','dataset','tolerance','contrast']
    summary = pd.read_csv(folder/'paired_summary.csv').set_index(summary_keys)
    assert summary.index.is_unique
    assert len(summary) == paired.groupby(summary_keys, dropna=False).ngroups
    for key, group in paired.groupby(summary_keys, dropna=False):
        row, valid = summary.loc[key], group[group.paired_success]
        assert row.cases == len(group) and row.paired_successes == len(valid)
        for name, value in [('median_speedup',valid.paired_speedup.median()),
            ('p10_speedup',valid.paired_speedup.quantile(.1)),
            ('p90_speedup',valid.paired_speedup.quantile(.9)),
            ('fraction_faster',(valid.paired_speedup>1).mean()),
            ('fraction_checks_saved',(valid.checks_saved>0).mean()),
            ('mean_checks_saved',valid.checks_saved.mean())]:
            close(row[name], value, 'paired summary '+name)
    quality = read_json(folder/'quality.json')
    assert quality['rows'] == len(frame) and quality['unique_inputs'] == expected_inputs
    assert quality['successes'] == int(frame.success.sum())
    assert quality['failed'] == int((~frame.success).sum())
    assert quality['reference_unresolved'] == int((~frame.reference_resolved).sum())
    report = dict(suite=suite, objective_cases=expected_inputs, raw_rows=len(frame),
        successful_rows=int(frame.success.sum()), failed_rows=int((~frame.success).sum()),
        successful_case_method_tolerance_groups=int(fresh.success.sum()),
        attempted_case_method_tolerance_groups=len(fresh),
        cases_successful_at_all_methods_tolerances_repeats=int(frame.groupby('case_id').success.all().sum()),
        unresolved_reference_cases=int((~reference_resolved).sum()),
        incomplete_timed_runs=int((~frame.timed_solve_completed).sum()),
        nonfinite_total_times=int((~np.isfinite(frame.end_to_end_seconds)).sum()),
        bound_violations=int(frame.bound_violation.sum()),
        paired_contrasts=pairs_report, source_hashes=manifest['source_hashes'])
    adapter = folder/'untimed_input_adapter.json'
    if adapter.is_file():
        adapter_info = read_json(adapter)
        assert adapter_info['script_sha256'] == sha256(ROOT/'run_recorded_cached.py')
        assert adapter_info['inputs'] == expected_inputs
        report['untimed_adapter_hash_verified'] = True
    frame['suite'] = suite
    return report, frame


def sample_direct(frame):
    # First and last case IDs in every structural stratum, independent of outcomes.
    attrs = ['suite','panel','regime','d','s','mode','dataset']
    cases = frame.sort_values('case_id').drop_duplicates('case_id')
    selected = []
    for _, group in cases.groupby(attrs, dropna=False, sort=True):
        selected.extend(group.iloc[[0, len(group)-1]].case_id.drop_duplicates().tolist())
    observations = []
    for case_id in selected:
        candidates = frame[frame.case_id.eq(case_id)]
        candidates = candidates[candidates.tolerance.eq(candidates.tolerance.min()) & candidates['repeat'].eq(0)]
        case = load_input(case_id)
        assert np.min(np.linalg.eigvalsh(case['A'])) > 0
        b_eigenvalues = np.linalg.eigvalsh(case['B'])
        assert b_eigenvalues.min() >= -1e-11*max(1., abs(b_eigenvalues).max())
        for row in candidates.itertuples(index=False):
            weights = np.array(json.loads(row.weights))
            if not len(weights):
                observations.append(dict(case_id=case_id, method=row.method, skipped='no returned weights'))
                continue
            direct = direct_fixed_weight(case, weights)
            cls = original_core.SCI if row.method == 'original-fw' else FastSCI
            oracle = cls(case['A'],case['B'],case['mode'],case.get('heading_length',1.),
                         pose_dimension=case.get('pose_dimension'))
            answer = oracle.oracle(weights, hessian=False)
            x, P = oracle.fuse(weights, case['means'])
            # oracle tuple begins with f, gradient; use fixed candidate only.
            f, gradient = answer[:2]
            objective_error = abs(direct['f']-row.objective)
            recorded_direct_error = abs(direct['f']-row.direct_objective)
            covariance_error = np.linalg.norm(P-direct['P'])/max(np.linalg.norm(direct['P']),1e-30)
            fused_error = np.linalg.norm(x-direct['fused'])/max(np.linalg.norm(direct['fused']),1e-30)
            gradient_error = np.linalg.norm(gradient-direct['g'])/max(np.linalg.norm(direct['g']),1e-30)
            fw_error = abs(max(0., direct['fw_gap'])-row.fw_gap)
            assert objective_error <= 1e-8 and recorded_direct_error <= 1e-8
            assert covariance_error <= 1e-6 and fused_error <= 1e-6
            assert gradient_error <= 1e-6 and fw_error <= 1e-8
            close(f, row.objective, 'fixed-weight objective', atol=1e-9)
            observations.append(dict(case_id=case_id, suite=row.suite, panel=row.panel,
                regime=row.regime, mode=row.mode, method=row.method,
                objective_error=objective_error, recorded_direct_error=recorded_direct_error,
                covariance_relative_error=float(covariance_error), fused_relative_error=float(fused_error),
                gradient_relative_error=float(gradient_error), fw_gap_error=float(fw_error),
                sampled_fw_gap=direct['fw_gap']))
    return dict(selection='First and last lexicographic case IDs in each suite/panel/regime/d/s/mode/dataset stratum; strictest tolerance; repeat 0; all four methods. No outcome filtering.',
                unique_case_count=len(selected), candidate_count=len(observations), observations=observations)


def main():
    for name, digest in read_json(ROOT/'FROZEN_PROTOCOL_SHA256.json').items():
        assert sha256(ROOT/name) == digest, ('frozen protocol', name)
    reports, frames = [], []
    for suite, (cases, rows) in EXPECTED.items():
        report, frame = audit_one(suite, cases, rows)
        reports.append(report)
        frames.append(frame)
        print('Record audit passed:', suite, len(frame), flush=True)
    assert all(report['source_hashes'] == reports[0]['source_hashes'] for report in reports)
    all_rows = pd.concat(frames, ignore_index=True)
    assert len(all_rows) == 182760 and all_rows.case_id.nunique() == 2450
    assert not all_rows.duplicated(['case_id','method','tolerance','repeat']).any()
    geometry = []
    grouping = ['suite','panel','regime','mode','dataset']
    for key, group in all_rows.groupby(grouping, dropna=False):
        item = dict(zip(grouping, key))
        item.update(rows=len(group), successful_rows=int(group.success.sum()),
            objective_cases=group.case_id.nunique(),
            fully_successful_cases=int(group.groupby('case_id').success.all().sum()))
        for name in ['reference_gap','audit_error_upper','oracle_f_error','simplex_error',
                     'covariance_relative_error','fused_relative_error']:
            item['max_'+name] = float(group[name].max())
        geometry.append(item)
    sample = sample_direct(all_rows)
    report = dict(status='passed', numerical_not_interval=True,
        scope='All raw rows and saved summaries; stratified fixed-weight direct matrix sample. No solver retiming.',
        total_objective_cases=2450, total_raw_rows=182760, suites=reports,
        quality_by_geometry=geometry, independent_direct_sample=sample)
    (ROOT/'INDEPENDENT_AUDIT.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False))
    sample_frame = pd.DataFrame(sample['observations'])
    lines = ['# Independent expanded-results audit', '',
        'Status: **passed**. All checks use float64; this is numerical validation, not rigorous interval verification.', '',
        'No optimization or retiming was performed. Source and input hashes, Cartesian run coverage, success definitions, per-input repetition medians and paired denominators were reconstructed from the raw records.', '',
        '| Suite | Objective cases | Raw runs | Successful runs | Fully successful cases |',
        '|---|---:|---:|---:|---:|']
    for item in reports:
        lines.append(f"| {item['suite']} | {item['objective_cases']} | {item['raw_rows']} | {item['successful_rows']} | {item['cases_successful_at_all_methods_tolerances_repeats']} |")
    lines += ['', f"Independent direct sample: {sample['unique_case_count']} inputs and {sample['candidate_count']} fixed candidates.",
        sample['selection'], '', 'Maximum sampled discrepancies:', '']
    for metric in ['objective_error','recorded_direct_error','covariance_relative_error','fused_relative_error','gradient_relative_error','fw_gap_error']:
        lines.append(f"- {metric}: {sample_frame[metric].max():.6g}")
    lines += ['', '## Quality by geometry', '',
        '| Suite / regime / criterion | Cases passed / attempted | Max reference width | Max objective disagreement | Max covariance relative error | Max fused relative error |',
        '|---|---:|---:|---:|---:|---:|']
    for item in geometry:
        name = f"{item['suite']} / {item['regime']} / {item['mode']} / D{item['dataset']}"
        lines.append(f"| {name} | {item['fully_successful_cases']} / {item['objective_cases']} | {item['max_reference_gap']:.4g} | {item['max_oracle_f_error']:.4g} | {item['max_covariance_relative_error']:.4g} | {item['max_fused_relative_error']:.4g} |")
    lines += ['', 'All failure and unresolved-reference counts remain in the JSON report. Timing repetitions are not independent samples; trace/logdet share covariance sets; trajectory time points and initialization variants are correlated. Paired summaries condition on both methods succeeding, with attempted and valid denominators checked explicitly.', '',
        'Fixed candidate weights were reevaluated by numpy.linalg.solve on each branch, independently of the Cholesky reference and spectral timed implementations. Timed output covariance/mean arrays are not stored in CSV; sampled output comparisons reevaluate the frozen implementation at its saved weight, without rerunning optimization. Complete per-row output checks remain those recorded by the benchmark harness.', '']
    (ROOT/'INDEPENDENT_AUDIT.md').write_text('\n'.join(lines))
    print('Wrote INDEPENDENT_AUDIT.json and INDEPENDENT_AUDIT.md', flush=True)


if __name__ == '__main__':
    main()
