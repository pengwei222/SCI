"""Reproduce two further recorded sequences with the unchanged frozen Round4 model.

No calibration path exists here. Replay preserves the historical reference-based
feedback; export selects epoch % 5 == 0, time >= 30 s, without outcome filtering.
Formal solver timing must be run separately with no concurrent numerical load.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / 'frozen_source' / 'round4_motion'
SOURCE = FROZEN / 'source_motion'
PROTOCOL = ROOT / 'EXTENSION_PROTOCOL.json'
DATASETS = [4, 5]
SEED = 42
SELECTED_KEYS = ['epoch', 'time', 'A', 'B', 'means', 'truth', 'ref_w', 'ref_f',
                 'ref_gap', 'ref_converged', 'ref_mean', 'ref_P', 'bias_mean',
                 'bias_P', 'central_mean', 'central_P', 'accepted', 'rejected',
                 'downweighted', 'reference_seconds', 'reference_checks']


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def verify_frozen():
    old = json.loads((FROZEN / 'FROZEN_ROUND4_PROTOCOL.json').read_text())
    paths = [FROZEN / 'calibration_response.json', SOURCE / 'configs/default.json',
             SOURCE / 'run.py', *sorted((SOURCE / 'sciexp').glob('*.py'))]
    hashes = {}
    for path in paths:
        historical_name = 'round4_motion/' + path.relative_to(FROZEN).as_posix()
        actual = digest(path)
        if actual != old['hashes'][historical_name]:
            raise ValueError(f'Frozen historical source changed: {historical_name}')
        hashes[historical_name] = actual
    config = json.loads((SOURCE / 'configs/default.json').read_text())
    calibration = json.loads((FROZEN / 'calibration_response.json').read_text())
    assert config['training_dataset'] == calibration['training_dataset'] == 1
    assert config['validation_datasets'] == [2]
    assert config['seeds'] == [SEED]
    assert config['benchmark_stride'] == 5 and config['burn_in_seconds'] == 30
    assert config['reference_tolerance'] == 1e-7 and config['max_iterations'] == 150
    if PROTOCOL.exists():
        extension = json.loads(PROTOCOL.read_text())
        assert extension['frozen_source_sha256'] == hashes
        assert extension['datasets'] == DATASETS and extension['seeds'] == [SEED]
    return hashes, config


def freeze():
    hashes, config = verify_frozen()
    if PROTOCOL.exists():
        print('Existing extension protocol verified; not rewritten.')
        return
    obj = dict(
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Additional recorded sequence validation of the unchanged model and solver inputs',
        datasets=DATASETS, seeds=[SEED],
        split=dict(training=1, development=2, previous_holdout=3,
                   additional_recorded_sequences=DATASETS),
        unit_of_independent_recording='MRCLAM Dataset number; epochs and seeds are not independent sequences',
        model='Unchanged archived Round4 causal stop-gated response model, tau 0.2, D1-fitted coefficients',
        state_dimension=25, pose_dimension=15, sensors=5,
        selection='epoch % 5 == 0 and time >= 30 seconds; no selection by convergence, accuracy, or runtime',
        full_recording=True, max_seconds=None,
        calibration_or_model_retuning=False, source_solver_changes=False,
        replay_feedback='Historical frozen reference solve, tolerance 1e-7, maxiter 150; even nonconverged epochs retained',
        source_archive='SCI_Motion_Response_Validation_2026-11-06.zip',
        frozen_source_sha256=hashes,
        archived_protocol_sha256=digest(FROZEN / 'FROZEN_ROUND4_PROTOCOL.json'),
        configuration=config,
        raw_source=dict(official='https://asrl.utias.utoronto.ca/datasets/mrclam/index.html',
                        mirror_repo='doclorenzo/SLAM-Project',
                        mirror_commit='729e8e35c27e9cfcd86ecefa294fca86aa59e576',
                        official_checksums_verified=False),
        reporting='Retain failed or inaccurate epochs. These tests cannot by themselves prove nonlinear filter consistency.'
    )
    save_json(PROTOCOL, obj)
    print(PROTOCOL)


def run(stage: str):
    if not PROTOCOL.exists():
        raise FileNotFoundError('Run freeze before download or replay')
    verify_frozen()
    command = [sys.executable, str(SOURCE / 'run.py'), stage,
               '--data', str(ROOT / 'raw'), '--out', str(ROOT / 'replay'),
               '--config', str(SOURCE / 'configs/default.json'),
               '--calibration', str(FROZEN / 'calibration_response.json'),
               '--datasets', *map(str, DATASETS), '--seeds', str(SEED)]
    if stage == 'download':
        command += ['--source', 'mirror']
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    subprocess.run(command, check=True, env=env)


def export():
    import numpy as np
    source_hashes, config = verify_frozen()
    if not PROTOCOL.exists():
        raise FileNotFoundError('Missing predeclared extension protocol')
    outputs = []
    for dataset in DATASETS:
        bank = ROOT / 'replay' / 'banks' / f'dataset{dataset:02d}_seed{SEED}'
        metadata = json.loads((bank / 'manifest.json').read_text())
        assert metadata['complete'], f'Incomplete replay: {bank}'
        assert metadata['config'] == config and metadata['max_seconds'] is None
        assert metadata['dataset'] == dataset and metadata['seed'] == SEED
        values = {key: [] for key in SELECTED_KEYS}
        source_shards = {}
        all_epochs = []
        full_reference_nonconverged = 0
        full_reference_max_gap = -float('inf')
        for shard in sorted(bank.glob('bank_*.npz')):
            source_shards[shard.name] = digest(shard)
            with np.load(shard, allow_pickle=False) as batch:
                epoch = batch['epoch']
                all_epochs.append(epoch)
                full_reference_nonconverged += int((~batch['ref_converged']).sum())
                full_reference_max_gap = max(full_reference_max_gap, float(batch['ref_gap'].max()))
                keep = (epoch % 5 == 0) & (batch['time'] >= 30)
                for key in values:
                    values[key].append(batch[key][keep])
        all_epochs = np.concatenate(all_epochs)
        assert np.array_equal(all_epochs, np.arange(metadata['epochs'])), 'Missing or duplicated replay epochs'
        values = {key: np.concatenate(parts, axis=0) for key, parts in values.items()}
        count = len(values['epoch'])
        assert count > 0
        assert values['A'].shape == values['B'].shape == (count, 5, 25, 25)
        assert values['means'].shape == (count, 5, 25)
        assert values['truth'].shape == (count, 15)
        for key, value in values.items():
            assert np.isfinite(value).all(), f'Nonfinite value in {key}'
        assert (values['epoch'] % 5 == 0).all() and (values['time'] >= 30).all()
        target = ROOT / 'frozen_inputs' / bank.name
        target.mkdir(parents=True, exist_ok=True)
        filename = target / 'bank_selected.npz'
        np.savez_compressed(filename, **values)
        provenance_path = ROOT / 'raw' / f'MRCLAM_Dataset{dataset}' / 'provenance.json'
        provenance = json.loads(provenance_path.read_text())
        for name, expected in provenance['sha256'].items():
            assert digest(provenance_path.parent / name) == expected
        manifest = dict(metadata)
        manifest.update(cache_scope='Complete primary selection only; not full replay bank',
                        cached_epochs=count, selection='epoch%5==0 and time>=30',
                        state_dimension=25, pose_dimension=15, sensors=5,
                        independent_recorded_sequence=dataset,
                        source_original_bank_path=str(bank),
                        source_shard_sha256=source_shards,
                        selected_input_sha256=digest(filename),
                        raw_data_provenance=provenance,
                        frozen_source_sha256=source_hashes,
                        extension_protocol_sha256=digest(PROTOCOL),
                        historical_reference_nonconverged_selected=int((~values['ref_converged']).sum()),
                        historical_reference_nonconverged_full=full_reference_nonconverged,
                        historical_reference_max_gap_full=full_reference_max_gap,
                        historical_reference_max_gap=float(values['ref_gap'].max()),
                        reference_caution='Historical feedback reference; independently re-audit before using for objective-error claims')
        save_json(target / 'manifest.json', manifest)
        outputs.append(dict(dataset=dataset, seed=SEED, cases=count,
                            bank=str(filename.relative_to(ROOT)),
                            input_sha256=digest(filename),
                            replay_epochs=metadata['epochs'],
                            historical_reference_nonconverged_selected=manifest['historical_reference_nonconverged_selected']))
        print(f'Dataset {dataset}: exported {count} of {metadata["epochs"]} epochs, unchanged.')
    save_json(ROOT / 'frozen_inputs_manifest.json', outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'verify', 'download', 'replay', 'export'])
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze()
    elif args.stage == 'verify':
        hashes, _ = verify_frozen()
        print(f'Verified {len(hashes)} archived source/model hashes.')
    elif args.stage in ('download', 'replay'):
        run(args.stage)
    else:
        export()
