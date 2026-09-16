import copy
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import pytest

STAGE2 = Path(__file__).resolve().parents[1] / 'scripts/rl2_vla/stage2_cluster_reward'
sys.path.insert(0, str(STAGE2))
import run_experiment
from data import ACTION_CONTRACT, FEATURE_CONTRACT, TASKS


def _episode(env_seed=1000, policy_seed=42):
    return {
        'schema_version': 1, 'feature_contract': FEATURE_CONTRACT,
        'action_contract': ACTION_CONTRACT, 'episode_id': f'e{env_seed}',
        'reset_id': f'r{env_seed}', 'task_id': TASKS[0], 'env_seed': env_seed,
        'policy_seed': policy_seed, 'success': False, 'end_reason': 'horizon',
        'chunks': [{
            'context': [0.] * 1024, 'start_step': 0,
            'executed_actions': [[0.] * 7], 'postprocessed_actions': [[0.] * 7] * 4,
            'normalized_actions': [[0.] * 7] * 4, 'step_reward': [0.],
            'step_success': [False], 'step_truncated': [False],
        }],
    }


def test_receipt_and_local_episode_are_deduplicated(tmp_path):
    payload = _episode()
    local = tmp_path / 'episode_1000.json'
    raw = json.dumps(payload).encode()
    local.write_bytes(raw)
    receipts = tmp_path / '.archive_receipts'
    receipts.mkdir()
    summary = {key: payload[key] for key in (
        'schema_version', 'feature_contract', 'action_contract', 'episode_id', 'reset_id',
        'task_id', 'env_seed', 'policy_seed', 'success', 'end_reason')}
    digest = hashlib.sha256(raw).hexdigest()
    proof = {'path': str(local), 'sha256': digest, 'size_bytes': len(raw)}
    (receipts / local.name).write_text(json.dumps({
        'episode': summary, 'episode_sha256': digest, 'episode_size_bytes': len(raw),
        'files': [{**proof, 'role': 'episode'}], 'verified_remote': [proof],
    }))
    records = run_experiment._completed_episodes(tmp_path)
    assert len(records) == 1
    assert records[0]['env_seed'] == 1000


def test_local_receipt_conflict_fails(tmp_path):
    payload = _episode()
    local = tmp_path / 'episode_1000.json'
    raw = json.dumps(payload).encode()
    local.write_bytes(raw)
    receipts = tmp_path / '.archive_receipts'
    receipts.mkdir()
    changed = copy.deepcopy(payload)
    changed['chunks'][0]['context'][0] = 1.0
    changed_raw = json.dumps(changed).encode()
    digest = hashlib.sha256(changed_raw).hexdigest()
    proof = {'path': str(local), 'sha256': digest, 'size_bytes': len(changed_raw)}
    summary = {key: changed[key] for key in (
        'schema_version', 'feature_contract', 'action_contract', 'episode_id', 'reset_id',
        'task_id', 'env_seed', 'policy_seed', 'success', 'end_reason')}
    (receipts / local.name).write_text(json.dumps({
        'episode': summary, 'episode_sha256': digest, 'episode_size_bytes': len(changed_raw),
        'files': [{**proof, 'role': 'episode'}], 'verified_remote': [proof],
    }))
    with pytest.raises(ValueError, match='conflict'):
        run_experiment._completed_episodes(tmp_path)


def test_collect_command_has_activation_flag_and_task_subset(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', [
        'run_experiment.py', '--mode', 'collect', '--rl2-root', '/tmp/rl2',
        '--gpu', '0', '--output', str(tmp_path := Path('/tmp/rl2-test-output')),
        '--tasks', TASKS[1], '--seeds', '42', '--trials', '1',
    ])
    run_experiment.main()
    output = capsys.readouterr().out
    assert TASKS[0] not in output
    assert '--stage2_save_activations' in output


def test_gpu_worker_keeps_uneven_jobs_exclusive(monkeypatch):
    active = {}
    peak = {}
    lock = threading.Lock()

    def fake_run(_rl2, lane, _cmd, _request, gpu, _trials, _start, _seed):
        with lock:
            active[gpu] = active.get(gpu, 0) + 1
            peak[gpu] = max(peak.get(gpu, 0), active[gpu])
        time.sleep(0.01 if lane == 'slow' else 0.001)
        with lock:
            active[gpu] -= 1

    monkeypatch.setattr(run_experiment, '_run_collection_lane', fake_run)
    jobs = [('slow', [], {}, 1), ('fast', [], {}, 2)]
    run_experiment._run_gpu_worker('7', jobs, Path('/tmp/rl2'), 1, 0)
    assert peak == {'7': 1}


def test_two_models_per_gpu_assigns_each_lane_once(monkeypatch, tmp_path):
    calls=[]
    monkeypatch.setattr(run_experiment, '_run_gpu_worker', lambda gpu,jobs,*_: calls.append((gpu,jobs)))
    monkeypatch.setattr(sys,'argv',['run_experiment','--mode','collect','--rl2-root',str(tmp_path),
        '--output',str(tmp_path/'out'),'--gpus','2','3','--models-per-gpu','2','--run'])
    run_experiment.main()
    assert sorted(g for g,_ in calls)==['2','2','3','3']
    lanes=[str(job[0]) for _,jobs in calls for job in jobs]
    assert len(lanes)==12 and len(set(lanes))==12
