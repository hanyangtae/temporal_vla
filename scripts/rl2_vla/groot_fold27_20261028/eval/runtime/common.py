"""fold27 online MIL + QAM/TRQAM evaluator: shared constants and fail-closed helpers (stdlib only).

Runs on every host (kanu / srv48 / srv50) with plain python3. Torch/jax pieces live in
gated_action.py / serve_mil_qam.py (model container or host conda) and restore_actor.py (.venvs/rl2_iid).

Contract summary (source of truth = the code below; receipt = outputs/analysis/groot_fold27_20261028/EVAL_RUNTIME_READY.md):
  eval rows   = fold_spec.json fold['eval'] (target j x n0..39, archive seeds); fit 160 never touched here
  baseline    = archive success in the same rows (no new base sweep in production)
  plans       = per noise range: n0_9 scene original plan, n10_19 748863ce9233, n20_39 2f26fb046fc4 (sha pinned)
  host        = fold['eval_host'] (= archive collection machine)
  policy      = frozen GR00T N1.5 ckpt-120000, H16 predict / execute 5, flow 4, max 720 env steps
  gate        = fold MIL (task_classification 33e5e04, L12 D3 token-mean 1536, w_F > 0.5) on EVERY inference
  QAM         = action_decoder input last16 mean denoise0 (1024), active12, lam = 0.5 (proven paired contract)
"""
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOLD27 = HERE.parents[1]
R = HERE.parents[4]
ADAPTER = R / 'scripts/rl2_vla/groot_robocasa_adapter_20261021'
for _p in (FOLD27, FOLD27 / 'mil'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FOLD_SPEC = 'outputs/analysis/groot_fold27_20261028/folds/fold_spec.json'
EVAL_ROOT = 'outputs/eval/robocasa/groot_n15/fold27_mil_qam_20261030'
SMOKE_ROOT = 'outputs/eval/robocasa/groot_n15/fold27_mil_qam_smoke_20261030'
COMMON_QAM_WORK = 'outputs/rl2_vla/atomic65_common_qam_work_20261029'
TC_COMMIT = '33e5e0495d8918b19e87da105bff735d2d80461d'
FEATURE_SIGNATURE = 'ccbfb51ca01724cb'
QAM_WEIGHT = 0.5
N_ACTION_STEPS, CHUNK, MAX_STEPS, FLOW_STEPS = 5, 16, 720, 4
COMMON_STEPS = 500_000

# plan_id -> repo-relative collection_plan.json (file's own plan_id is re-checked on load)
PLAN_FILES = {
    '08f1c9df8207': 'configs/collect/n15_grid_v6_scene_jitter/collection_plan.json',
    '4ab360df2b71': '.claude/worktrees/collect-topup/configs/collect/n15_topup_remaining_20260909/collection_plan.json',
    '4fa6496cd684': '.claude/worktrees/collect-topup/configs/collect/n15_topup_ppcc_20260909/collection_plan.json',
    '748863ce9233': '.claude/worktrees/collect-topup/configs/collect/n15_target9_n20_20261013/collection_plan.json',
    '2f26fb046fc4': '.claude/worktrees/collect-topup/configs/collect/n15_target9_n40_20261025/collection_plan.json',
}

# arm -> (serve mode, actor kind). Production = gated only; smoke_* never accepted by production paths.
PRODUCTION_ARMS = {'mil_qam65': ('gated', 'common'), 'mil_trqam_basic': ('gated', 'trqam_basic'),
                   'mil_trqam_pbrs': ('gated', 'trqam_pbrs'), 'mil_trqam_mix': ('gated', 'trqam_mix')}
SMOKE_ARMS = {'smoke_base': ('base', None), 'smoke_always': ('always', 'smoke'), 'smoke_gated': ('gated', 'smoke')}

# docs/05 + configs/harness/gpu_policy.json (all-session caps). archive machine names are folds.py canonical.
HOSTS = {
    'kanu': {'archive': 'kanu', 'serve': 'docker_run', 'env': 'docker_run', 'port0': 8860},
    'srv48': {'archive': 'worker1', 'serve': 'host_conda', 'env': 'docker_exec', 'port0': 8890},
    'srv50': {'archive': 'worker2', 'serve': 'host_conda', 'env': 'docker_exec', 'port0': 8890},
}
SRV_SERVE_PY = '~/miniconda3/envs/lerobot_050_groot/bin/python'
SRV_ENV_CONTAINER = 'robocasa'
CONTAINER_REPO = '/temporal_vla'
FIXTURE_ATTR = {'OpenDrawer': 'drawer', 'SlideOvenRack': 'oven', 'SlideDishwasherRack': 'dishwasher',
                'PickPlaceCounterToCabinet': 'cab'}


class RuntimeContractError(RuntimeError):
    pass


def need(ok, msg):
    if not ok:
        raise RuntimeContractError(msg)


def sha256_path(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def rel(p):
    return str(Path(p).resolve().relative_to(R))


def write_json_new(p, obj):
    """O_EXCL create; an identical existing file is accepted, a different one is refused."""
    p = Path(p)
    text = json.dumps(obj, indent=1, sort_keys=True) + '\n'
    if p.exists():
        need(p.read_text() == text, f'refusing to overwrite different {p}')
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, 'w') as f:
        f.write(text)


def policy_hostnames():
    pol = json.loads((R / 'configs/harness/gpu_policy.json').read_text())
    return {m: v['hostname'] for m, v in pol['machines'].items()}, pol


def check_host(host):
    need(host in HOSTS, f'unknown eval host {host}')
    names, _ = policy_hostnames()
    actual = socket.gethostname().split('.')[0]
    need(actual == names[host], f'eval host {host} must run on hostname {names[host]}, this is {actual}')


def caps(host):
    _, pol = policy_hostnames()
    m = pol['machines'][host]
    return int(m['max_gpus_all_sessions']), int(m['max_serves_per_gpu'])


def tc_commit():
    """HEAD of task_classification (same resolution as mil/mil_gate.tc_commit, stdlib only)."""
    tc = R / 'task_classification'
    g = tc / '.git'
    need(g.exists(), f'task_classification checkout missing on this host: {tc}')
    if g.is_file():
        g = (tc / g.read_text().split(':', 1)[1].strip()).resolve()
    head = (g / 'HEAD').read_text().strip()
    if head.startswith('ref: '):
        head = (g / head[5:]).read_text().strip()
    return head


def load_plan(plan_id):
    need(plan_id in PLAN_FILES, f'no known plan file for plan_id {plan_id}')
    p = R / PLAN_FILES[plan_id]
    need(p.is_file(), f'plan file missing on this host: {p}')
    d = json.loads(p.read_text())
    need(d.get('plan_id') == plan_id, f'{p}: plan_id {d.get("plan_id")} != {plan_id}')
    return p, d


def check_row_against_plan(plan, key, scene, row):
    """Seeds/coordinates of an archive row must be exactly what the plan prescribes (fail closed)."""
    for k in ('env_seed', 'inference_seed', 'n', 'j', 'plan_id'):
        need(row.get(k) is not None, f'row n{row.get("n")} missing {k}')
    sc = plan['scenes'][key][scene]
    need(int(sc['env_seed']) == int(row['env_seed']), f'{key} s{scene} n{row["n"]}: env_seed {row["env_seed"]} != plan {sc["env_seed"]}')
    need(int(plan['noise_seeds'][int(row['n'])]) == int(row['inference_seed']),
         f'{key} s{scene} n{row["n"]}: inference_seed {row["inference_seed"]} != plan {plan["noise_seeds"][int(row["n"])]}')
    need(0 <= int(row['j']) < len(plan['jitters'][key][scene]), f'j{row["j"]} outside plan jitters')
    return sc


def task_identity(plan, key, scene):
    ex = plan['extra']
    sc = plan['scenes'][key][scene]
    task = ex['tasks'][key]
    need(task in FIXTURE_ATTR, f'no fixture identity rule for task {task}')
    return {'task': task, 'env_name': ex['env_names'][key], 'lang': sc['lang'], 'layout': sc['layout'], 'style': sc['style'],
            'fixture_group': sc['fixture_group'], 'fixture_attr': FIXTURE_ATTR[task]}
