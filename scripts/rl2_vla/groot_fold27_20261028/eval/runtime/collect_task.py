"""Env-side collector wrapper (robocasa container): collect_safe_qam.py generalized to the 9 fold instructions.

Runs scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py unchanged, with fail-closed per-step task identity
(language / layout / style / fixture group / observed language / HTTP payload task) taken from the row's own
noise-range plan (--plan-json, --grid-instruction, --scene-idx). Trace: $PDK_LOG_FILE -> .task_trace.jsonl.
Note: the fixture-name rule (name == group or startswith group_) was proven only for OpenDrawer in rescue9;
other families fail closed at the first step if it does not hold (integration smoke per family).
"""
import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
sys.path.insert(0, str(C.ADAPTER))
sys.path.insert(0, str(C.R / 'scripts/utils'))


def argv_value(flag):
    return sys.argv[sys.argv.index(flag) + 1]


def expected_identity():
    plan = json.loads(Path(argv_value('--plan-json')).read_text())
    key, sid = argv_value('--grid-instruction'), int(argv_value('--scene-idx'))
    ident = C.task_identity(plan, key, sid)
    C.need(argv_value('--task') == ident['task'] and argv_value('--env-name') == ident['env_name'], 'task/env differ from plan')
    C.need(argv_value('--canonical-instruction') == ident['lang'], 'canonical instruction differs from plan')
    return ident


def main():
    import numpy as np
    from readonly_assets import install
    from task_language_guard import canonical_hash
    install()
    ident = expected_identity()
    stream = Path(os.environ['PDK_LOG_FILE']).with_suffix('.task_trace.jsonl').open('x', buffering=1)
    ctx = {'env': None, 'started': False, 'steps': 0, 'initial': None, 'latest_obs': None}

    def emit(event, **kw):
        stream.write(json.dumps({'event': event, 'time': time.time(), 'step': ctx['steps'], **kw}, sort_keys=True) + '\n')

    def check():
        e = ctx['env'].env
        m = e.get_ep_meta()
        fx = getattr(e, ident['fixture_attr'])
        task = {'language': m['lang'], 'fixture': fx.name, 'layout': m['layout_id'], 'style': m['style_id']}
        if ident['task'] == 'OpenDrawer':
            task['drawer_side'] = e.drawer_side
            C.need(task['language'] == f"Open the {task['drawer_side']} drawer.", 'drawer side differs from language')
        C.need(task['language'] == ident['lang'], f'language {task["language"]!r} != plan {ident["lang"]!r}')
        C.need(task['layout'] == ident['layout'] and task['style'] == ident['style'], f'layout/style {task} != plan')
        g = ident['fixture_group']
        C.need(task['fixture'] == g or task['fixture'].startswith(g + '_'), f'fixture {task["fixture"]} not in group {g}')
        C.need(ctx['latest_obs']['annotation.human.action.task_description'] == task['language'], 'observed language differs')
        if not ctx['started']:
            ctx['initial'] = dict(task); ctx['started'] = True
            st = np.asarray(e.sim.get_state().flatten())
            emit('reset', task=task, state_sha256=hashlib.sha256(st.tobytes()).hexdigest())
        C.need(task == ctx['initial'], 'task changed within episode')
        return task

    from robocasa.utils.gym_utils.gymnasium_groot import GrootRoboCasaEnv
    reset0, step0 = GrootRoboCasaEnv.reset, GrootRoboCasaEnv.step

    def reset(self, *args, **kwargs):
        C.need(not ctx['started'], 'reset after episode start')
        obs, info = reset0(self, *args, **kwargs)
        ctx['env'] = self; ctx['latest_obs'] = obs
        return obs, info

    def step(self, action):
        t = check()
        emit('before_step', task=t, actions={k: np.asarray(v).tolist() for k, v in action.items()})
        obs, reward, terminated, truncated, info = step0(self, action)
        ctx['latest_obs'] = obs; ctx['steps'] += 1
        t = check()
        emit('after_step', task=t, success=bool(self.env._check_success()), terminated=bool(terminated), truncated=bool(truncated))
        return obs, reward, terminated, truncated, info
    GrootRoboCasaEnv.reset, GrootRoboCasaEnv.step = reset, step
    import requests
    send0 = requests.sessions.Session.send

    def send(self, request, **kw):
        if request.method == 'POST' and request.url.split('?')[0].endswith('/act_with_features'):
            payload = json.loads(request.body)
            t = check()
            C.need(payload['task'] == t['language'], 'HTTP task differs from env language')
            emit('http_send', task=payload['task'], payload_sha256=canonical_hash(payload), inference_seed=payload.get('inference_seed'))
            response = send0(self, request, **kw)
            emit('http_response', status=response.status_code)
            return response
        return send0(self, request, **kw)
    requests.sessions.Session.send = send
    try:
        runpy.run_path(str(C.R / 'scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py'), run_name='__main__')
        emit('complete', steps=ctx['steps'])
    except BaseException as exc:
        emit('failed', error=repr(exc))
        raise
    finally:
        stream.close()


if __name__ == '__main__':
    main()
