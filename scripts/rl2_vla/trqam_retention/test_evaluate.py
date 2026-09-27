"""CPU contract tests for evaluate.py (no GPU/JAX imports, no subprocess launches)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import evaluate as ev
from data import TASKS, sha256

TASK = TASKS[0]


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def make_tree(root, train_seeds=(42, 0, 7), steps=50000):
    cache = root / 'cache'
    rows = [dict(episode_id='a', split='train', success=True, task_id=TASK, env_seed=1000),
            dict(episode_id='b', split='train', success=False, task_id=TASK, env_seed=1001),
            dict(episode_id='c', split='validation', success=True, task_id=TASK, env_seed=1002),
            dict(episode_id='d', split='train', success=True, task_id=TASKS[1], env_seed=1000)]
    write(cache / 'manifest.json', dict(episodes=rows))
    write(cache / 'cache.json', dict(manifest_sha256=sha256(cache / 'manifest.json'),
                                     qam_transitions=dict(cluster_potential=dict(sha256='t' * 64))))
    original = root / 'original.pkl'
    original.write_bytes(b'original')
    data = dict(manifest_sha256=sha256(cache / 'manifest.json'), transitions_sha256='t' * 64,
                cluster_sha256='c' * 64, episode_ids=['a', 'd'], episodes=2, transitions=10, selection='train AND success')
    for method in ev.METHODS:
        for seed in train_seeds:
            run = root / 'training' / f'{method}_seed{seed}'
            ckpt = run / f'params_{steps}.pkl'
            ckpt.parent.mkdir(parents=True)
            ckpt.write_bytes(f'{method}{seed}'.encode())
            write(run / 'flags.json', dict(seed=seed, agent=dict(horizon_length=4, retention_method=method),
                  retention=dict(method=method, seed=seed, steps=steps, batch_size=256, kl_budget=1., smoke=False),
                  data=data, initial_checkpoint_sha256=sha256(original)))
            write(run / 'DONE.json', dict(steps=steps, smoke=False, checkpoint=f'/remote/x/{ckpt.name}',
                                          sha256=sha256(ckpt), actor_changed=True, reference_frozen=True))
            write(run / 'migration.json', dict(strict_restore=True, action_identity=True))
    return cache, original, root / 'training'


def edit_flags(training, name, fn):
    path = training / name / 'flags.json'
    flags = json.loads(path.read_text())
    fn(flags)
    path.write_text(json.dumps(flags))


def chunk(step, trigger, action=0.):
    return dict(start_step=step, safe_trigger=trigger, context=np.zeros(4), proposed_actions=np.full(2, action),
                executed_actions=np.full(2, action), input_hashes={'raw_observation': str(step)},
                policy_noise_sha256=str(step))


def episode(chunks, success, key=f'{TASK}/env11000/policy42'):
    return dict(episode_id=key, task_id=TASK, reset_id=f'{TASK}/env11000', env_seed=11000, policy_seed=42,
                episode_rng_seed=1, policy_checkpoint='pi0', action_contract='a', feature_contract='f',
                rng_contract='paired_rng_v2', environment_contract='fresh_env_per_episode_v1',
                success=success, chunks=chunks)


class TrainingProvenance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache, self.original, self.training = make_tree(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, seeds=(42, 0, 7), start=11000, trials=25):
        return ev.validate_training(self.training, self.original, self.cache, list(seeds), list(TASKS), start, trials)

    def test_valid_tree_yields_seven_semantic_arms(self):
        arms, training = self.check()
        self.assertEqual(list(arms), ev.arm_names([42, 0, 7]))
        self.assertEqual(list(arms), ['original', 'qam_trainseed42', 'qam_trainseed0', 'qam_trainseed7',
                                      'trqam_trainseed42', 'trqam_trainseed0', 'trqam_trainseed7'])
        self.assertEqual(arms['original']['checkpoint_sha256'], sha256(self.original))
        self.assertEqual(training['steps'], 50000)
        self.assertEqual(training['success_train_episodes'], 2)

    def test_subset_for_smoke(self):
        arms, _ = self.check(seeds=[42])
        self.assertEqual(list(arms), ['original', 'qam_trainseed42', 'trqam_trainseed42'])

    def test_swapped_method_rejected(self):
        edit_flags(self.training, 'qam_seed0', lambda f: f['retention'].update(method='trqam'))
        with self.assertRaisesRegex(ValueError, 'retention method'):
            self.check()

    def test_wrong_seed_rejected(self):
        edit_flags(self.training, 'trqam_seed7', lambda f: f.update(seed=42))
        with self.assertRaisesRegex(ValueError, 'training seed'):
            self.check()

    def test_unmatched_budget_rejected(self):
        edit_flags(self.training, 'trqam_seed0', lambda f: f['retention'].update(batch_size=128))
        with self.assertRaisesRegex(ValueError, 'unmatched budget'):
            self.check()

    def test_failure_episode_in_training_set_rejected(self):
        edit_flags(self.training, 'qam_seed42', lambda f: f['data'].update(episode_ids=['a', 'b', 'd'], episodes=3))
        with self.assertRaisesRegex(ValueError, 'train-split successes'):
            self.check()

    def test_other_initialization_rejected(self):
        self.original.write_bytes(b'other')
        with self.assertRaisesRegex(ValueError, 'original checkpoint'):
            self.check()

    def test_checkpoint_hash_mismatch_rejected(self):
        (self.training / 'trqam_seed42' / 'params_50000.pkl').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.check()

    def test_smoke_training_rejected(self):
        edit_flags(self.training, 'qam_seed7', lambda f: f['retention'].update(smoke=True))
        with self.assertRaisesRegex(ValueError, 'smoke'):
            self.check()

    def test_eval_reset_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            self.check(start=999, trials=3)


class Command(unittest.TestCase):
    def test_flags_match_stage2_eval(self):
        cmd = ev.lane_command('py', Path('/rl2'), 'pi0', TASK, 42, 25, 11000, Path('/out/l'), 'q.pkl', Path('/safe'), False)
        pairs = dict(zip(cmd[2::2], cmd[3::2]))
        expected = {'--use_verifier': 'False', '--action_samples': '1', '--composed_samples': '1',
                    '--use_failure_prediction': 'True', '--merge_rel_weight': '0.5', '--n_action_steps': '4',
                    '--use_taskwise_cp_band': 'True', '--failure_cp_alpha': '0.2', '--qam_ckpt': 'q.pkl',
                    '--env_seed_start': '11000', '--num_trials_per_task': '25', '--seed': '42',
                    '--stage2_single_candidate': 'True'}
        for key, value in expected.items():
            self.assertEqual(pairs[key], value, key)
        self.assertNotIn('--stage2_force_gate_off', cmd)
        gated = ev.lane_command('py', Path('/rl2'), 'pi0', TASK, 42, 25, 11000, Path('/out/l'), 'q.pkl', Path('/safe'), True)
        self.assertEqual(dict(zip(gated[2::2], gated[3::2]))['--stage2_force_gate_off'], 'True')


class Pairing(unittest.TestCase):
    def arms(self, arm_chunks, arm_success, base_chunks=None, base_success=False):
        base = episode(base_chunks or [chunk(0, False), chunk(4, True, 1.), chunk(8, False, 1.)], base_success)
        return dict(original={base['episode_id']: base},
                    qam_trainseed42={base['episode_id']: episode(arm_chunks, arm_success)})

    def test_same_prefix_then_divergence_after_trigger_passes(self):
        ev.validate_pairs(self.arms([chunk(0, False), chunk(4, True, 9.)], True))

    def test_different_first_trigger_rejected(self):
        with self.assertRaisesRegex(ValueError, 'first SAFE trigger'):
            ev.validate_pairs(self.arms([chunk(0, True, 9.), chunk(4, True, 9.)], True))

    def test_pre_trigger_action_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'pre-intervention'):
            ev.validate_pairs(self.arms([chunk(0, False, 5.), chunk(4, True, 9.)], True))

    def test_untriggered_outcome_must_match(self):
        untriggered = [chunk(0, False), chunk(4, False)]
        ev.validate_pairs(self.arms(untriggered, False, base_chunks=untriggered, base_success=False))
        with self.assertRaisesRegex(ValueError, 'nonintervention'):
            ev.validate_pairs(self.arms(untriggered, True, base_chunks=untriggered, base_success=False))

    def test_missing_episode_rejected(self):
        arms = self.arms([chunk(0, False), chunk(4, True)], True)
        arms['qam_trainseed42'] = {}
        with self.assertRaisesRegex(ValueError, 'unpaired'):
            ev.validate_pairs(arms)

    def test_summary_counts(self):
        arms = self.arms([chunk(0, False), chunk(4, True, 9.)], True)
        summary = ev.summarize(arms, [TASK])
        row = summary['qam_trainseed42'][TASK]
        self.assertEqual((row['episodes'], row['successes'], row['triggered_episodes']), (1, 1, 1))
        self.assertEqual(row['vs_original']['original_fail_arm_success'], 1)
        self.assertEqual(summary['original']['all']['successes'], 0)


class DryRun(unittest.TestCase):
    def test_contract_and_plan_serialized_and_locked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache, original, training = make_tree(root)
            rl2, policy, safe = root / 'rl2', root / 'pi0', root / 'safe'
            for f in ev.UPSTREAM:
                (rl2 / f).parent.mkdir(parents=True, exist_ok=True)
                (rl2 / f).write_text(f)
            policy.mkdir()
            (policy / 'model.safetensors').write_bytes(b'w')
            write(safe / 'provenance.json', {})
            argv = ['--rl2-root', str(rl2), '--gpu', '0', '--policy', str(policy), '--safe-dir', str(safe),
                    '--original', str(original), '--training-root', str(training), '--cache', str(cache),
                    '--output', str(root / 'out')]
            with mock.patch.object(ev, 'validate'), mock.patch.object(ev, '_run_collection_lane') as run:
                contract = ev.main(argv)
                run.assert_not_called()
                self.assertEqual(contract['episodes_per_arm'], 300)
                self.assertEqual(contract['total_episodes'], 2100)
                plan = json.loads((root / 'out' / 'plan.json').read_text())
                self.assertEqual(len(plan), 7 * 4 * 3)
                self.assertEqual({p['command'][p['command'].index('--seed') + 1] for p in plan}, {'42', '0', '7'})
                self.assertEqual(copy.deepcopy(contract), json.loads((root / 'out' / 'contract.json').read_text()))
                with self.assertRaisesRegex(ValueError, 'different evaluation contract'):
                    ev.main(argv + ['--trials', '1'])
                smoke = ev.main(argv[:-1] + [str(root / 'smoke'), '--tasks', TASK, '--train-seeds', '42',
                                             '--seeds', '42', '--trials', '1', '--force-gate-off'])
                self.assertEqual((len(smoke['arms']), smoke['total_episodes'], smoke['force_gate_off']), (3, 3, True))


if __name__ == '__main__':
    unittest.main()
