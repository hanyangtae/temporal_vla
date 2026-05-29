"""Processor Pipeline 유닛 테스트.

Docker 없이 로컬에서 실행 가능. 통합 테스트 전에 로직을 검증한다.

실행:
  python3 -m pytest tests/test_processor.py -v
  또는
  python3 tests/test_processor.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from processor.types import (
    FeatureType,
    Features,
    PipelineFeatureType,
    PolicyFeature,
    TransitionKey,
    create_transition,
)
from processor.base import DataProcessorPipeline
from processor.obs.calvin import CalvinObsProcessor
from processor.obs.robocasa import RoboCasaObsProcessor
from processor.action.calvin import CalvinActionProcessor
from processor.action.robocasa import RoboCasaActionProcessor
from processor.factory import make_calvin_processors, make_robocasa_processors


# ============================================================================
# CalvinObsProcessor
# ============================================================================

class TestCalvinObsProcessor(unittest.TestCase):
    """CalvinObsProcessor: CHW [-1,1] → HWC uint8 [0,255]."""

    def setUp(self):
        self.proc = CalvinObsProcessor(use_wrist=False, image_size=200)

    def _make_obs(self, with_wrist=False):
        """Calvin env가 반환하는 raw obs를 시뮬레이션."""
        # CHW float in [-1, 1]
        static = np.random.uniform(-1, 1, (3, 200, 200)).astype(np.float32)
        obs = {"rgb_obs": {"rgb_static": static}}
        if with_wrist:
            gripper = np.random.uniform(-1, 1, (3, 200, 200)).astype(np.float32)
            obs["rgb_obs"]["rgb_gripper"] = gripper
        return obs

    def test_static_only(self):
        obs = self._make_obs()
        result = self.proc.process_observation(obs)

        self.assertIn("observation.images.static", result)
        self.assertNotIn("observation.images.wrist", result)

        img = result["observation.images.static"]
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(img.shape, (200, 200, 3))  # HWC
        self.assertTrue(img.min() >= 0)
        self.assertTrue(img.max() <= 255)

    def test_with_wrist(self):
        proc = CalvinObsProcessor(use_wrist=True, image_size=200)
        obs = self._make_obs(with_wrist=True)
        result = proc.process_observation(obs)

        self.assertIn("observation.images.static", result)
        self.assertIn("observation.images.wrist", result)

        for key in ["observation.images.static", "observation.images.wrist"]:
            img = result[key]
            self.assertEqual(img.dtype, np.uint8)
            self.assertEqual(img.shape, (200, 200, 3))

    def test_conversion_range(self):
        """[-1, 1] 전체 범위 변환 정확성."""
        # min=-1 → 0, max=1 → 255
        static = np.full((3, 4, 4), -1.0, dtype=np.float32)
        obs = {"rgb_obs": {"rgb_static": static}}
        result = self.proc.process_observation(obs)
        self.assertEqual(result["observation.images.static"].min(), 0)

        static = np.full((3, 4, 4), 1.0, dtype=np.float32)
        obs = {"rgb_obs": {"rgb_static": static}}
        result = self.proc.process_observation(obs)
        self.assertEqual(result["observation.images.static"].max(), 255)

    def test_already_uint8(self):
        """이미 uint8인 경우 변환 없이 통과."""
        static = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        obs = {"rgb_obs": {"rgb_static": static}}
        result = self.proc.process_observation(obs)
        np.testing.assert_array_equal(result["observation.images.static"], static)

    def test_pipeline_integration(self):
        """TransitionKey 래핑으로 pipeline 호출."""
        obs = self._make_obs()
        transition = {TransitionKey.OBSERVATION: obs}
        result = self.proc(transition)

        self.assertIn(TransitionKey.OBSERVATION, result)
        self.assertIn("observation.images.static", result[TransitionKey.OBSERVATION])

    def test_transform_features(self):
        features = {}
        result = self.proc.transform_features(features)
        obs_feats = result[PipelineFeatureType.OBSERVATION]
        self.assertIn("observation.images.static", obs_feats)
        self.assertEqual(obs_feats["observation.images.static"].shape, (200, 200, 3))

    def test_get_config(self):
        config = self.proc.get_config()
        self.assertEqual(config["use_wrist"], False)
        self.assertEqual(config["image_size"], 200)


# ============================================================================
# RoboCasaObsProcessor
# ============================================================================

class TestRoboCasaObsProcessor(unittest.TestCase):
    """RoboCasaObsProcessor: robosuite obs → 통일 키 리매핑."""

    def setUp(self):
        self.proc = RoboCasaObsProcessor()

    def _make_obs(self, with_state=True):
        """robosuite env가 반환하는 raw obs를 시뮬레이션."""
        obs = {
            "robot0_agentview_left_image": np.random.randint(
                0, 256, (224, 224, 3), dtype=np.uint8
            ),
            "robot0_eye_in_hand_image": np.random.randint(
                0, 256, (224, 224, 3), dtype=np.uint8
            ),
        }
        if with_state:
            obs["robot0_proprio-state"] = np.random.randn(68).astype(np.float32)
        return obs

    def test_key_remapping(self):
        obs = self._make_obs()
        result = self.proc.process_observation(obs)

        self.assertIn("observation.images.static", result)
        self.assertIn("observation.images.wrist", result)
        self.assertIn("observation.state.eef_pos", result)
        self.assertIn("observation.state.eef_quat", result)
        self.assertIn("observation.state.gripper_qpos", result)
        self.assertIn("observation.state.joint_pos", result)

        # 원본 이미지가 그대로 전달되는지
        np.testing.assert_array_equal(
            result["observation.images.static"],
            obs["robot0_agentview_left_image"],
        )
        np.testing.assert_array_equal(
            result["observation.images.wrist"],
            obs["robot0_eye_in_hand_image"],
        )

    def test_state_float32(self):
        obs = self._make_obs()
        result = self.proc.process_observation(obs)
        self.assertEqual(result["observation.state.eef_pos"].dtype, np.float32)
        self.assertEqual(result["observation.state.eef_pos"].shape, (3,))
        self.assertEqual(result["observation.state.eef_quat"].shape, (4,))
        self.assertEqual(result["observation.state.gripper_qpos"].shape, (2,))

    def test_missing_cameras_fallback(self):
        """카메라 누락 시 zero 이미지로 fallback."""
        obs = {}  # 카메라 키가 전혀 없음
        result = self.proc.process_observation(obs)

        for key in ["observation.images.static", "observation.images.wrist"]:
            img = result[key]
            self.assertEqual(img.shape, (224, 224, 3))
            self.assertEqual(img.dtype, np.uint8)
            self.assertTrue(np.all(img == 0))

    def test_no_state_key(self):
        """state 키가 없으면 결과에 state 미포함."""
        obs = self._make_obs(with_state=False)
        result = self.proc.process_observation(obs)
        self.assertFalse(any(k.startswith("observation.state.") for k in result))

    def test_custom_camera_names(self):
        proc = RoboCasaObsProcessor(
            static_cam="cam_a", wrist_cam="cam_b", state_key="my_state"
        )
        obs = {
            "cam_a_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "cam_b_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "my_state": np.zeros(68, dtype=np.float32),
        }
        result = proc.process_observation(obs)
        self.assertIn("observation.images.static", result)
        self.assertIn("observation.images.wrist", result)
        self.assertIn("observation.state.eef_pos", result)
        self.assertIn("observation.state.joint_pos", result)

    def test_transform_features(self):
        features = {}
        result = self.proc.transform_features(features)
        obs_feats = result[PipelineFeatureType.OBSERVATION]
        self.assertEqual(
            obs_feats["observation.state.eef_pos"].shape, (3,)
        )
        self.assertEqual(
            obs_feats["observation.state.eef_quat"].shape, (4,)
        )
        self.assertEqual(
            obs_feats["observation.state.joint_pos"].shape, (7,)
        )
        self.assertEqual(
            obs_feats["observation.images.static"].shape, (224, 224, 3)
        )

    def test_get_config(self):
        config = self.proc.get_config()
        self.assertEqual(config["static_cam"], "robot0_agentview_left")
        self.assertEqual(config["wrist_cam"], "robot0_eye_in_hand")


# ============================================================================
# CalvinActionProcessor
# ============================================================================

class TestCalvinActionProcessor(unittest.TestCase):
    """CalvinActionProcessor: gripper 이산화 {-1, 1}."""

    def setUp(self):
        self.proc = CalvinActionProcessor(threshold=0.0)

    def test_1d_positive_gripper(self):
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.3], dtype=np.float32)
        result = self.proc.process_action(action)
        # gripper 0.3 > 0.0 → 1.0
        self.assertEqual(result[-1], 1.0)
        # arm 부분 불변
        np.testing.assert_array_almost_equal(result[:6], action[:6])

    def test_1d_negative_gripper(self):
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.3], dtype=np.float32)
        result = self.proc.process_action(action)
        # gripper -0.3 > 0.0 → False → -1.0
        self.assertEqual(result[-1], -1.0)

    def test_1d_zero_gripper(self):
        """threshold=0.0일 때 정확히 0.0은 닫힘(-1.0)."""
        action = np.array([0.0] * 6 + [0.0], dtype=np.float32)
        result = self.proc.process_action(action)
        self.assertEqual(result[-1], -1.0)

    def test_2d_batch(self):
        """2D batch 처리."""
        actions = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
        ], dtype=np.float32)
        result = self.proc.process_action(actions)
        self.assertEqual(result[0, -1], 1.0)
        self.assertEqual(result[1, -1], -1.0)

    def test_custom_threshold(self):
        """threshold=0.5 사용."""
        proc = CalvinActionProcessor(threshold=0.5)
        # 0.3 > 0.5 → False → -1.0
        action = np.array([0.0] * 6 + [0.3], dtype=np.float32)
        result = proc.process_action(action)
        self.assertEqual(result[-1], -1.0)

        # 0.7 > 0.5 → True → 1.0
        action = np.array([0.0] * 6 + [0.7], dtype=np.float32)
        result = proc.process_action(action)
        self.assertEqual(result[-1], 1.0)

    def test_does_not_mutate_input(self):
        """입력 array를 변경하지 않음."""
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.3], dtype=np.float32)
        original = action.copy()
        self.proc.process_action(action)
        np.testing.assert_array_equal(action, original)

    def test_dreamvla_output_idempotent(self):
        """DreamVLA 서버가 이미 {-1, 1}로 이산화한 경우 결과 불변.

        DreamVLA serve: (pred > 0.5).float() → (x - 0.5) * 2 → {-1, 1}
        CalvinActionProcessor: > 0.0 → {-1, 1}
        -1 > 0 = False → -1 ✓
         1 > 0 = True  →  1 ✓
        """
        for gripper_val in [-1.0, 1.0]:
            action = np.array([0.0] * 6 + [gripper_val], dtype=np.float32)
            result = self.proc.process_action(action)
            self.assertEqual(result[-1], gripper_val)

    def test_pipeline_integration(self):
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.3], dtype=np.float32)
        transition = {TransitionKey.ACTION: action}
        result = self.proc(transition)
        self.assertEqual(result[TransitionKey.ACTION][-1], 1.0)

    def test_transform_features_unchanged(self):
        """CalvinActionProcessor는 shape 불변 (7D → 7D)."""
        features = {
            PipelineFeatureType.ACTION: {
                "action": PolicyFeature(FeatureType.ACTION, (7,)),
            }
        }
        result = self.proc.transform_features(features)
        self.assertEqual(
            result[PipelineFeatureType.ACTION]["action"].shape, (7,)
        )

    def test_wrong_dimension_1d_raises(self):
        """7D 이외 입력 시 ValueError."""
        for dim in [5, 12]:
            with self.assertRaises(ValueError):
                self.proc.process_action(np.zeros(dim, dtype=np.float32))

    def test_wrong_dimension_2d_raises(self):
        with self.assertRaises(ValueError):
            self.proc.process_action(np.zeros((3, 5), dtype=np.float32))


# ============================================================================
# RoboCasaActionProcessor
# ============================================================================

class TestRoboCasaActionProcessor(unittest.TestCase):
    """RoboCasaActionProcessor: 7D → 12D (PandaMobile)."""

    def setUp(self):
        self.proc = RoboCasaActionProcessor(arm_dim=6)

    def test_7d_to_12d(self):
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8], dtype=np.float32)
        result = self.proc.process_action(action)

        self.assertEqual(result.shape, (12,))
        self.assertEqual(result.dtype, np.float32)

        # arm[0:6] 그대로
        np.testing.assert_array_almost_equal(result[:6], action[:6])
        # gripper × 2
        self.assertAlmostEqual(result[6], 0.8)
        self.assertAlmostEqual(result[7], 0.8)
        # base = 0
        np.testing.assert_array_almost_equal(result[8:11], [0.0, 0.0, 0.0])
        # torso = 0
        self.assertAlmostEqual(result[11], 0.0)

    def test_list_input(self):
        """리스트 입력도 처리."""
        action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
        result = self.proc.process_action(action)
        self.assertEqual(result.shape, (12,))

    def test_negative_gripper(self):
        action = np.array([0.0] * 6 + [-1.0], dtype=np.float32)
        result = self.proc.process_action(action)
        self.assertAlmostEqual(result[6], -1.0)
        self.assertAlmostEqual(result[7], -1.0)

    def test_transform_features(self):
        features = {
            PipelineFeatureType.ACTION: {
                "action": PolicyFeature(FeatureType.ACTION, (7,)),
            }
        }
        result = self.proc.transform_features(features)
        self.assertEqual(
            result[PipelineFeatureType.ACTION]["action"].shape, (12,)
        )

    def test_wrong_dimension_raises(self):
        """7D 이외 입력 시 ValueError."""
        for dim in [5, 12]:
            with self.assertRaises(ValueError):
                self.proc.process_action(np.zeros(dim, dtype=np.float32))

    def test_get_config(self):
        config = self.proc.get_config()
        self.assertEqual(config["arm_dim"], 6)

    def test_subkeyed_groot_action_with_base_motion(self):
        action = {
            "action.eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "action.eef_axisangle": np.array([0.4, 0.5, 0.6], dtype=np.float32),
            "action.gripper": np.array([0.8], dtype=np.float32),
            "action.base_motion": np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32),
            "action.control_mode": np.array([1.0], dtype=np.float32),
        }

        result = self.proc.process_action(action)

        self.assertEqual(result.shape, (12,))
        np.testing.assert_array_almost_equal(
            result,
            np.array(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.8, 0.01, 0.02, 0.03, 0.04],
                dtype=np.float32,
            ),
        )


# ============================================================================
# DataProcessorPipeline
# ============================================================================

class TestDataProcessorPipeline(unittest.TestCase):
    """DataProcessorPipeline: 체이닝, save/load, feature 추적."""

    def test_obs_pipeline_chain(self):
        """obs_pipeline: Calvin raw obs → 통일 키."""
        pipeline = DataProcessorPipeline(
            steps=[CalvinObsProcessor(use_wrist=False)],
            name="test_obs",
        )
        obs = {
            "rgb_obs": {
                "rgb_static": np.random.uniform(-1, 1, (3, 200, 200)).astype(
                    np.float32
                ),
            }
        }
        result = pipeline({TransitionKey.OBSERVATION: obs})
        self.assertIn("observation.images.static", result[TransitionKey.OBSERVATION])

    def test_action_pipeline_chain(self):
        """action_pipeline: CalvinActionProcessor 적용."""
        pipeline = DataProcessorPipeline(
            steps=[CalvinActionProcessor(threshold=0.0)],
            name="test_action",
        )
        action = np.array([0.0] * 6 + [0.5], dtype=np.float32)
        result = pipeline({TransitionKey.ACTION: action})
        self.assertEqual(result[TransitionKey.ACTION][-1], 1.0)

    def test_save_load_roundtrip(self):
        """save_pretrained → from_pretrained 라운드트립."""
        pipeline = DataProcessorPipeline(
            steps=[CalvinActionProcessor(threshold=0.3)],
            name="test_save",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline.save_pretrained(tmpdir)

            # pipeline.json 존재 확인
            config_path = os.path.join(tmpdir, "pipeline.json")
            self.assertTrue(os.path.exists(config_path))

            with open(config_path) as f:
                config = json.load(f)
            self.assertEqual(config["name"], "test_save")
            self.assertEqual(len(config["steps"]), 1)

            # 복원
            loaded = DataProcessorPipeline.from_pretrained(tmpdir)
            self.assertEqual(loaded.name, "test_save")
            self.assertEqual(len(loaded.steps), 1)
            self.assertIsInstance(loaded.steps[0], CalvinActionProcessor)
            self.assertEqual(loaded.steps[0].threshold, 0.3)

            # 동작 동일성
            action = np.array([0.0] * 6 + [0.5], dtype=np.float32)
            r1 = pipeline({TransitionKey.ACTION: action})
            r2 = loaded({TransitionKey.ACTION: action.copy()})
            np.testing.assert_array_equal(
                r1[TransitionKey.ACTION], r2[TransitionKey.ACTION]
            )

    def test_save_load_obs_pipeline(self):
        """ObsProcessor도 save/load 가능."""
        pipeline = DataProcessorPipeline(
            steps=[RoboCasaObsProcessor(static_cam="cam_a", wrist_cam="cam_b")],
            name="robocasa_obs_test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline.save_pretrained(tmpdir)
            loaded = DataProcessorPipeline.from_pretrained(tmpdir)
            self.assertIsInstance(loaded.steps[0], RoboCasaObsProcessor)
            self.assertEqual(loaded.steps[0].static_cam, "cam_a")
            self.assertEqual(loaded.steps[0].wrist_cam, "cam_b")

    def test_feature_tracking(self):
        """transform_features가 파이프라인을 통해 전파."""
        pipeline = DataProcessorPipeline(
            steps=[RoboCasaActionProcessor(arm_dim=6)],
            name="test_features",
        )
        features = {
            PipelineFeatureType.ACTION: {
                "action": PolicyFeature(FeatureType.ACTION, (7,)),
            }
        }
        result = pipeline.transform_features(features)
        self.assertEqual(
            result[PipelineFeatureType.ACTION]["action"].shape, (12,)
        )

    def test_reset(self):
        """reset() 호출이 에러 없이 동작."""
        pipeline = DataProcessorPipeline(
            steps=[CalvinObsProcessor(), CalvinActionProcessor()],
            name="test_reset",
        )
        pipeline.reset()  # 에러 없으면 통과

    def test_repr(self):
        pipeline = DataProcessorPipeline(
            steps=[CalvinObsProcessor(), CalvinActionProcessor()],
            name="test",
        )
        r = repr(pipeline)
        self.assertIn("CalvinObsProcessor", r)
        self.assertIn("CalvinActionProcessor", r)

    def test_none_observation_passthrough(self):
        """observation이 None이면 변환 없이 통과."""
        pipeline = DataProcessorPipeline(
            steps=[CalvinObsProcessor()],
            name="test",
        )
        result = pipeline({TransitionKey.OBSERVATION: None})
        self.assertIsNone(result[TransitionKey.OBSERVATION])

    def test_none_action_passthrough(self):
        """action이 None이면 변환 없이 통과."""
        pipeline = DataProcessorPipeline(
            steps=[CalvinActionProcessor()],
            name="test",
        )
        result = pipeline({TransitionKey.ACTION: None})
        self.assertIsNone(result[TransitionKey.ACTION])


# ============================================================================
# Factory
# ============================================================================

class TestFactory(unittest.TestCase):
    """make_calvin_processors / make_robocasa_processors."""

    def test_make_calvin_processors(self):
        obs_p, act_p = make_calvin_processors()
        self.assertEqual(obs_p.name, "calvin_obs")
        self.assertEqual(act_p.name, "calvin_action")
        self.assertEqual(len(obs_p.steps), 1)
        self.assertEqual(len(act_p.steps), 1)
        self.assertIsInstance(obs_p.steps[0], CalvinObsProcessor)
        self.assertIsInstance(act_p.steps[0], CalvinActionProcessor)

    def test_make_calvin_processors_params(self):
        obs_p, act_p = make_calvin_processors(
            use_wrist=True, image_size=128, gripper_threshold=0.5
        )
        self.assertTrue(obs_p.steps[0].use_wrist)
        self.assertEqual(obs_p.steps[0].image_size, 128)
        self.assertEqual(act_p.steps[0].threshold, 0.5)

    def test_make_robocasa_processors(self):
        obs_p, act_p = make_robocasa_processors()
        self.assertEqual(obs_p.name, "robocasa_obs")
        self.assertEqual(act_p.name, "robocasa_action")
        self.assertIsInstance(obs_p.steps[0], RoboCasaObsProcessor)
        self.assertIsInstance(act_p.steps[0], RoboCasaActionProcessor)

    def test_make_robocasa_processors_params(self):
        obs_p, act_p = make_robocasa_processors(
            static_cam="cam_x", wrist_cam="cam_y", arm_dim=7
        )
        self.assertEqual(obs_p.steps[0].static_cam, "cam_x")
        self.assertEqual(act_p.steps[0].arm_dim, 7)

    def test_calvin_end_to_end(self):
        """Calvin 전체 파이프라인 end-to-end 테스트."""
        obs_p, act_p = make_calvin_processors()

        # obs 변환
        obs = {
            "rgb_obs": {
                "rgb_static": np.random.uniform(-1, 1, (3, 200, 200)).astype(
                    np.float32
                ),
            }
        }
        obs_result = obs_p({TransitionKey.OBSERVATION: obs})
        processed_obs = obs_result[TransitionKey.OBSERVATION]
        self.assertIn("observation.images.static", processed_obs)
        self.assertEqual(processed_obs["observation.images.static"].dtype, np.uint8)

        # action 변환 (모델 응답 시뮬레이션)
        model_action = np.array(
            [0.1, -0.2, 0.3, -0.1, 0.05, -0.05, 0.7], dtype=np.float32
        )
        act_result = act_p({TransitionKey.ACTION: model_action})
        action = act_result[TransitionKey.ACTION]
        self.assertEqual(action[-1], 1.0)  # 0.7 > 0.0 → 열림
        self.assertEqual(action.shape, (7,))

    def test_robocasa_end_to_end(self):
        """RoboCasa 전체 파이프라인 end-to-end 테스트."""
        obs_p, act_p = make_robocasa_processors()

        # obs 변환
        obs = {
            "robot0_agentview_left_image": np.random.randint(
                0, 256, (224, 224, 3), dtype=np.uint8
            ),
            "robot0_eye_in_hand_image": np.random.randint(
                0, 256, (224, 224, 3), dtype=np.uint8
            ),
            "robot0_proprio-state": np.random.randn(68).astype(np.float32),
        }
        obs_result = obs_p({TransitionKey.OBSERVATION: obs})
        processed_obs = obs_result[TransitionKey.OBSERVATION]
        self.assertIn("observation.images.static", processed_obs)
        self.assertIn("observation.state.eef_pos", processed_obs)
        self.assertIn("observation.state.eef_quat", processed_obs)
        self.assertIn("observation.state.gripper_qpos", processed_obs)
        self.assertEqual(processed_obs["observation.state.eef_pos"].shape, (3,))

        # action 변환 (7D → 12D)
        model_action = np.array(
            [0.1, -0.2, 0.3, -0.1, 0.05, -0.05, 0.8], dtype=np.float32
        )
        act_result = act_p({TransitionKey.ACTION: model_action})
        action = act_result[TransitionKey.ACTION]
        self.assertEqual(action.shape, (12,))
        # gripper 복제 확인
        self.assertAlmostEqual(action[6], 0.8)
        self.assertAlmostEqual(action[7], 0.8)
        # base, torso = 0
        np.testing.assert_array_almost_equal(action[8:], [0.0, 0.0, 0.0, 0.0])


# ============================================================================
# Types
# ============================================================================

class TestTypes(unittest.TestCase):
    """types.py 기본 동작."""

    def test_create_transition(self):
        t = create_transition(
            observation={"a": 1},
            action=np.array([1, 2, 3]),
        )
        self.assertEqual(t[TransitionKey.OBSERVATION], {"a": 1})
        np.testing.assert_array_equal(t[TransitionKey.ACTION], [1, 2, 3])
        self.assertEqual(t[TransitionKey.INFO], {})

    def test_policy_feature_equality(self):
        f1 = PolicyFeature(FeatureType.VISUAL, (3, 224, 224))
        f2 = PolicyFeature(FeatureType.VISUAL, (3, 224, 224))
        f3 = PolicyFeature(FeatureType.STATE, (7,))
        self.assertEqual(f1, f2)
        self.assertNotEqual(f1, f3)

    def test_feature_type_values(self):
        self.assertEqual(FeatureType.STATE.value, "STATE")
        self.assertEqual(FeatureType.VISUAL.value, "VISUAL")
        self.assertEqual(FeatureType.ACTION.value, "ACTION")


if __name__ == "__main__":
    unittest.main()
