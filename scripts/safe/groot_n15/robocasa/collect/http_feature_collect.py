#!/usr/bin/env python3
"""Collect GR00T N1.5 LeRobot HTTP RoboCasa features as SAFE triplets.

The observation bridge follows the benchmark-style N1.5 LeRobot HTTP eval
client. The env/video/action-chunk loop and output contract follow the N1.6
SAFE collector:
``task{id}--ep{idx}--succ{0|1}.{pkl,csv,mp4}``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[5]
N15_EVAL_ROOT = REPO_ROOT / "scripts" / "safe" / "groot_n15" / "robocasa" / "eval"
N16_COLLECT_ROOT = REPO_ROOT / "scripts" / "safe" / "groot_n16" / "robocasa" / "collect"
N15_GROOT_ROOT = REPO_ROOT / "src" / "policies" / "Isaac-GR00T-N1.5"
N16_GROOT_ROOT = REPO_ROOT / "src" / "policies" / "Isaac-GR00T"
DEFAULT_MAX_EPISODE_STEPS = 720


def _prepend_path(path: Path) -> None:
    path_str = str(path)
    if not path.exists():
        return
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)


for path in reversed(
    (
        N16_GROOT_ROOT,
        N15_GROOT_ROOT,
        N16_COLLECT_ROOT,
        N15_EVAL_ROOT,
        REPO_ROOT / "scripts" / "utils",
        REPO_ROOT / "src" / "benchmarks" / "robocasa",
        REPO_ROOT / "src" / "benchmarks" / "robosuite",
        REPO_ROOT,
    )
):
    _prepend_path(path)

from collect_artifacts import (  # noqa: E402
    write_collect_ep_meta_manifest,
    write_safe_triplet,
)
from collect_schema import (  # noqa: E402
    _extract_groot_action_vector,
    _extract_safe_action_vector,
    _to_pickleable_numpy,
)
from lerobot_http_eval import (  # noqa: E402
    official_obs_to_lerobot_inputs,
    step_success,
)
from robocasa_event_labeler import make_robocasa_event_labeler  # noqa: E402
from env_step_phase import EnvStepGT, StepPhaseProbeWrapper, find_probe_wrapper  # noqa: E402
from perturbation import Perturber, PerturbSpec  # noqa: E402
from src.policies.groot.robocasa.io import convert_http_actions_to_groot_chunk  # noqa: E402
from src.policies.groot.robocasa.scenario_replay import (  # noqa: E402
    ep_meta_manifest_path,
    get_robocasa_ep_meta,
    json_safe,
    load_ep_meta_manifest,
    set_robocasa_ep_meta,
)
from src.policies.safe_metadata import normalize_feature_metadata  # noqa: E402
from vla_client import VLAClient  # noqa: E402


def _prefer_present(new: Any, old: Any) -> Any:
    return old if new is None else new


def _lerobot_action_to_official_chunk(action: dict[str, Any]) -> dict[str, np.ndarray]:
    chunk = convert_http_actions_to_groot_chunk(action)
    unbatched = {}
    for key, value in chunk.items():
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        unbatched[key] = arr
    return unbatched


class N15LerobotHttpFeatureClient(VLAClient):
    """VLAClient-backed LeRobot GR00T N1.5 policy client with SAFE records."""

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 300.0,
        inference_seed: int | None = None,
        no_features: bool = False,
        expect_chunk_len: int | None = None,
        attn_only: bool = False,
        instruction_override: str | None = None,
    ):
        super().__init__(url, timeout=timeout)
        self.inference_seed = inference_seed
        # exp4-2 B1: 모델에 보내는 instruction 만 교체 (env 는 원 과제로 정상 실행).
        # kitchen 은 lang 을 ep_meta 에서 읽지 않고 task 로직에서 재생성하므로 ep_meta
        # lang 편집은 무효 — 클라이언트 오버라이드가 유일한 타 instruction 주입 경로.
        self.instruction_override = instruction_override
        # exp3(구 pq3) eval 캡처-OFF 모드: skip_features 로 chunk 추론 경로만 사용, records 를
        # 만들지 않는다 (eval activation 미저장 규약). 추론 횟수는 n_calls 로 계수.
        self.no_features = no_features
        # cam-attention 전용 수집: record 에서 hidden_state/vl_hidden_state 텐서를
        # 버리고 cross_attn 요약만 유지 (pkl 수 MB 이하 — eval purge 규약 준수).
        self.attn_only = attn_only
        self.expect_chunk_len = expect_chunk_len
        self.n_calls = 0
        self.records: list[dict[str, Any]] = []
        self.task_description: str | None = None
        self.feature_kind: str | None = None
        self.capture_token_mode: str | None = None
        self.feature_axes: list[str] | None = None
        self.feature_slice: str | None = None
        self.exported_action_token_count: int | None = None
        self.feature_action_horizon: int | None = None
        self.valid_action_horizon: int | None = None
        self.model_action_horizon: int | None = None
        self.num_inference_timesteps: int | None = None
        self.capture_layers: list[int] | None = None
        self.layer_count: int | None = None
        self.token_count: int | None = None
        self.vl_feature_kind: str | None = None
        self.vl_feature_axes: list[str] | None = None
        self.vl_feature_dim: int | None = None
        self.cross_attn_axes: list[str] | None = None
        self.cross_attn_blocks: list[int] | None = None
        self.cross_attn_qgroups: list[str] | None = None
        self.cross_attn_kgroups: list[str] | None = None
        self.view_token_spans: list[list[int]] | None = None

    def reset(self) -> None:
        self.records.clear()
        self.n_calls = 0
        self.task_description = None
        self.feature_kind = None
        self.capture_token_mode = None
        self.feature_axes = None
        self.feature_slice = None
        self.exported_action_token_count = None
        self.feature_action_horizon = None
        self.valid_action_horizon = None
        self.model_action_horizon = None
        self.num_inference_timesteps = None
        self.capture_layers = None
        self.layer_count = None
        self.token_count = None
        self.vl_feature_kind = None
        self.vl_feature_axes = None
        self.vl_feature_dim = None
        self.cross_attn_axes = None
        self.cross_attn_blocks = None
        self.cross_attn_qgroups = None
        self.cross_attn_kgroups = None
        self.view_token_spans = None
        super().reset()

    def _update_metadata(self, features: dict[str, Any]) -> None:
        metadata = normalize_feature_metadata(features)
        self.feature_kind = _prefer_present(metadata.feature_kind, self.feature_kind)
        self.feature_axes = _prefer_present(metadata.feature_axes, self.feature_axes)
        self.feature_slice = _prefer_present(metadata.feature_slice, self.feature_slice)
        self.exported_action_token_count = _prefer_present(
            metadata.exported_action_token_count, self.exported_action_token_count
        )
        self.feature_action_horizon = _prefer_present(
            metadata.feature_action_horizon, self.feature_action_horizon
        )
        self.valid_action_horizon = _prefer_present(
            metadata.valid_action_horizon, self.valid_action_horizon
        )
        self.model_action_horizon = _prefer_present(
            metadata.model_action_horizon, self.model_action_horizon
        )
        self.num_inference_timesteps = _prefer_present(
            metadata.num_inference_timesteps, self.num_inference_timesteps
        )
        capture_layers = features.get("capture_layers", features.get("layer_indices"))
        self.capture_layers = _prefer_present(
            None if capture_layers is None else [int(layer) for layer in capture_layers],
            self.capture_layers,
        )
        self.layer_count = _prefer_present(features.get("layer_count"), self.layer_count)
        self.token_count = _prefer_present(features.get("token_count"), self.token_count)
        self.capture_token_mode = _prefer_present(
            features.get("capture_token_mode"), self.capture_token_mode
        )
        self.vl_feature_kind = _prefer_present(
            features.get("vl_feature_kind"), self.vl_feature_kind
        )
        self.vl_feature_axes = _prefer_present(
            features.get("vl_feature_axes"), self.vl_feature_axes
        )
        self.vl_feature_dim = _prefer_present(
            features.get("vl_feature_dim"), self.vl_feature_dim
        )
        self.cross_attn_axes = _prefer_present(
            features.get("cross_attn_axes"), self.cross_attn_axes
        )
        self.cross_attn_blocks = _prefer_present(
            features.get("cross_attn_blocks"), self.cross_attn_blocks
        )
        self.cross_attn_qgroups = _prefer_present(
            features.get("cross_attn_qgroups"), self.cross_attn_qgroups
        )
        self.cross_attn_kgroups = _prefer_present(
            features.get("cross_attn_kgroups"), self.cross_attn_kgroups
        )
        spans = features.get("view_token_spans")
        if spans is not None:
            spans = [[int(s), int(e)] for s, e in spans]
            if self.view_token_spans is not None and spans != self.view_token_spans:
                # 같은 에피소드 내 instruction/이미지 토큰 배치는 불변이어야 함.
                raise RuntimeError(
                    f"view_token_spans changed mid-episode: {self.view_token_spans} -> {spans}"
                )
            self.view_token_spans = spans

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del options
        images, states, instruction = official_obs_to_lerobot_inputs(observation)
        if self.instruction_override is not None:
            instruction = self.instruction_override
        if self.task_description is None:
            self.task_description = instruction
        # 캡처 모드는 구 배선(len(records)) 그대로, no-features 는 n_calls 로 동일 산법.
        calls_done = self.n_calls if self.no_features else len(self.records)
        seed_base = self.inference_seed
        # noise_resample arm (exp4-1): K번째 record 부터 offset 가산 → denoise noise 재샘플
        # (steering 없음 — 방향 없는 개입 대조. offset 고정이라 결정적.)
        if (seed_base is not None
                and getattr(self, "reseed_from_call", None) is not None
                and calls_done >= self.reseed_from_call):
            seed_base += self.reseed_offset
        call_inference_seed = None if seed_base is None else seed_base + calls_done
        if self.no_features:
            # /act 는 groot 16-큐 팝(16콜당 1추론)이라 캡처 경로와 실행 단위가 어긋남
            # (Gate 2 치명#1) — skip_features 로 /act_with_features 의 chunk 추론 경로를
            # hook 없이 그대로 사용 (매 콜 = denoise 1회, seed·phase 정합 보존).
            actions, features, _latency_ms = self.predict_with_features(
                images,
                states,
                instruction,
                inference_seed=call_inference_seed,
                extra_payload={"skip_features": 1},
            )
            self.n_calls += 1
            if features is not None:
                raise RuntimeError("skip_features 요청인데 features 가 반환됨 (serve 배선 확인)")
            if not isinstance(actions, dict):
                raise RuntimeError("LeRobot GR00T N1.5 /act_with_features must return sub-keyed actions")
            chunk = _lerobot_action_to_official_chunk(actions)
            n_steps = next(iter(chunk.values())).shape[0] if chunk else 0
            if self.expect_chunk_len and n_steps != self.expect_chunk_len:
                raise RuntimeError(
                    f"chunk 길이 {n_steps} != 기대 {self.expect_chunk_len} — "
                    "queue-pop 경로 혼입 의심 (Gate 2 치명#1 가드)"
                )
            return chunk, {}
        actions, features, _latency_ms = self.predict_with_features(
            images,
            states,
            instruction,
            inference_seed=call_inference_seed,
        )
        self.n_calls += 1
        if not isinstance(actions, dict):
            raise RuntimeError("LeRobot GR00T N1.5 /act_with_features must return sub-keyed actions")

        official_action = _lerobot_action_to_official_chunk(actions)
        if features is None:
            # 여기는 no_features 분기를 지난 캡처-ON 경로라 features 는 반드시 와야 한다.
            # 구 배선은 조용히 return 해서 record 를 안 만들었고, feature_phases 도 같이
            # 안 늘어나 run() 의 정합 검사(len(feature_phases)==n_inferences)를 통과했다
            # → 활성이 하나도 없는 pkl 이 [done] 으로 남는다. OFF 갈래의 역방향 가드
            # ("skip_features 인데 features 가 왔다")와 대칭이 되도록 fail-loud 로 바꾼다.
            raise RuntimeError(
                "캡처 ON 인데 features 가 반환되지 않음 — serve 가 --collect 로 떴는지, "
                "capture layer 인자가 걸렸는지 확인 (조용한 무캡처 방지)"
            )

        hidden_states = np.asarray(features["hidden_states"])
        if hidden_states.ndim == 4 and hidden_states.shape[0] == 1:
            hidden_states = hidden_states[0]
        record = {
            "hidden_state": torch.from_numpy(np.ascontiguousarray(hidden_states)),
            "action_vector": _extract_safe_action_vector(official_action),
            "groot_action_vector": _extract_groot_action_vector(official_action),
            "action": _to_pickleable_numpy(official_action),
            # Proprio state fed to this inference (paper expert-vs-VL state probe target).
            "state": _to_pickleable_numpy(states),
        }
        vl_hidden_states = features.get("vl_hidden_states")
        if vl_hidden_states is not None:
            record["vl_hidden_state"] = torch.from_numpy(
                np.ascontiguousarray(np.asarray(vl_hidden_states))
            )
        cross_attn = features.get("cross_attn")
        if cross_attn is not None:
            # [n_cross_blocks, K, qgroup, kgroup] float32 — 요약뿐이라 numpy 로 유지.
            record["cross_attn"] = np.asarray(cross_attn, dtype=np.float32)
        if self.attn_only:
            if cross_attn is None:
                raise RuntimeError(
                    "--attn-only-records 인데 features.cross_attn 이 없음 "
                    "(serve 를 --capture-cross-attn 으로 띄웠는지 확인)"
                )
            record.pop("hidden_state", None)
            record.pop("vl_hidden_state", None)
        self._update_metadata(features)
        self.records.append(record)
        return official_action, {}


def _post_steering_phase(server: str, phase: str) -> bool:
    """Oracle-gated steering: 현재 phase 를 serve 에 통지 (실패 시 즉시 예외 — 무음 미조향 방지).

    Returns: 해당 phase 에 실제 matrix 가 등록돼 있는지 (False = identity fallback —
    dose 로깅·부분 적용 감사용, Gate 2 높음#10).
    """
    import json as _json
    import urllib.request as _rq

    req = _rq.Request(
        f"{server.rstrip('/')}/steering_phase",
        data=_json.dumps({"phase": phase}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _rq.urlopen(req, timeout=5) as resp:
        if resp.status != 200:
            raise RuntimeError(f"/steering_phase HTTP {resp.status}")
        body = _json.loads(resp.read().decode() or "{}")
    return bool(body.get("gated", False))


def _get_serve_identity(server: str) -> dict:
    """serve /health 에서 GPU·boot 지문을 1회 조회 (exp4-1 arm×GPU confound 사후 감사용).

    실패해도 수집은 계속한다 (지문 없음으로 기록) — steering 무결성과 달리 감사 메타라
    fail-loud 대상이 아님.
    """
    import json as _json
    import urllib.request as _rq

    try:
        with _rq.urlopen(f"{server.rstrip('/')}/health", timeout=5) as resp:
            body = _json.loads(resp.read().decode() or "{}")
        return {
            "serve_gpu": body.get("serve_gpu"),
            "serve_boot_id": body.get("boot_id"),
            "serve_steering": body.get("steering"),
            # docs/04 규약 — rollout 인덱스의 machine·ckpt 열. serve 가 도는 머신이
            # 수집기와 다를 수 있으므로 /health 응답을 정본으로 쓴다.
            # machine 이 없으면 머신 간 재현 편차(같은 seed 에서 succ/fail 12.7% 반전)를
            # 층화할 수 없고, ckpt 가 없으면 베이스와 파인튜닝 rollout 이 섞인다.
            "machine": body.get("serve_machine"),
            "ckpt": body.get("serve_ckpt"),
        }
    except Exception as exc:  # noqa: BLE001
        return {"serve_gpu": None, "serve_boot_id": None, "machine": None, "ckpt": None,
                "serve_health_error": str(exc)}


def _find_latest_video(video_dir: Path) -> Path | None:
    videos = sorted(video_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
    return videos[-1] if videos else None


def _append_summary(summary_path: Path, task_id: int, task: str, episode_idx: int, seed: int | None, pkl_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_path.exists():
        summary_path.write_text("task_id\ttask\tepisode_idx\tseed\texit_code\tpkl\n")
    with summary_path.open("a") as f:
        f.write(
            f"{task_id}\t{task}\t{episode_idx}\t"
            f"{'' if seed is None else seed}\t0\t{pkl_path}\n"
        )


def _action_tremor_summary(
    traj: list, steer_on: list, t0_record: "int | None"
) -> "dict[str, Any] | None":
    """개입(steering)이 액션 궤적에 주는 떨림(jitter) 진단 — record 해상도.

    traj = record 별 groot action 벡터(`_extract_groot_action_vector`, 첫 실행 스텝) → [R, D].
    replan(get_action) 단위로 명령이 얼마나 튀는지를 잰다:
      speed_r = ‖a_r − a_{r-1}‖   (record 간 명령 변화율)
      jerk_r  = ‖Δ³a_r‖           (3차 차분 = 고주파 떨림 대리지표)
    base(무개입/개입 전)와 개입 이후를 나눠 저장 → offline 에서 같은 episode 의 base arm
    (noise_resample/A0) 대비 개입 후 떨림 증가량(Δjerk)을 비교한다. steer_on 은 record 별
    개입 활성 flag (steer_from_record, 없으면 reseed_from_record 기준). 판정에는 무영향,
    사이드카에 진단 필드만 추가한다.
    """
    if not traj or len(traj) < 4:
        return None
    A = np.asarray(traj, dtype=np.float64)                                   # [R, D]
    R, D = A.shape[0], (A.shape[1] if A.ndim == 2 else 1)
    if A.ndim == 1:
        A = A.reshape(R, 1)
    speed = np.concatenate([[0.0], np.linalg.norm(np.diff(A, axis=0), axis=1)])          # [R]
    jerk = np.concatenate([[0.0, 0.0, 0.0], np.linalg.norm(np.diff(A, n=3, axis=0), axis=1)])
    steer = np.asarray(steer_on, dtype=bool)[:R]

    def _agg(mask: "np.ndarray") -> "dict[str, Any] | None":
        m = np.asarray(mask, dtype=bool)
        if int(m.sum()) < 2:
            return None
        return {
            "n": int(m.sum()),
            "speed_mean": float(speed[m].mean()),
            "speed_p95": float(np.percentile(speed[m], 95)),
            "jerk_mean": float(jerk[m].mean()),
            "jerk_p95": float(np.percentile(jerk[m], 95)),
            "jerk_max": float(jerk[m].max()),
        }

    return {
        "dim": int(D),
        "n_records": int(R),
        "t0_record": (int(t0_record) if t0_record is not None else None),
        # record 해상도 배열 — 개입 전후 정렬·base arm 대조를 offline 에서 하기 위한 원자료
        "speed": [round(float(x), 6) for x in speed],
        "jerk": [round(float(x), 6) for x in jerk],
        # 개입 활성 여부로 split (base arm 은 pre 만; steered arm 은 post 가 개입 구간)
        "pre_intervention": _agg(~steer),
        "post_intervention": _agg(steer),
    }


def _resolve_task_dir(root_or_task_dir: Path, task: str) -> Path:
    """Return the task-specific directory used by the N1.6 collection layout."""

    return root_or_task_dir if root_or_task_dir.name == task else root_or_task_dir / task


def _resolve_cell_dir(root_or_cell_dir: Path, task: str, cell_id: str) -> Path:
    if root_or_cell_dir.name == cell_id and root_or_cell_dir.parent.name == task:
        return root_or_cell_dir
    if root_or_cell_dir.name == task:
        return root_or_cell_dir / cell_id
    return root_or_cell_dir / task / cell_id


def _normalize_instruction(value: str) -> str:
    return " ".join(value.strip().split())


def make_env(
    task: str,
    split: str,
    *,
    env_name: str,
    scenario_seed: int | None = None,
    video_dir: Path | None = None,
    video_fps: int = 20,
    steps_per_render: int = 2,
    overlay_text: bool = True,
    n_action_steps: int = 16,
    max_episode_steps: int = 720,
):
    del task, split
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
    import robosuite  # noqa: F401
    from gr00t.eval.rollout_policy import MultiStepConfig, VideoConfig, WrapperConfigs
    from src.policies.groot.robocasa.env_wrappers import wrap_groot_robocasa_eval_env

    env = gym.make(env_name, enable_render=True, seed=scenario_seed)
    env = StepPhaseProbeWrapper(env)  # env-step GT probe (MultiStep 아래층)
    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=None if video_dir is None else str(video_dir),
            fps=video_fps,
            steps_per_render=steps_per_render,
            max_episode_steps=max_episode_steps,
            overlay_text=overlay_text,   # --no-overlay → 캡션 없는 클린 영상
        ),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        )
    )
    return wrap_groot_robocasa_eval_env(env, wrapper_configs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vla-server", default="http://127.0.0.1:8400")
    parser.add_argument("--task", default="OpenFridge")
    parser.add_argument(
        "--env-name",
        required=True,
        help=(
            "Explicit RoboCasa env id. Use the same env id as N1.6 collection, "
            "e.g. robocasa_panda_omron/CloseFridge_PandaOmron_Env."
        ),
    )
    parser.add_argument("--split", choices=["pretrain", "target"], default="target")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--cell-id", default=None)
    parser.add_argument("--cell-index", type=int, default=None)
    parser.add_argument("--canonical-instruction", default=None)
    parser.add_argument("--task-description", default=None)
    parser.add_argument("--episode-start-idx", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--inference-seed", type=int, default=None)
    parser.add_argument(
        "--proximity-phases",
        action="store_true",
        help="6-phase 라벨(reach/grasp/transport/place/insert-settle/terminal, proximity 기반 causal).",
    )
    parser.add_argument(
        "--env-step-phases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="매 env-step 후 별도 라벨러로 phase/성공 GT 기록 (env_step_* pkl 필드)",
    )
    parser.add_argument(
        "--no-features",
        action="store_true",
        help=(
            "exp3 eval 캡처-OFF 모드: /act_with_features 를 skip_features 로 호출해 "
            "캡처 없는 chunk 추론 경로를 사용하고 activation 을 저장하지 않는다. "
            "산출물은 pkl 삼중항 대신 경량 판정 사이드카 task*--ep*--succ{0|1}.json + mp4 "
            "(eval activation 미저장 규약, memory: eval-activation-purged)."
        ),
    )
    parser.add_argument(
        "--attn-only-records",
        action="store_true",
        help=(
            "cam-attention 전용 수집: record 에서 hidden_state/vl_hidden_state 텐서를 "
            "버리고 cross_attn 요약(뷰별 mass)만 pkl 에 기록 (serve 는 --capture-cross-attn "
            "필요, pkl 수 MB 이하)."
        ),
    )
    parser.add_argument(
        "--expect-chunk-len",
        type=int,
        default=16,
        help="--no-features 시 응답 chunk 길이 assert (queue-pop 혼입 가드, 기본 16)",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="러너 ARM_SPEC sha 태그 — 사이드카에 기록해 재라벨링 오염을 집계가 탐지",
    )
    parser.add_argument(
        "--gated-steering",
        action="store_true",
        help=(
            "Oracle phase-gated steering: 매 get_action 전에 현재 phase 를 serve 의 "
            "/steering_phase 로 POST 해 conceptor 를 스위칭 (serve 는 --steering-phase-npz-base 필요)."
        ),
    )
    parser.add_argument(
        "--instruction-override",
        default=None,
        help=(
            "모델에 보내는 instruction 을 이 텍스트로 교체 (env 는 원 과제 그대로 — "
            "exp4-2 B1 타 instruction VL donor 수집용). --canonical-instruction 도 "
            "같은 값으로 지정할 것 (사이드카 ep_meta.lang 에 env 원 지시가 남음)."
        ),
    )
    parser.add_argument(
        "--perturb-spec",
        default=None,
        help=(
            "exp4-2 Track P 섭동 spec — inline json('{'로 시작) 또는 json 파일 경로 "
            "(perturbation.PerturbSpec 스키마). resolved spec/창/실측치는 "
            "사이드카·pkl 의 perturb_* 키로 기록된다."
        ),
    )
    parser.add_argument(
        "--steer-from-record",
        type=int,
        default=None,
        metavar="K",
        help=(
            "exp4-1 oracle latch: inference 직전 record 수(progress_before)>=K 이면 "
            "steering ON POST, 아니면 'off' — K번째 record 부터 episode 끝까지. "
            "POST 값은 --steer-phase-mode 참조. --gated-steering 과 상호배타."
        ),
    )
    parser.add_argument(
        "--steer-phase-name",
        default=None,
        metavar="NAME",
        help=(
            "latch ON 시 POST 할 phase 이름 (기본 'steer'). LOO arm 은 ep{E} 를 넘겨 "
            "serve 의 phase registry 에 등록된 per-episode 연산자를 스위칭 — serve 재기동 0회."
        ),
    )
    parser.add_argument(
        "--steer-phase-mode",
        choices=("global", "current"),
        default="global",
        help=(
            "latch ON 시 POST 할 phase: global='steer' 고정(permanent arm — serve 는 "
            "phase 디렉토리 steer 하나) | current=라벨러의 현재 phase 명(gated arm — "
            "serve 는 phase 별 NPZ 등록, 미등록 phase 는 identity)."
        ),
    )
    parser.add_argument(
        "--reseed-from-record",
        type=int,
        default=None,
        metavar="K",
        help=(
            "noise_resample arm: K번째 record 부터 inference_seed 에 --reseed-offset 을 "
            "더해 denoise noise 를 재샘플 (steering 없음 — 방향 없는 개입 대조). "
            "결정적(offset 고정)."
        ),
    )
    parser.add_argument(
        "--reseed-offset",
        type=int,
        default=500000,
        help="--reseed-from-record 의 seed offset (사전등록 고정값)",
    )
    parser.add_argument(
        "--n_action_steps",
        "--n-action-steps",
        dest="n_action_steps",
        type=int,
        default=16,
        help="Number of leading decoded actions to execute per policy inference.",
    )
    parser.add_argument(
        "--ep-meta-dir",
        type=Path,
        default=None,
        help="Optional directory for seed-keyed RoboCasa ep_meta JSON import/export.",
    )
    parser.add_argument(
        "--ep-meta-load-env-name",
        default=None,
        help=(
            "Optional env name used only when loading ep_meta manifests. This lets "
            "N1.5 replay manifests exported by the N1.6 env id."
        ),
    )
    parser.add_argument(
        "--probe-seeds",
        default=None,
        metavar="S1,S2,...",
        help=(
            "exp5-4 probe-only 모드: 콤마 구분 inference_seed 목록. --probe-out 과 함께 "
            "지정하면 env.reset 직후 t=0 관측으로 seed 별 첫 inference(=record 0) 활성만 "
            "뽑고 즉시 종료한다 (env.step·labeler·pkl/json 산출 없음)."
        ),
    )
    parser.add_argument(
        "--probe-out",
        default=None,
        metavar="PATH.npz",
        help="probe-only 모드 산출 npz 경로 (--probe-seeds 와 동시 지정 필수).",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--wait-ready", action="store_true")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--steps-per-render", type=int, default=2)
    parser.add_argument("--no-overlay", action="store_true",
                        help="영상에 instruction/success 캡션을 안 그림(클린 mp4). "
                             "instruction은 ep_meta.lang 에 그대로 남음.")
    return parser.parse_args()


# probe 경로(_ProbeClient/_obs_hash/_flatten_action_chunk/_run_probe, 205줄)는
# exp5-4 전용이라 steer/exp5_4/probe_lib.py 로 분리했다(2026-07-31). CLI 계약은
# 그대로 유지 — run() 이 --probe-seeds 를 받아 아래 run_probe 로 위임하므로
# probe_collect.sh 는 수정 불필요. 같은 모듈의 _ProbeClient/_obs_hash/
# _flatten_action_chunk 는 exp5_4/smoke_probe.py(캡처 hook 무해성·서버 오염 검출)와
# check_probe_identity.py 가 계속 쓴다 — 라운드 종결과 무관한 무결성 도구다.


def _run_probe(args):
    _prepend_path(REPO_ROOT / "scripts" / "safe" / "groot_n15" / "robocasa" / "steer" / "exp5_4")
    from probe_lib import run_probe  # noqa: PLC0415  (순환 import 회피 — 지연 로드)

    return run_probe(args)




def run() -> dict[str, Any]:
    args = parse_args()
    probe_seeds = getattr(args, "probe_seeds", None)
    probe_out = getattr(args, "probe_out", None)
    if (probe_seeds is None) != (probe_out is None):
        raise ValueError("--probe-seeds 와 --probe-out 은 함께 지정해야 함")
    if probe_seeds is not None:
        # probe-only: 아래 rollout/저장 경로를 전혀 타지 않는다.
        return _run_probe(args)
    if args.n_action_steps <= 0:
        raise ValueError(f"--n_action_steps must be positive: {args.n_action_steps}")
    if args.max_episode_steps <= 0:
        raise ValueError(f"--max-episode-steps must be positive: {args.max_episode_steps}")
    output_root = Path(args.output_dir)
    cell_id = getattr(args, "cell_id", None)
    cell_index = getattr(args, "cell_index", None)
    canonical_instruction = getattr(args, "canonical_instruction", None)
    if cell_id is not None and cell_index is None:
        raise ValueError("--cell-index is required when --cell-id is set")
    if cell_id is not None and canonical_instruction is None:
        raise ValueError("--canonical-instruction is required when --cell-id is set")
    effective_task_id = args.task_id if cell_index is None else cell_index

    output_dir = (
        _resolve_task_dir(output_root, args.task)
        if cell_id is None
        else _resolve_cell_dir(output_root, args.task, cell_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        output_root / "collection_summary.tsv"
        if cell_id is not None
        else output_dir.parent / "collection_summary.tsv"
    )
    ep_meta_dir = (
        None
        if args.ep_meta_dir is None
        else (
            _resolve_task_dir(Path(args.ep_meta_dir), args.task)
            if cell_id is None
            else _resolve_cell_dir(Path(args.ep_meta_dir), args.task, cell_id)
        )
    )
    if args.ep_meta_load_env_name is not None and ep_meta_dir is None:
        raise ValueError("--ep-meta-dir is required when --ep-meta-load-env-name is set")

    policy = N15LerobotHttpFeatureClient(
        args.vla_server,
        timeout=args.timeout,
        inference_seed=args.inference_seed,
        no_features=getattr(args, "no_features", False),
        expect_chunk_len=(
            getattr(args, "expect_chunk_len", None)
            if getattr(args, "no_features", False)
            else None
        ),
        attn_only=getattr(args, "attn_only_records", False),
        instruction_override=getattr(args, "instruction_override", None),
    )
    if getattr(args, "attn_only_records", False) and getattr(args, "no_features", False):
        raise ValueError("--attn-only-records 와 --no-features 는 동시 사용 불가")
    steer_from_record = getattr(args, "steer_from_record", None)
    if steer_from_record is not None and getattr(args, "gated_steering", False):
        raise ValueError("--steer-from-record 와 --gated-steering 은 상호배타 (latch vs phase-gated)")
    if steer_from_record is not None and steer_from_record < 0:
        raise ValueError(f"--steer-from-record 는 음수 불가: {steer_from_record}")
    reseed_from_record = getattr(args, "reseed_from_record", None)
    if reseed_from_record is not None and steer_from_record is not None:
        raise ValueError("--reseed-from-record 는 steering latch 와 상호배타 (noise_resample 단독 arm)")
    if reseed_from_record is not None:
        policy.reseed_from_call = int(reseed_from_record)
        policy.reseed_offset = int(getattr(args, "reseed_offset", 500000))
    if args.wait_ready:
        policy.wait_until_ready(max_wait=args.timeout)
    serve_identity = _get_serve_identity(args.vla_server)

    max_steps = args.max_episode_steps
    results: list[dict[str, Any]] = []
    env_name = args.env_name
    for local_ep_idx in range(args.n_episodes):
        episode_idx = args.episode_start_idx + local_ep_idx
        scenario_seed = None if args.seed is None else args.seed + local_ep_idx
        ep_meta_path = None
        replay_ep_meta = None
        if ep_meta_dir is not None and scenario_seed is not None:
            replay_requested = args.ep_meta_load_env_name is not None
            load_env_name = args.ep_meta_load_env_name if replay_requested else env_name
            ep_meta_path = ep_meta_manifest_path(ep_meta_dir, load_env_name, scenario_seed)
            if replay_requested and ep_meta_path.exists():
                replay_ep_meta = load_ep_meta_manifest(
                    ep_meta_path,
                    env_name=load_env_name,
                    scenario_seed=scenario_seed,
                )
            elif replay_requested:
                raise FileNotFoundError(
                    "--ep-meta-load-env-name requested but ep_meta manifest is missing: "
                    f"{ep_meta_path}"
                )

        upstream_video_dir = output_dir / ".groot_video_tmp" / f"task{effective_task_id}--ep{episode_idx}"
        if upstream_video_dir.exists():
            shutil.rmtree(upstream_video_dir)
        upstream_video_dir.mkdir(parents=True, exist_ok=True)
        env = make_env(
            args.task,
            args.split,
            env_name=args.env_name,
            scenario_seed=scenario_seed,
            video_dir=upstream_video_dir,
            video_fps=args.video_fps,
            steps_per_render=args.steps_per_render,
            overlay_text=not args.no_overlay,   # --no-overlay → 캡션 없는 클린 영상
            n_action_steps=args.n_action_steps,
            max_episode_steps=max_steps,
        )
        try:
            if replay_ep_meta is not None:
                set_robocasa_ep_meta(env, replay_ep_meta)
            obs, _info = env.reset(seed=scenario_seed)
            captured_ep_meta = replay_ep_meta or get_robocasa_ep_meta(env)
            if (
                ep_meta_dir is not None
                and scenario_seed is not None
                and replay_ep_meta is None
            ):
                write_collect_ep_meta_manifest(
                    ep_meta_manifest_path(ep_meta_dir, env_name, scenario_seed),
                    env_name=env_name,
                    scenario_seed=scenario_seed,
                    ep_meta=captured_ep_meta,
                    robocasa_env_source="robocasa365",
                )
            policy.reset()
            # Event-anchored phase labeler: one labeler.step() per get_action (= one SAFE
            # feature record = n_action_steps env-steps), called BEFORE env.step so the phase
            # matches the state the policy acted on. Aligns 1:1 with policy.records.
            labeler = make_robocasa_event_labeler(
                env, env_name, proximity_phases=getattr(args, "proximity_phases", False)
            )
            labeler.reset()
            env_gt = None
            if getattr(args, "env_step_phases", True):
                _probe = find_probe_wrapper(env)
                if _probe is not None:
                    env_gt = EnvStepGT(env, env_name, getattr(args, "proximity_phases", False))
                    _probe.set_gt(env_gt)
                    env_gt.start()
            perturber = None
            if getattr(args, "perturb_spec", None):
                # env_gt.start() 이후에 실행해 G1 scripted env-step 도 GT 타임라인에 계수
                # (record↔env-step 정렬은 perturb_env_step_offset 으로 보정).
                perturber = Perturber(
                    env,
                    PerturbSpec.from_json(args.perturb_spec),
                    n_action_steps=args.n_action_steps,
                )
                obs = perturber.on_episode_start(obs)
            feature_phases: list[str] = []
            phase_gated_flags: list[bool] = []
            success = False
            first_success_step = None
            step_i = 0
            no_features = getattr(args, "no_features", False)
            # 떨림(jitter) 진단: record(=replan) 별 action 벡터·개입 활성 flag 누적.
            # 개입점 = steer_from_record(steered arm) 없으면 reseed_from_record(noise base) —
            # base·steered 를 같은 t0 로 split 해 개입 후 떨림 증가량을 대조한다.
            action_traj: list = []
            action_steer_on: list = []
            _interv_record = (
                steer_from_record if steer_from_record is not None else reseed_from_record
            )
            while step_i < max_steps:
                # no-features(캡처 OFF) 모드는 records 가 없으므로 추론 호출 수(n_calls)로
                # phase 정합을 잡는다 (skip_features 는 매 콜 chunk 추론 — 두 계수 동치).
                progress_before = policy.n_calls if no_features else len(policy.records)
                # labeler.step() reads the CURRENT sim state (= the obs the policy will act
                # on; env.step has not run yet), so calling it before or after get_action is
                # equivalent for alignment. Gated steering needs the phase BEFORE inference
                # to switch the serve-side conceptor for this call.
                phase = labeler.step()
                gated_now = False
                if getattr(args, "gated_steering", False):
                    gated_now = _post_steering_phase(args.vla_server, phase)
                elif steer_from_record is not None:
                    # exp4-1 latch: K번째 record 부터 끝까지 ON (record r ⇔ env-step
                    # [nas·r, nas·r+nas)). global='steer' 고정 / current=라벨러 현재 phase
                    # (gated arm — 미등록 phase 는 serve identity 폴백).
                    if progress_before < steer_from_record:
                        want = "off"
                    elif getattr(args, "steer_phase_mode", "global") == "current":
                        want = phase
                    else:
                        want = getattr(args, "steer_phase_name", None) or "steer"
                    gated_now = _post_steering_phase(args.vla_server, want)
                official_action, _ = policy.get_action(obs)
                if perturber is not None:
                    official_action = perturber.maybe_apply(progress_before, official_action)
                progress_after = policy.n_calls if no_features else len(policy.records)
                if progress_after > progress_before:
                    feature_phases.append(phase)
                    phase_gated_flags.append(gated_now)
                    # 떨림 진단: 이 record 의 명령 action 벡터(첫 실행 스텝)와 개입 활성 여부.
                    action_traj.append(_extract_groot_action_vector(official_action))
                    action_steer_on.append(
                        _interv_record is not None and progress_before >= _interv_record
                    )
                obs, reward, terminated, truncated, info = env.step(official_action)
                step_i += 1
                success_now = step_success(reward, info, env=env)
                if success_now and first_success_step is None:
                    first_success_step = step_i
                success = success or success_now
                if terminated or truncated or success:
                    break
            # Terminal off-by-one: advance the timeline/event_steps once past the loop end
            # (no feature record here, so this phase is not appended to feature_phases).
            labeler.step()
            n_inferences = policy.n_calls if no_features else len(policy.records)
            if len(feature_phases) != n_inferences:
                raise RuntimeError(
                    "feature_phases/inference-count misaligned: "
                    f"{len(feature_phases)} != {n_inferences}"
                )

            rendered_video = env.render()
            upstream_video_path = _find_latest_video(upstream_video_dir)
            if upstream_video_path is None and rendered_video is not None:
                upstream_video_path = Path(rendered_video)
            stem = f"task{effective_task_id}--ep{episode_idx}--succ{int(success)}"
            task_description = args.task_description or policy.task_description or args.task
            if canonical_instruction is not None and (
                _normalize_instruction(task_description)
                != _normalize_instruction(canonical_instruction)
            ):
                raise RuntimeError(
                    "Collected task description does not match canonical instruction: "
                    f"{task_description!r} != {canonical_instruction!r}"
                )
            extra_metadata: dict[str, Any] = {
                # State-based phase labels aligned 1:1 with hidden_states records
                # (non-monotone: drop reverts transport->reach, re-grasp allowed).
                "phase_scheme": "event_state",
                "feature_phases": feature_phases,
                "phase_timeline": labeler.phase_timeline,
                **(env_gt.export(feature_phases, args.n_action_steps) if env_gt is not None else {}),
                "event_steps": labeler.event_steps,
                "event_order": labeler.event_order_keys,
                # drop-aware fields (per-step grasp signal for offline drop analysis)
                "grasp_steps": list(getattr(labeler, "grasp_steps", []) or []),
                "drop_steps": list(getattr(labeler, "drop_steps", []) or []),
                "grasp_count": int(getattr(labeler, "grasp_count", 0) or 0),
                "grasp_timeline": list(getattr(labeler, "grasp_timeline", []) or []),
                "wrong_grasp_steps": list(getattr(labeler, "wrong_grasp_steps", []) or []),
                "wrong_grasp_timeline": list(getattr(labeler, "wrong_grasp_timeline", []) or []),
            }
            # 떨림(jitter) 진단 — 개입이 액션 궤적에 준 speed/jerk (record 해상도, 판정 무영향).
            extra_metadata["action_kinematics"] = _action_tremor_summary(
                action_traj, action_steer_on, _interv_record
            )
            # exp2(구 pq2) 이중 채점 (docs/steering/18): apple corrected env fork 가 노출하는
            # (_pq2_* attr 명은 robocasa fork 계약 — rename 금지)
            # per-episode ever 플래그 {"0.07": bool, "0.1": bool}. 미지원 task 는 None.
            _kenv = env
            while hasattr(_kenv, "env") and not hasattr(_kenv, "_pq2_succ_ever"):
                _kenv = _kenv.env
            _ever = getattr(_kenv, "_pq2_succ_ever", None)
            extra_metadata["succ_ever_th"] = dict(_ever) if _ever is not None else None
            if perturber is not None:
                extra_metadata.update(perturber.export())
            # serve 지문(serve_gpu/serve_boot_id/serve_steering)은 캡처 ON(pkl) 경로에도 남긴다.
            # 구 배선은 사이드카(캡처 OFF)에만 있어 fit 재료 pkl 에서 ① 어느 GPU·serve 인스턴스
            # 에서 나온 활성인지 ② 그 serve 에 steering 이 걸려 있지 않았는지를 사후 확인할 수
            # 없었다. ②는 fit 재료의 전제(무개입)라 GPU 추적과 달리 대체 경로가 없다.
            # (2026-07-31 이전 수집분에는 이 필드가 없다 — 소급 불가.)
            extra_metadata.update(serve_identity)
            if cell_id is not None:
                extra_metadata.update(
                    {
                        "cell_id": cell_id,
                        "cell_index": effective_task_id,
                        "robocasa_task": args.task,
                        "canonical_instruction": canonical_instruction,
                    }
                )
            if no_features:
                # exp3 eval 캡처-OFF: 판정 데이터만 — 경량 json 사이드카(스템 규약 동일) + mp4.
                sidecar = {
                    "task_suite_name": "lerobot_groot_n15_robocasa",
                    "model_family": "lerobot_groot_n15",
                    "policy_transport": "http",
                    "capture": "none",
                    "task_id": effective_task_id,
                    "task": args.task,
                    "task_description": task_description,
                    "episode_idx": episode_idx,
                    "scenario_seed": scenario_seed,
                    "seed": scenario_seed,
                    "inference_seed": args.inference_seed,
                    "episode_success": int(success),
                    "first_success_step": first_success_step,
                    "steps": step_i,
                    "n_inferences": n_inferences,
                    "n_action_steps": args.n_action_steps,
                    "max_episode_steps": max_steps,
                    "env_name": env_name,
                    "robocasa_env_source": "robocasa365",
                    "run_tag": getattr(args, "run_tag", None),
                    "expect_chunk_len": policy.expect_chunk_len,
                    # gated dose 감사: 추론별 phase 가 실제 matrix 를 탔는지 (False=identity)
                    "phase_gated_flags": phase_gated_flags,
                    # exp4-1 latch 감사: sum(flags)==n_inferences-K && first_true==K 대조용
                    "steer_from_record": steer_from_record,
                    "steer_phase_mode": getattr(args, "steer_phase_mode", "global"),
                    "steer_phase_name": getattr(args, "steer_phase_name", None),
                    "reseed_from_record": reseed_from_record,
                    "reseed_offset": (int(getattr(args, "reseed_offset", 500000))
                                      if reseed_from_record is not None else None),
                    **serve_identity,
                    "ep_meta": captured_ep_meta,
                    **extra_metadata,
                }
                out_path = output_dir / f"{stem}.json"
                out_path.write_text(
                    json.dumps(json_safe(sidecar), indent=2, ensure_ascii=False)
                )
                if upstream_video_path is not None and upstream_video_path.exists():
                    shutil.copy2(upstream_video_path, output_dir / f"{stem}.mp4")
                shutil.rmtree(upstream_video_dir, ignore_errors=True)
            else:
                write_safe_triplet(
                    output_dir=output_dir,
                    stem=stem,
                    policy=policy,
                    task_id=effective_task_id,
                    task_description=task_description,
                    episode_idx=episode_idx,
                    scenario_seed=scenario_seed,
                    episode_success=success,
                    env_name=env_name,
                    upstream_video_path=upstream_video_path,
                    ep_meta=captured_ep_meta,
                    n_action_steps=args.n_action_steps,
                    robocasa_env_source="robocasa365",
                    max_episode_steps=max_steps,
                    video_fps=args.video_fps,
                    steps_per_render=args.steps_per_render,
                    inference_seed=args.inference_seed,
                    model_family="lerobot_groot_n15",
                    policy_transport="http",
                    task_suite_name="lerobot_groot_n15_robocasa",
                    extra_metadata=extra_metadata,
                    include_hidden_states=not getattr(args, "attn_only_records", False),
                )
                out_path = output_dir / f"{stem}.pkl"
            _append_summary(summary_path, effective_task_id, args.task, episode_idx, scenario_seed, out_path)
            results.append(
                {
                    "episode_idx": episode_idx,
                    "success": success,
                    "first_success_step": first_success_step,
                    "steps": step_i,
                    "records": n_inferences,
                    "feature_phases": list(feature_phases),
                    "event_steps": dict(labeler.event_steps),
                    "max_phase": labeler.max_phase_label,
                    "pkl": str(out_path),
                }
            )
            print(
                f"wrote {stem}: steps={step_i} success={int(success)} "
                f"records={n_inferences} feature_kind={policy.feature_kind} "
                f"capture={'off' if no_features else 'on'} "
                f"max_phase={labeler.max_phase_label} events={dict(labeler.event_steps)}"
            )
        finally:
            env.close()

    return {"episodes": results}


if __name__ == "__main__":
    run()
