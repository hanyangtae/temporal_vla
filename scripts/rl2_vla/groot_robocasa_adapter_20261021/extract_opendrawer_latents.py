"""OpenDrawer 20250819 expert episode → frozen GR00T N1.5 decoder-입력 latent 추출 (lerobot 컨테이너, GPU).

실제 serve 경로를 그대로 쓴다: ``scripts/serve/lerobot.py`` 의 ``parse_payload`` →
``_apply_input_remap`` → ``GrootPolicyAdapter`` preprocessor → ``GrootPolicy.predict_action_chunk``.
feature 는 action_decoder forward pre-hook (``composition.FirstCallCapture``) 로 첫 denoise call
입력만 캡처해 last16 평균 1024 로 만든다 (signature ccbfb51ca01724cb). 결정 시점 0,5,10,...<T.
expert target 은 데이터셋 action 이다 (모델 예측 아님). 정본 규약: CONTRACT.md §11, baseline_data.py.

이 스크립트는 GPU lease·컨테이너 발사를 하지 않는다 (메인이 담당). 실행 예 (컨테이너 안, repo=/temporal_vla):

  HF_HUB_OFFLINE=1 python /temporal_vla/scripts/rl2_vla/groot_robocasa_adapter_20261021/extract_opendrawer_latents.py \
    --data-root /cache/datasets --out /temporal_vla/outputs/rl2_vla/<new_dir> \
    --image-id <docker image id> --device cuda:0 [--episodes 0-1]

출력 폴더 규약·resume 규칙은 baseline_data.py 모듈 docstring 참고. 이미 commit 된 episode 는 계약·지문
검증 후 건너뛰고, 다르면 거부한다 (덮어쓰기 없음).
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parents[2]
for _p in (ADAPTER_DIR, REPO_ROOT, REPO_ROOT / "scripts" / "utils"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from baseline_data import (  # noqa: E402
    CHECKPOINT_REVISION,
    EXPECTED_FEATURE_SIGNATURE,
    commit_episode,
    contract_sha256,
    decision_seed,
    decision_steps,
    ensure_same_or_create,
    episode_index_list,
    extraction_contract,
    sha256_path,
    verify_committed_episode,
    build_episode_arrays,
)
from contract import (  # noqa: E402
    CHECKPOINT_SUBFOLDER,
    ContractError,
    HF_REPO_CACHE_DIR,
    NUM_INFERENCE_STEPS,
    load_checkpoint_contract,
    raw_to_model_index,
)
from data_contract import (  # noqa: E402
    AUX_SPLIT_V2_REL,
    AUX_SPLIT_V2_SHA256,
    OPENDRAWER_PACKAGE_REL,
    validate_aux_split,
)

PROFILE_REL = "configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml"
SERVE_REL = "scripts/serve/lerobot.py"
CODE_FILES = (
    "scripts/rl2_vla/groot_robocasa_adapter_20261021/extract_opendrawer_latents.py",
    "scripts/rl2_vla/groot_robocasa_adapter_20261021/baseline_data.py",
    "scripts/rl2_vla/groot_robocasa_adapter_20261021/composition.py",
    "scripts/rl2_vla/groot_robocasa_adapter_20261021/contract.py",
    "scripts/rl2_vla/groot_robocasa_adapter_20261021/data_contract.py",
    "scripts/rl2_vla/groot_robocasa_adapter_20261021/transitions.py",
    SERVE_REL,
    "scripts/serve/lerobot_adapters/groot.py",
    "scripts/serve/lerobot_adapters/common.py",
    "src/policies/groot/core/schema.py",
    "lerobot/src/lerobot/policies/groot/modeling_groot.py",
    "lerobot/src/lerobot/policies/groot/action_head/flow_matching_action_head.py",
)
SERVE_MODULE_NAME = "_temporal_vla_serve_lerobot_for_extraction"


def code_hashes() -> dict:
    return {rel: sha256_path(REPO_ROOT / rel) for rel in CODE_FILES}


# ---------------------------------------------------------------------------
# runtime (컨테이너 전용 import)
# ---------------------------------------------------------------------------

def import_serve_module():
    """lerobot 패키지를 먼저 import 한 뒤 serve lerobot.py 를 다른 이름으로 불러온다.

    serve 파일은 scripts/serve 를 sys.path 앞에 넣는다. 그 뒤에 처음 ``import lerobot`` 을 하면
    serve 파일(lerobot.py) 이 패키지를 가린다 → 패키지를 먼저 sys.modules 에 올려 둔다.
    """
    lerobot_src = REPO_ROOT / "lerobot" / "src"
    if lerobot_src.is_dir() and str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    import lerobot  # noqa: F401  (패키지 선점)
    import lerobot.policies.factory  # noqa: F401
    import lerobot.policies.groot.modeling_groot  # noqa: F401

    pkg_file = Path(lerobot.__file__).resolve()
    if pkg_file.name != "__init__.py":
        raise ContractError(f"'lerobot' 이 패키지가 아님 (shadowing): {pkg_file}")
    spec = importlib.util.spec_from_file_location(SERVE_MODULE_NAME, REPO_ROOT / SERVE_REL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[SERVE_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    if Path(sys.modules["lerobot"].__file__).resolve() != pkg_file:
        raise ContractError("serve 모듈 로드 후 lerobot 패키지가 바뀜 (shadowing)")
    return mod


def checkpoint_dir(data_root: Path) -> Path:
    d = (data_root / "huggingface" / "hub" / HF_REPO_CACHE_DIR / "snapshots" / CHECKPOINT_REVISION /
         CHECKPOINT_SUBFOLDER)
    load_checkpoint_contract(d)  # config/metadata sha 고정 + 구조 검사
    return d


class FrozenGroot:
    """모델을 한 번만 올리고 serve 전처리로 결정 1개씩 latent 를 뽑는다."""

    def __init__(self, serve, *, profile_path: Path, pretrained: Path, device: str):
        import torch

        from composition import check_head_contract

        self.torch = torch
        self.serve = serve
        profile = serve.load_profile(profile_path)
        if profile.model_specific.get("policy_type") != "groot":
            raise ContractError("profile policy_type != groot")
        adapter = serve.make_policy_adapter("groot")
        from lerobot.policies.factory import get_policy_class

        loaded = adapter.load(profile=profile, policy_cls=get_policy_class("groot"),
                              pretrained_path=pretrained, dataset_stats=serve.load_dataset_stats(profile),
                              device=device)
        # serve _load_model_impl 과 같은 globals (parse_payload / _apply_input_remap 가 읽는다)
        serve._profile = profile
        serve._policy_type = "groot"
        serve._policy_adapter = adapter
        serve.policy, serve.preprocessor, serve.postprocessor = (loaded.policy, loaded.preprocessor,
                                                                 loaded.postprocessor)
        from lerobot.configs.types import FeatureType

        visual_keys = [k for k, v in loaded.policy.config.input_features.items()
                       if (v.get("type") if isinstance(v, dict) else v.type) ==
                       (FeatureType.VISUAL if not isinstance(v, dict) else "VISUAL")]
        state_feat = getattr(loaded.policy.config, "robot_state_feature", None)
        serve._state_dim = 0
        if state_feat is None and "observation.state" in loaded.policy.config.input_features:
            sf = loaded.policy.config.input_features["observation.state"]
            serve._state_dim = sf["shape"][0] if isinstance(sf, dict) else sf.shape[0]
        serve._camera_key_map, sd = adapter.build_remap_config(visual_keys, state_feat)
        if sd > 0:
            serve._state_dim = sd
        self.profile = profile
        self.policy = loaded.policy
        self.preprocessor = loaded.preprocessor
        self.head = loaded.policy._groot_model.action_head
        check_head_contract(self.head)

    def latent(self, payload: dict, seed: int) -> tuple[np.ndarray, int]:
        from composition import SPEC, FeatureRecord, FirstCallCapture, validate_feature_record

        torch = self.torch
        serve = self.serve
        if serve._apply_inference_seed({"inference_seed": int(seed)}) != int(seed):
            raise ContractError("inference seed 적용 실패")
        batch = serve._apply_input_remap(serve.parse_payload(payload))
        batch = self.preprocessor(batch)
        cap = FirstCallCapture()
        hook = self.head.action_decoder.register_forward_pre_hook(cap)
        try:
            with torch.inference_mode():
                self.policy.predict_action_chunk(batch)
        finally:
            hook.remove()
        if cap.calls != NUM_INFERENCE_STEPS:
            raise ContractError(f"action_decoder 호출 {cap.calls} != {NUM_INFERENCE_STEPS}")
        rec = FeatureRecord(value=SPEC.pool_decoder_input(cap.first).cpu().numpy(), env_step=0,
                            flow_step=0, signature=SPEC.signature())
        v = validate_feature_record(rec, spec=SPEC, decision_env_step=0, batch_size=1)
        return np.array(v[0], dtype=np.float32), cap.calls


# ---------------------------------------------------------------------------
# dataset 읽기
# ---------------------------------------------------------------------------

def decode_video(path: Path, length: int, timestamps: np.ndarray, fps: float) -> np.ndarray:
    """mp4 를 한 번 decode → (T, H, W, 3) uint8. 프레임 수·시간이 parquet 와 맞아야 한다."""
    import av

    frames, times = [], []
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        for fr in c.decode(stream):
            frames.append(fr.to_ndarray(format="rgb24"))
            times.append(float(fr.time) if fr.time is not None else np.nan)
    if len(frames) != length:
        raise ContractError(f"{path.name}: 프레임 {len(frames)} != parquet 행 {length}")
    dt = np.abs(np.asarray(times) - np.asarray(timestamps, dtype=np.float64))
    if not np.all(dt < 0.5 / fps):
        raise ContractError(f"{path.name}: 프레임 시간 ≠ parquet timestamp (max {np.nanmax(dt):.4f}s)")
    return np.stack(frames)


def png_b64(img: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(buf, format="PNG")  # 무손실
    return base64.b64encode(buf.getvalue()).decode()


class OpenDrawerPackage:
    def __init__(self, data_root: Path):
        self.root = data_root / OPENDRAWER_PACKAGE_REL
        meta = self.root / "meta"
        self.info = json.loads((meta / "info.json").read_text())
        self.modality = json.loads((meta / "modality.json").read_text())
        self.tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
                      for l in (meta / "tasks.jsonl").read_text().splitlines() if l.strip()}
        self.episodes = {json.loads(l)["episode_index"]: json.loads(l)
                         for l in (meta / "episodes.jsonl").read_text().splitlines() if l.strip()}
        self.fps = float(self.info["fps"])

    def rel(self, episode_index: int, video_key: str | None = None) -> str:
        chunk = int(episode_index) // int(self.info["chunks_size"])
        if video_key is None:
            return self.info["data_path"].format(episode_chunk=chunk, episode_index=int(episode_index))
        return self.info["video_path"].format(episode_chunk=chunk, episode_index=int(episode_index),
                                              video_key=video_key)

    def read(self, episode_index: int) -> dict:
        import pyarrow.parquet as pq

        t = pq.read_table(self.root / self.rel(episode_index)).to_pydict()
        return {
            "state": np.asarray(t["observation.state"], dtype=np.float64),
            "raw_actions": np.asarray(t["action"], dtype=np.float64),
            "rewards": np.asarray(t["next.reward"], dtype=np.float32).reshape(-1),
            "done": np.asarray(t["next.done"], dtype=bool).reshape(-1),
            "frame_index": np.asarray(t["frame_index"], dtype=np.int64).reshape(-1),
            "timestamp": np.asarray(t["timestamp"], dtype=np.float64).reshape(-1),
            "episode_index": np.asarray(t["episode_index"], dtype=np.int64).reshape(-1),
            "task_desc": np.asarray(t["annotation.human.task_description"], dtype=np.int64).reshape(-1),
        }


def state_payload_map(modality: dict, profile) -> dict:
    """modality.json state slice (metadata 이름) → serve payload key. GrootPolicyAdapter 표를 역으로 쓴다."""
    from lerobot_adapters.groot import GROOT_STATE_METADATA_KEYS

    meta_to_profile = {v: k for k, v in GROOT_STATE_METADATA_KEYS.items()}
    out = {}
    for meta_key, sl in modality["state"].items():
        if sl.get("original_key") != "observation.state":
            raise ContractError(f"state {meta_key} original_key {sl.get('original_key')}")
        pk = meta_to_profile.get(meta_key)
        if pk is None:
            raise ContractError(f"state key {meta_key} 를 profile key 로 대응할 수 없음")
        out[f"observation.state.{pk}"] = (int(sl["start"]), int(sl["end"]))
    want = {f"observation.state.{k}" for k in profile.observation_requirements.state}
    if set(out) != want:
        raise ContractError(f"state payload key {sorted(out)} != profile {sorted(want)}")
    return out


def video_payload_map(modality: dict) -> dict:
    """modality video key → serve 카메라 alias (src.policies.groot.core.schema 단일 출처)."""
    from src.policies.groot.core.schema import GROOT_ENV_VIDEO_TO_UNIFIED_CAM

    out = {}
    for vk, v in modality["video"].items():
        cam = GROOT_ENV_VIDEO_TO_UNIFIED_CAM.get(f"video.{vk}")
        if cam is None:
            raise ContractError(f"video {vk} 의 카메라 alias 없음")
        out[v["original_key"]] = f"observation.images.{cam}"
    if sorted(out.values()) != ["observation.images.left", "observation.images.right",
                                "observation.images.wrist"]:
        raise ContractError(f"video alias {out}")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def extract_episode(model: FrozenGroot, pkg: OpenDrawerPackage, aux_ep: dict, *, base_seed: int,
                    state_map: dict, video_map: dict, codec, raw_index) -> tuple[dict, dict]:
    idx = int(aux_ep["episode_index"])
    eid = aux_ep["episode_id"]
    t0 = time.time()
    d = pkg.read(idx)
    T = len(d["raw_actions"])
    if T != int(aux_ep["length"]) or T != int(pkg.episodes[idx]["length"]):
        raise ContractError(f"{eid}: 길이 {T} != split/episodes.jsonl")
    if not np.all(d["episode_index"] == idx):
        raise ContractError(f"{eid}: parquet episode_index 불일치")
    tidx = set(d["task_desc"].tolist())
    if len(tidx) != 1:
        raise ContractError(f"{eid}: episode 안에서 task 가 바뀜 {tidx}")
    task = pkg.tasks[tidx.pop()]
    if [task] != list(pkg.episodes[idx]["tasks"]) or [task] != list(aux_ep["tasks"]):
        raise ContractError(f"{eid}: task {task!r} 가 episodes.jsonl/split 과 다름")
    videos, vsrc = {}, {}
    for vkey, payload_key in video_map.items():
        rel = pkg.rel(idx, vkey)
        videos[payload_key] = decode_video(pkg.root / rel, T, d["timestamp"], pkg.fps)
        vsrc[vkey] = {"rel": f"{OPENDRAWER_PACKAGE_REL}/{rel}", "sha256": sha256_path(pkg.root / rel),
                      "frames": T}
    steps = decision_steps(T)
    lat, calls, seeds = [], [], []
    for t in steps:
        payload = {k: png_b64(v[t]) for k, v in videos.items()}
        for k, (lo, hi) in state_map.items():
            payload[k] = d["state"][t, lo:hi].tolist()
        payload["task"] = task
        s = decision_seed(base_seed, eid, int(t))
        v, c = model.latent(payload, s)
        lat.append(v)
        calls.append(c)
        seeds.append(s)
    arrays = build_episode_arrays(episode_id=eid, latents=np.stack(lat), decoder_calls=calls, seeds=seeds,
                                  raw_actions=d["raw_actions"], rewards=d["rewards"], done=d["done"],
                                  frame_index=d["frame_index"], codec=codec, raw_index=raw_index,
                                  base_seed=base_seed)
    from baseline_data import audit_rewards

    prel = pkg.rel(idx)
    meta = {"episode_id": eid, "split": aux_ep["split"], "task": task, "end_reason": "terminal",
            "terminal_convention": "offline_dataset_end", "physical_env_termination": "unknown",
            "reward_audit": audit_rewards(d["rewards"], d["done"]),
            "sources": {"parquet": {"rel": prel, "package_rel": OPENDRAWER_PACKAGE_REL,
                                    "sha256": aux_ep["parquet_sha256"]}, "videos": vsrc},
            "walltime_s": round(time.time() - t0, 3)}
    if sha256_path(pkg.root / prel) != aux_ep["parquet_sha256"]:
        raise ContractError(f"{eid}: parquet sha 불일치")
    return arrays, meta


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="OpenDrawer expert episode → frozen GR00T decoder-입력 latent 추출")
    ap.add_argument("--out", required=True, type=Path, help="추출 출력 폴더 (새 폴더 또는 같은 계약의 resume)")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="dataset root (기본 scripts.path_setup.DATA_ROOT; 컨테이너는 /cache/datasets)")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / AUX_SPLIT_V2_REL)
    ap.add_argument("--image-id", required=True, help="실행 docker image id (provenance; 계약에 포함)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--episodes", default=None, help="episode index 부분집합 (예: 0-1 또는 0,1). 기본 102 전부")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.data_root is None:
        from scripts.path_setup import DATA_ROOT

        a.data_root = DATA_ROOT
    if sha256_path(a.split) != AUX_SPLIT_V2_SHA256:
        raise ContractError(f"split sha != v2 {AUX_SPLIT_V2_SHA256}")
    aux = validate_aux_split(a.split, data_root=a.data_root, verify_files=True)
    split_doc = json.loads(a.split.read_text())
    eps = {int(e["episode_index"]): e for e in split_doc["episodes"]}
    todo = episode_index_list(a.episodes, eps)
    contract = extraction_contract(base_seed=a.base_seed, aux_split_sha256=aux["aux_split_sha256"],
                                   code_sha256=code_hashes(), profile_rel=PROFILE_REL,
                                   profile_sha256=sha256_path(REPO_ROOT / PROFILE_REL), image_id=a.image_id)
    a.out.mkdir(parents=True, exist_ok=True)
    ensure_same_or_create(a.out / "extraction_contract.json", contract, what="extraction_contract.json")
    pending = [i for i in todo if verify_committed_episode(
        a.out, i, contract, expect={"episode_id": eps[i]["episode_id"], "split": eps[i]["split"]}) is None]
    print(json.dumps({"contract_sha256": contract_sha256(contract), "selected": len(todo),
                      "pending": len(pending)}), flush=True)
    progress = a.out / "progress.jsonl"
    if pending:
        ck = checkpoint_dir(Path(a.data_root))
        cc = load_checkpoint_contract(ck)
        pkg = OpenDrawerPackage(Path(a.data_root))
        raw_index = raw_to_model_index(pkg.modality["action"])
        serve = import_serve_module()
        t0 = time.time()
        model = FrozenGroot(serve, profile_path=REPO_ROOT / PROFILE_REL, pretrained=ck, device=a.device)
        load_s = time.time() - t0
        from composition import SPEC

        if SPEC.signature() != EXPECTED_FEATURE_SIGNATURE:
            raise ContractError("feature signature")
        state_map = state_payload_map(pkg.modality, model.profile)
        video_map = video_payload_map(pkg.modality)
        import torch

        runtime = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": a.device,
                   "gpu": torch.cuda.get_device_name(a.device) if torch.cuda.is_available() else None,
                   "python": platform.python_version(), "image_id": a.image_id,
                   "model_load_s": round(load_s, 2)}
        for n, i in enumerate(pending):
            arrays, meta = extract_episode(model, pkg, eps[i], base_seed=a.base_seed, state_map=state_map,
                                           video_map=video_map, codec=cc.codec, raw_index=raw_index)
            meta["runtime"] = runtime
            rec = commit_episode(a.out, episode_index=i, arrays=arrays, meta=meta, contract=contract)
            line = {"episode_index": i, "episode_id": rec["episode_id"], "split": rec["split"],
                    "n_decisions": rec["n_decisions"], "npz_sha256": rec["npz"]["sha256"],
                    "walltime_s": meta["walltime_s"], "done": n + 1, "pending": len(pending),
                    "time": time.time()}
            with open(progress, "a") as f:
                f.write(json.dumps(line) + "\n")
                f.flush()
            print(json.dumps(line), flush=True)
    committed = {i: verify_committed_episode(a.out, i, contract) for i in sorted(eps)}
    if all(v is not None for v in committed.values()):
        summary = {"contract_sha256": contract_sha256(contract), "episodes": len(committed),
                   "counts": aux["counts"],
                   "npz_sha256": {str(i): r["npz"]["sha256"] for i, r in committed.items()},
                   "decisions": int(sum(r["n_decisions"] for r in committed.values())),
                   "npz_bytes": int(sum(r["npz"]["bytes"] for r in committed.values()))}
        ensure_same_or_create(a.out / "extraction_complete.json", summary, what="extraction_complete.json")
        print(json.dumps({"complete": True, **{k: summary[k] for k in ("episodes", "decisions", "npz_bytes")}}))
    else:
        print(json.dumps({"complete": False,
                          "committed": sum(v is not None for v in committed.values()), "total": len(eps)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
