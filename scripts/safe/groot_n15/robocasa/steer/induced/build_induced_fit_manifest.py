"""exp4-2 fit manifest 빌더 — 유도실패 캡처 pkl → fit manifest + record-start + 3분할 계약.

절단 규칙 (24b §2.1·§4.1, anti-circularity 시간분리):
- Track P: start = perturb_record_window[1] + 2. **P1 은 +2 추가** (=창끝+4, teleport 시각
  불연속). C1(persistent) 은 start = 0 (지속형 — 동일 섭동 하 outcome 대조로 해석).
  G1(pre-record, window [-1,-1]) 은 start = 1.
- Track I: status_ep{N}.json 의 fired_records 를 기대창(start_record..+patch_len)과 대조 —
  불일치 rollout 은 **제외 목록**에 기록 (fit 미포함). start = max(fired)+2.

출력 (--out-dir):
  fit_manifest.tsv   (pkl<TAB>label<TAB>scene — label=episode_success, perturbed-succ=1)
  record_start.tsv   (pkl<TAB>start)
  details.json       (행별 mode/config/창/제외 사유)
  split_contract.json (--role 별 episode key 목록 + 교집합 0 assert — 반복 호출로 누적)

pkl 로드에 torch 필요 — lerobot 컨테이너 실행.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path


def _ep_key(d: dict, config: str) -> str:
    return f"{config}|ep{d.get('episode_idx')}|inf{d.get('inference_seed')}|s{d.get('scenario_seed')}"


def _p_start(d: dict) -> tuple[int | None, str]:
    spec = d.get("perturb_spec") or {}
    mode = spec.get("mode")
    win = d.get("perturb_record_window")
    if mode == "C1_camera":
        return 0, "C1 persistent — 전 구간 (동일 섭동 하 outcome 대조)"
    if mode == "G1_gripper_init":
        return 1, "G1 pre-record — 창끝(-1)+2"
    if mode in ("P1_displace", "P2_force"):
        if not win or len(win) != 2:
            return None, f"{mode}: perturb_record_window 없음/이상 ({win})"
        if mode == "P1_displace":
            return int(win[1]) + 4, "P1 창끝+4 (시각 불연속 +2 추가)"
        return int(win[1]) + 2, "P2 창끝+2"
    return None, f"unknown mode {mode}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perturb-capture-dir", action="append", default=[],
                    help="Track P 캡처 루트 (<dir>/<config>/raw_rollouts/**/*.pkl) — 반복 지정")
    ap.add_argument("--patch-rollout-dir", action="append", default=[],
                    help="Track I 변형 루트 (<dir>/raw_rollouts/**/*.pkl + status_ep*.json)")
    ap.add_argument("--role", required=True, choices=["calibration", "fit", "locked"],
                    help="이번 호출 행들의 3분할 역할 (교집합 0 강제)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import torch  # noqa: F401 — pkl unpickle (컨테이너 전용)

    rows = []       # (pkl, label, scene, start, note, key)
    excluded = []   # (pkl, reason)

    for root in args.perturb_capture_dir:
        rootp = Path(root)
        for p in sorted(rootp.glob("*/raw_rollouts/*/*/task*--ep*--succ*.pkl")):
            config = p.relative_to(rootp).parts[0]
            d = pickle.load(open(p, "rb"))
            spec = d.get("perturb_spec") or {}
            if not spec:
                excluded.append((str(p), "perturb_spec 없음"))
                continue
            if spec.get("sham"):
                excluded.append((str(p), "sham (fit 제외)"))
                continue
            start, note = _p_start(d)
            if start is None:
                excluded.append((str(p), note))
                continue
            n_rec = len(d.get("hidden_states") or [])
            if start >= n_rec:
                excluded.append((str(p), f"start {start} >= records {n_rec}"))
                continue
            rows.append((str(p), int(d.get("episode_success", 0)),
                         str(d.get("scenario_seed", "")), start,
                         f"{spec.get('mode')}/{config}: {note}", _ep_key(d, config)))

    for root in args.patch_rollout_dir:
        rootp = Path(root)
        variant = rootp.name
        for p in sorted(rootp.glob("raw_rollouts/*/*/task*--ep*--succ*.pkl")):
            m = re.search(r"--ep(\d+)--", p.name)
            ep = int(m.group(1)) if m else -1
            st_path = rootp / f"status_ep{ep}.json"
            if not st_path.exists():
                excluded.append((str(p), f"status_ep{ep}.json 없음"))
                continue
            st = json.loads(st_path.read_text())
            patch = st.get("patch") or {}
            t0 = patch.get("start_record")
            plen = patch.get("patch_len")
            fired_all = []
            ok = True
            for layer, h in (st.get("hooks") or {}).items():
                fired = h.get("fired_records") or []
                fired_all.extend(fired)
                if t0 is not None and plen not in (None, -1):
                    want = list(range(int(t0), int(t0) + int(plen)))
                    if fired != want:
                        excluded.append((str(p), f"L{layer} fired {fired} != 기대 {want} "
                                         "(anti-circularity ③ — 폐기)"))
                        ok = False
                        break
            if not ok:
                continue
            if not fired_all:
                excluded.append((str(p), "fired 0 (미발화)"))
                continue
            d = pickle.load(open(p, "rb"))
            start = max(fired_all) + 2
            n_rec = len(d.get("hidden_states") or [])
            if start >= n_rec:
                excluded.append((str(p), f"start {start} >= records {n_rec}"))
                continue
            rows.append((str(p), int(d.get("episode_success", 0)),
                         str(d.get("scenario_seed", "")), start,
                         f"trackI/{variant}: 창끝+2", _ep_key(d, variant)))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "fit_manifest.tsv").open("w") as f:
        for pkl_path, label, scene, _s, _n, _k in rows:
            f.write(f"{pkl_path}\t{label}\t{scene}\n")
    with (out / "record_start.tsv").open("w") as f:
        for pkl_path, _l, _sc, start, _n, _k in rows:
            f.write(f"{pkl_path}\t{start}\n")
    (out / "details.json").write_text(json.dumps(
        {"role": args.role,
         "rows": [{"pkl": r[0], "label": r[1], "scene": r[2], "start": r[3],
                   "note": r[4], "key": r[5]} for r in rows],
         "excluded": [{"pkl": e[0], "reason": e[1]} for e in excluded]},
        indent=1, ensure_ascii=False))

    # 3분할 계약 누적 + 교집합 0 assert (Gate1)
    contract_path = out.parent / "split_contract.json"
    contract = json.loads(contract_path.read_text()) if contract_path.exists() else {}
    keys = sorted({r[5] for r in rows})
    contract[args.role] = sorted(set(contract.get(args.role, [])) | set(keys))
    roles = list(contract.keys())
    for i, a in enumerate(roles):
        for b in roles[i + 1:]:
            inter = set(contract[a]) & set(contract[b])
            if inter:
                print(f"ABORT: 3분할 교집합 위반 {a}∩{b}: {sorted(inter)[:5]}", file=sys.stderr)
                return 2
    contract_path.write_text(json.dumps(contract, indent=1))

    n_f = sum(1 for r in rows if r[1] == 0)
    print(f"[{args.role}] rows={len(rows)} (fail={n_f} succ={len(rows) - n_f}) "
          f"excluded={len(excluded)} -> {out}")
    for e in excluded[:8]:
        print(f"  [excluded] {Path(e[0]).name}: {e[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
