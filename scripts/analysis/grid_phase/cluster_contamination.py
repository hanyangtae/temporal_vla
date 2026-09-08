#!/usr/bin/env python3
"""cluster 가 phase 를 담았나, 초기조건(j)을 담았나 — 오염 진단표.

**왜**: 초기 record 의 활성화는 j(초기조건)를 거의 그대로 담는다(early_record_probe 실측:
첫 record 만으로 j 5-class 정확도 .78~.98, chance .2). AE 는 무라벨 재구성이라 가장 큰
분산 축을 먼저 담으므로, **초반 cluster 는 "phase 초기" 가 아니라 "어느 j 인가" 그룹일 수
있다**. 그러면 per-cluster 연산자는 phase-matched 가 아니라 초기조건-matched 가 된다
(의도와 다른 조건화, 에러는 안 난다).

**측정**: 번들 라벨(cluster) × {j, GT phase} 의 상호정보를 **창별로** 낸다.
  - `MI(cluster; j)`      — 오염 지표. 높을수록 그 구간 cluster 는 j 그룹.
  - `MI(cluster; phase)`  — 의도한 신호.
  - 둘 다 정규화값(H 로 나눈 것)을 병기 — cluster 수·라벨 수가 달라 절대 bit 비교는 위험.
초기 창에서 MI(c;j) ≫ MI(c;phase) 면 그 cluster 층은 연산자 조건화에 부적합하다.

cluster 라벨 출처는 `ae_cluster.py --dump-labels` 산출 `labels_<slug>_k<K>.npz`
(열: ep_id·rec_idx·scene·noise·succ·phase_code·cluster). **j 열은 없으므로** 같은 slug 의
shard 에서 `jitter` 를 행 순서로 가져온다(둘은 1:1 같은 순서 계약).

사용:
    python3 cluster_contamination.py --labels <ae_dir>/labels_<slug>_k8.npz \
        --shard <segA>/<slug>.npz --windows 0:10,10:30,30:9999
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def entropy(counts):
    p = counts[counts > 0].astype(np.float64)
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def mutual_info(a, b):
    """이산 라벨 두 벡터의 MI(bit)와 정규화값 (MI / min(H_a, H_b))."""
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    tab = np.zeros((len(ua), len(ub)), dtype=np.int64)
    np.add.at(tab, (ia, ib), 1)
    n = tab.sum()
    if n == 0:
        return float("nan"), float("nan")
    pxy = tab / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    mi = float((pxy[nz] * np.log2(pxy[nz] / (px @ py)[nz])).sum())
    hmin = min(entropy(tab.sum(axis=1)), entropy(tab.sum(axis=0)))
    return mi, (mi / hmin if hmin > 0 else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, type=Path, action="append")
    ap.add_argument("--shard", required=True, type=Path, action="append",
                    help="같은 순서로 대응하는 shard (jitter 열 출처)")
    ap.add_argument("--windows", default="0:10,10:30,30:9999")
    ap.add_argument("--max-j-share", type=float, default=0.40,
                    help="cluster 의 최빈 j 점유율이 이 값 이상이면 연산자 대상에서 제외 "
                         "(기본 0.40 = 5-j 균등 0.20 의 2배)")
    ap.add_argument("--min-n", type=int, default=50,
                    help="이보다 적은 record 의 cluster 는 점유율이 흔들려 판정 보류")
    ap.add_argument("--mi-j-norm-warn", type=float, default=0.10,
                    help="창별 MI(c;j) 정규화값이 이 값 이상이면 그 창을 경고 표시")
    ap.add_argument("--out-json", type=Path, default=None)
    a = ap.parse_args()

    if len(a.labels) != len(a.shard):
        raise SystemExit("--labels 와 --shard 개수가 다르다")
    wins = [tuple(int(x) for x in w.split(":")) for w in a.windows.split(",")]

    report = {}
    for lab_p, sh_p in zip(a.labels, a.shard):
        with np.load(lab_p, allow_pickle=False) as z:
            cluster = z["cluster"]
            rec = z["rec_idx"]
            phase = z["phase_code"]
            cb = json.loads(str(z["phase_codebook"])) if "phase_codebook" in z else {}
        with np.load(sh_p, allow_pickle=False) as z:
            jit = z["jitter"]
            n_sh = int(z["X"].shape[0])
        if len(cluster) != n_sh:
            raise SystemExit(
                f"{lab_p.name}({len(cluster)}) vs {sh_p.name}({n_sh}) 행 수 불일치 — "
                "같은 shard 로 만든 라벨인지 확인할 것 (행 순서 1:1 계약)")

        rows = []
        name = lab_p.stem
        print(f"\n=== {name} ===  (cluster {len(np.unique(cluster))}종, "
              f"j {len(np.unique(jit))}종, phase {len(cb) or len(np.unique(phase))}종)")
        print(f"{'창':>10} {'n':>7} {'MI(c;j)':>9} {'norm':>6} "
              f"{'MI(c;phase)':>12} {'norm':>6}  판정")
        for lo, hi in wins:
            m = (rec >= lo) & (rec < hi)
            if m.sum() < 50:
                print(f"{f'{lo}:{hi}':>10} {int(m.sum()):>7}  (표본 부족 — 건너뜀)")
                continue
            mij, nij = mutual_info(cluster[m], jit[m])
            mip, nip = mutual_info(cluster[m], phase[m])
            verdict = ("j 우세" if nij > nip * 1.2 else
                       "phase 우세" if nip > nij * 1.2 else "혼재")
            if nij >= a.mi_j_norm_warn:
                verdict += " ⚠오염"
            print(f"{f'{lo}:{hi}':>10} {int(m.sum()):>7} {mij:>9.3f} {nij:>6.3f} "
                  f"{mip:>12.3f} {nip:>6.3f}  {verdict}")
            rows.append({"window": [lo, hi], "n_records": int(m.sum()),
                         "mi_cluster_j": mij, "mi_cluster_j_norm": nij,
                         "mi_cluster_phase": mip, "mi_cluster_phase_norm": nip,
                         "verdict": verdict})

        # cluster 별 j 쏠림 — 어느 cluster 를 연산자 대상에서 뺄지의 **기계 판독 근거**.
        # 규칙(연산자 세션 합의 2026-09-04): max-j 점유율 ≥ --max-j-share 면 제외.
        # 균등 점유율은 1/(j 종수) 이므로 기본 0.40 = 5-j 균등(0.20)의 2배.
        # ⚠ 작은 cluster 는 점유율이 흔들린다(n=250·5j 면 SE≈0.025 라 0.52 는 유의하지만,
        # n<50 이면 우연으로도 0.4 를 넘는다) → n 을 함께 싣고 --min-n 미만은 판정 보류.
        n_j = len(np.unique(jit))
        uniform = 1.0 / n_j if n_j else float("nan")
        print(f"  cluster 별 최빈 j 점유율(전 구간, 균등={uniform:.2f}, "
              f"제외기준 ≥{a.max_j_share:.2f}):")
        per_cluster = {}
        for c in np.unique(cluster):
            mc = cluster == c
            vals, cnt = np.unique(jit[mc], return_counts=True)
            share = float(cnt.max() / cnt.sum())
            n_c = int(mc.sum())
            if n_c < a.min_n:
                verdict, flag = "보류(표본부족)", " ·표본부족"
            elif share >= a.max_j_share:
                verdict, flag = "제외", " ⚠j쏠림→제외"
            else:
                verdict, flag = "사용", ""
            per_cluster[int(c)] = {"n": n_c, "top_j": int(vals[cnt.argmax()]),
                                   "top_j_share": share, "uniform_share": uniform,
                                   "verdict": verdict,
                                   "exclude": verdict == "제외"}
            print(f"    c{int(c)}: n={n_c:>6} 최빈 j{int(vals[cnt.argmax()])} "
                  f"{share:.2f}{flag}")
        n_excl = sum(1 for v in per_cluster.values() if v["exclude"])
        print(f"  → 제외 {n_excl} / {len(per_cluster)} cluster")
        report[name] = {"windows": rows, "per_cluster": per_cluster,
                        "rule": {"max_j_share": a.max_j_share, "min_n": a.min_n,
                                 "mi_j_norm_warn": a.mi_j_norm_warn}}

    if a.out_json:
        a.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                              encoding="utf-8")
        print(f"\n[contam] → {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
