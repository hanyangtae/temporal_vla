"""키 맞교환 전후 연산자 산출물 대조 — 값은 같고 라벨만 바뀌었는지 확인.

oven/washer 의 left↔right 키 교환은 **이름만 바꾸는 조작**이라(pkl·좌표 불변), 교환 후
새 키로 재fit 한 연산자는 교환 전 상대 키의 연산자와 **값이 같아야** 한다.

⚠ 이 대조가 검증하는 것과 못 하는 것
  - 검증한다: ① 재fit 이 엉뚱한 shard·scene·지터를 읽지 않았는지 ② rename 이 shard
    **내용**을 건드리지 않았는지 (내용이 바뀌면 클래스 평균이 달라져 값이 갈린다).
  - 검증하지 못한다: **rebase 자체의 순수성**. rename 방식에서는 교환 후 shard 가 교환
    전 shard 그 파일이므로, 아카이브에서 무슨 일이 있었든 이 대조는 통과한다. 순수성은
    post-rebase 아카이브 지문(meta.json pkl_sha256) 전수 대조로만 판정된다.

파일 해시가 아니라 **배열 값**으로 비교한다 — npz 는 zip 컨테이너라 같은 배열이어도
타임스탬프·압축 차이로 파일 해시가 갈린다.

usage:
  python compare_prerebase.py --old outputs/.../_prerebase --new outputs/.../instr_setm_v6_gt \
      --map "OvenRack_out-left=OvenRack_out-right,OvenRack_out-right=OvenRack_out-left"
"""
import argparse
import glob
import json
import os

import numpy as np

# 키 이름이 들어가는 metadata 필드 — 값 비교에서 제외한다(바뀌는 게 정상)
KEY_FIELDS = ("variant",)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="교환 전 산출물 루트(_prerebase)")
    ap.add_argument("--new", required=True, help="교환 후 재fit 산출물 루트")
    ap.add_argument("--map", required=True,
                    help='"구키=신키,구키=신키" — 교환이므로 양방향 다 적어야 한다')
    ap.add_argument("--rtol", type=float, default=0.0,
                    help="0(기본)이면 완전 일치 요구. float 재현성 문제 시에만 완화")
    return ap.parse_args()


def load(p):
    z = np.load(p)
    md = json.load(open(os.path.join(os.path.dirname(p), "metadata.json")))
    return z["alpha0_v_steer"], z["alpha0_s"], md


def main():
    args = parse_args()
    kmap = dict(kv.split("=", 1) for kv in args.map.split(",") if kv.strip())
    olds = sorted(glob.glob(f"{args.old}/*/*/*/*/dit_L12/conceptors.npz"))
    if not olds:
        raise SystemExit(f"{args.old}: 대조할 산출물이 없다")

    ok = miss = bad_val = bad_md = 0
    for op in olds:
        rel = op.split(args.old.rstrip("/") + "/", 1)[1]
        key = rel.split("/")[0]
        if key not in kmap:
            continue                      # 교환 대상 아닌 키는 건너뜀
        np_ = os.path.join(args.new, kmap[key], *rel.split("/")[1:])
        if not os.path.exists(np_):
            print(f"[결손] {rel} → {kmap[key]} 쪽에 없음")
            miss += 1
            continue
        v0, s0, m0 = load(op)
        v1, s1, m1 = load(np_)
        same = (np.array_equal(v0, v1) and np.array_equal(s0, s1)) if args.rtol == 0 else (
            np.allclose(v0, v1, rtol=args.rtol) and np.allclose(s0, s1, rtol=args.rtol))
        if not same:
            cos = float(v0 @ v1 / (np.linalg.norm(v0) * np.linalg.norm(v1) + 1e-12))
            print(f"[값 불일치] {rel}: cos={cos:.6f} Δs={float(s1) - float(s0):+.4f}")
            bad_val += 1
            continue
        diff = [k for k in set(m0) | set(m1)
                if k not in KEY_FIELDS and m0.get(k) != m1.get(k)]
        if diff:
            print(f"[meta 불일치] {rel}: {diff}")
            bad_md += 1
            continue
        ok += 1

    print(f"\n대조 {len(olds)}건 중 교환 대상 {ok + miss + bad_val + bad_md}건 | "
          f"일치 {ok} · 결손 {miss} · 값 불일치 {bad_val} · meta 불일치 {bad_md}")
    if miss or bad_val or bad_md:
        print("★ 불일치 있음 — 재fit 이 다른 데이터를 읽었거나 rename 이 내용을 바꿨다. 보고 필요")
        return 1
    print("★ 값 동일·라벨만 변경 확인 (단, rebase 순수성 판정은 아님 — docstring 참조)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
