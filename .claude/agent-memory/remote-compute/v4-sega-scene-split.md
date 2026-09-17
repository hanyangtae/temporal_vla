---
name: v4-sega-scene-split
description: v4 segA는 task당 scene 5개뿐이라 detector 기본 분할 6/2/2가 train=1 scene으로 퇴화 — 3/1/1 seed0을 쓸 것; PPCC_apple·CoffeeSetupMug는 구조적 제외
metadata:
  type: project
---

v4 segA(`grid_phase_v4/segA`, `segA_v4_ck8`)는 **task당 scene 5개(s0~s4) × 25ep = 125ep**
뿐이다. `failure_detector_sim.py` 기본값 `--train-scenes 6 --calib-scenes 2 --test-scenes 2`는
`split_scenes()`의 "모자라면 test/calib부터 채운다" 규칙 때문에 **train=1 scene(25ep)**으로
퇴화 → train 단일클래스 skip이 속출한다. **v4에는 `--train-scenes 3 --calib-scenes 1
--test-scenes 1` (train 75 / calib 25 / test 25), seed 0** 을 쓴다 — 실측으로 10 slug 중
8개가 seed 0 한 번에 cp_bands 확보(2026-08-28).

구조적 제외 2 task (seed 재추첨으로 해결 불가):
- **PPCC_apple**: 125/125 성공, 실패 판 0 → 판별기 학습 자체가 불가.
- **CoffeeSetupMug**: 성공 9/125 인데 8개가 scene s2 한 곳에 몰려 있음. band(train 성공)와
  calib 둘 다 성공 ≥3(`MIN_BAND_EPS`)이 필요한데 s2를 어느 쪽에 줘도 반대쪽이 굶는다 →
  scene-disjoint 분할에서 증명적 불가. (`--band-mu calib`로 우회는 가능하나 다른 task와
  규약이 달라져 미채택.)

**Why:** 08-28 v4 정렬 체인 step5가 7/10 task cp_bands 실패로 죽었고, 원인이 데이터 부족이
아니라 분할 파라미터 퇴화였다. 같은 실수를 v4 계열에서 반복하기 쉽다.
**How to apply:** v4(또는 scene 5개짜리) shard로 `failure_detector_sim.py`를 돌릴 땐 분할을
명시할 것. 위 2 task는 cp_bands 없는 게 정상이니 실패로 보고하지 말고 제외 목록으로 다룰 것.
[[remote-node-hardware]]
