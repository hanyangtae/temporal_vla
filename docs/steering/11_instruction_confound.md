# Phase A 결과 — instruction confound 판정

작성: 2026-06-05. Phase A (VL/DiT LDA 사분면 분석, 원격 compute) 결과.

## 한 줄 결론

헤드라인 VL-우위 task(SlideDishwasherRack)의 Phase 3 VL AUROC=0.93 은 **failure 전조 신호가
아니라, VL goal 토큰이 instruction(slide in/out)을 인코딩하고 그 instruction 이 성공/실패와
거의 1:1 로 상관된 아티팩트**일 가능성이 크다. 길이 confound 에 이어 **instruction confound**
가 추가로 확인됨.

## 근거 — instruction 별 성공/실패 쏠림

| task / instruction | succ | fail | 성공률 |
|---|---|---|---|
| SlideDishwasherRack — slide **in** | 50 | 4 | **93%** |
| SlideDishwasherRack — slide **out** | 6 | 40 | **13%** |
| OpenCabinet — **doors**(이중) | 37 | 13 | 74% |
| OpenCabinet — **door**(단일) | 16 | 34 | 32% |
| OpenDrawer — right | 28 | 34 | 45% |
| OpenDrawer — left | 17 | 21 | 45% |

- **SlideDishwasherRack**: success ≈ slide_in, failure ≈ slide_out 로 거의 분리됨. mixed 데이터의
  succ/fail 분류기는 사실상 in/out 분류기. VL(goal) pathway 가 instruction 을 인코딩하므로
  VL AUROC 가 높게 나오는 것은 당연 — genuine pre-failure 신호의 증거가 아니다.
- **OpenCabinet**: 단일/이중 문 난이도 차이로 instruction 이 성공률과 상관. 같은 confound 약하게 존재.
- **OpenDrawer**: left/right 둘 다 ~45% → instruction-failure 쏠림 없음. 이 task 의 분리력(DiT 우위)은
  instruction 아티팩트가 아닐 여지가 크다.

## instruction-pure 로 가면 분리 자체가 무너짐

instruction 을 고정하면 소수 클래스 샘플이 부족해 within-instruction succ-vs-fail 분리를 세울 수 없다.
- slide_in: fail 4 개뿐 → 실패 분포 추정 불가.
- slide_out: succ 6 개뿐 → 성공 reference 추정 불가.
즉 mixed 의 0.93 은 instruction identity 에 올라탄 값이고, instruction 안에서는 신호가 사라진다.

## 함의

1. **VL=goal-type failure 감지** 주장은 SlideDishwasherRack/OpenCabinet 에서 instruction-difficulty
   confound 와 분리 불가 → 헤드라인 숫자 재해석 필요.
2. type-matched VL steering 의 근거(이들 task 에서 VL 우위)가 약해짐. mixed VL conceptor 는
   success(≈slide_in) 부분공간으로 steer = goal 을 in 쪽으로 미는 것에 가까워 의미가 불분명.
3. 상대적으로 신뢰할 신호: **instruction-balanced task**(OpenDrawer 등)의 pathway 분리력.
4. 다음 단계 판단은 사용자 결정 영역 (아래).

## 산출물 경로

`outputs/eval/robocasa/groot_n16/target_atomic_moderate10_pathway_pertoken_100ep/analysis/lda_quadrant/`
- `<task>/` (mixed, 10개), `<task>__instr_<tag>/` (binary 3 × instruction, 6개)
- 각 dir: lda_scatter.png, lda_trajectory.png, lda_per_quad.png, video_list.html, lda_summary.json
