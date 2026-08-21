# 45 — VLA Hotfix 시나리오 명세 (논문 프레이밍 정본)

2026-08-18, "시나리오 구체화" 세션. 목적: activation steering 기반 **VLA hotfix** 주장의
스코프·요구 일반화·검증 프로토콜을 조작적으로 고정한다. 근거 실측은 42(파이프)·43(detector
절제·LOTO)·44(condg)와 이 세션의 환경 코드 검증.

## 1. 시나리오 정의

배포된 VLA가 특정 상황에서 실패를 반복한다. 사용자는 다음 모델 업데이트 전까지, 자기
로그(sparse 실패 rollout + 약간의 성공)만으로 백본 무학습 패치를 만들어 그 상황의 SR을
올린다. 패치 = 실패 detector(발화 게이트) + phase-매칭 activation 연산자, 둘 다 로그에서
자동 fit·자동 등록 판정.

**상황(situation)의 조작적 정의** = (task, 배포지 환경, 물건 집합) — fit에 노출 허용, 명시.
**재발(recurrence)** = 같은 상황에서 아래 3축 변주로 다시 발생한 실패:

| 재발 변주 축 | RoboCasa 지원 | 근거 |
|---|---|---|
| 정책 확률성 (denoise seed) | ✓ (현 격자 커버) | drawer-L 실측: 동일 reset에서 seed만으로 succ/fail 갈리는 scene 7/10 |
| 물건 배치 지터 (주방 고정) | ✓ (ep_meta 고정 + env_seed 변주) | kitchen.py: ep_meta는 reset_region(규칙)만 저장, 실제 배치는 reset마다 placement_initializer.sample() 재추첨 (L818) |
| 로봇 초기 자세 지터 | ✓ | robosuite initialization_noise="default" (reset마다 관절 지터) |

**스코프 밖(보너스 실험)**: 물건 종류/집합 변화, 새 layout/scene, 새 task, 초기조건형
실패(OvenRack류 — detector 후보 제외, 43; 처방은 재샘플/섭동 계열로 분리).

## 2. 요구 일반화 (컴포넌트별)

- **Detector (더 넓은 쪽)**: 표적 상황 내 새-episode TPR + **배포 트래픽 전체에서의 FPR
  상한**. 회귀 위험 ≈ FPR × 개입 파손률 (개입은 위약조차 성공 판을 깨뜨릴 수 있음 — 42·44
  실측). 구성은 **per-task** (task 혼합·zero-shot 전이는 LOTO에서 chance — 43), instruction
  family 공유는 방향 증거만. 학습은 phase-gt 길이 절제(검출률 불변·조기성 preW 0.17→1.00,
  drawer-L) — full 학습 신호는 컨택-후에만 분리(α sweep: α0.2=공짜 5-record 조기가 한계,
  α0.3=FPR 붕괴).
- **연산자 (좁은 쪽)**: 실패 manifold 안 보간만 — 재발 변주 3축 커버, scene 밖 외삽 불요.
  scene 암기는 결함이 아니라 자원(배포지 고정). 층층 게이트(발화→phase→margin)가 연산자의
  특이성 부담을 흡수.
- **cross-task/scene 일반화 불요**가 프레임의 세일즈 포인트: 요구하지 않는 일반화가 실제로
  불가능함(43 LOTO)을 실측으로 방어.

## 3. 검증 프로토콜

- **분할**: scene(배포지)은 fit-노출 허용·명시. held-out은 **episode 축**(env_seed ×
  inference_seed). 1차 지표 = in-fit 변주의 재발 구제율, 부지표 = held-out 변주.
- **arm 표준**: base / online(처치) / online_pl(라벨순열 위약) / **timer-gated**(t≥W 발화 —
  TPR 1.0·FPR 0.12의 강기준선, detector 가치=조기성뿐이므로 필수 대조, 43) / 연산자 ablation.
- **판정**: 셀-paired 구제/파손 + 분모 명시(발화 수/실개입 수/뒤집힘 수) + 비회귀 조항
  (성공 셀 파손률 ≤ 위약).
- **자동 등록 게이트**: cell별 held-out margin AUROC > 길이단독(5-seed CV 중앙값+과반),
  클래스당 ≥8 ep 경성 하한·<15 소표본 딱지 — 미달 cell은 identity. "데이터 부족하면 패치
  안 함"이 시나리오의 안전장치.
- **최소 로그 규모**: task당 ~50판 (25판 규모에서 등록 0 cell 실측 — 44).

## 4. 수집 설계 (n15_grid_v2 제안)

현 격자(10 scene × 5 noise)는 cross-scene fit을 강제해 시나리오와 어긋남. 재배치:

```
task 2–3종 (drawer-L 포함, 실행표류형)
 × scene 3–5개 (중간 SR 0.2–0.8 선별 = 독립 case study 반복 축)
 × env_seed 20–30 (배치·로봇 초기 지터)  ← 신규 축
 × inference_seed 2–4 (정책 확률성)
= scene당 60–100판, fit/eval은 episode 축 분할
```

수집 전 10분 검증 1건: ep_meta 고정 + env_seed 변주 시 "배치만" 바뀌고 물건 종류 불변인지
reset 덤프로 확인.

**주의 (08-18, exp6 세션 실측)**: 기존 n15_grid_v2 plan(3134e339de4c)은 v1의 단순 확장
(scene +5 × denoise seed n10-14)이라 **이 축이 아니다** — 셀당 env_seed 1개 고정이라 배치
잔차이 변동이 없음. 위 구조의 수집은 **별도 plan(v3)** 필요. env_seed 축은 기존 replay
인프라(index env_seed 재생)와 그대로 호환 — 러너 변경 불요.

## 4b. 벤치마크 구성 — 2단 (2026-08-18 검증 완료)

RoboCasa 제약 프로토콜(ep_meta 고정)이 "벤치 원래 의미(scene 일반화) 훼손" 논란을 부를 수
있다는 우려에 대한 구성:

1. **주 실험 = RoboCasa 제약 프로토콜** — 정당화: ① ep_meta 고정은 RoboCasa 공식
   재현성 API, ② leaderboard SR 비교를 하지 않고 hotfix 프로토콜 신규 정의, ③ 대형
   generalist 벤치의 자연 실패를 고정 배포지에서 반복 측정하는 것 자체가 시나리오의
   현실 대응물.
2. **부 실험 = SimplerEnv 1–2 task** — native로 우리 축(고정 장면·소폭 변주 반복·중간
   SR이 설계 목표)인 주류 벤치. 검증 실측(08-18): 채택 모델 = RT-1·Octo·OpenVLA·CogACT·
   SpatialVLA·π0·GR00T (커뮤니티 leaderboard); **GR00T 공식 SimplerEnv ckpt 존재**
   (파인튜닝 불요) — N1.6 Bridge 56.6%/Fractal 52.0%, N1.7(=Cosmos-Reason2-2B 백본)
   Bridge 62.3%/Fractal 72.5%, task별 편차 커서(eggplant 2%~drawer 100%) 중간 대역
   task의 자연 실패 풍부. RL²(경쟁 니치)와 동일 무대 정면 비교 가능 + 재현 인프라 보유
   (docs/steering/38).
- CALVIN(실패 원인이 subtask 체인과 얽힘)·LIBERO(SR 과높아 자연 실패 희소)는 기각.

## 5. baseline·경쟁 구도

- **재샘플 baseline(감지 후 denoise 재추첨)이 필수 정면 대조**: 시나리오의 재발 축에 정책
  확률성이 포함되는 순간, "방향 정보가 재샘플보다 나은가"가 연산자 가치의 시금석. 지면
  "가장 싼 hotfix는 재샘플"도 성립하는 결론 (OvenRack 상시위약 +0.175가 이 방향의 선행 신호).
  - **재샘플의 대조 정의 (08-18 exp6 합의)**: 방향이 없어 라벨-순열 위약이 정의 불가 →
    기준 ③을 "위약 우위"가 아니라 **"타이밍 정보 우위"**로 치환: 감지-후 재샘플 vs
    **상시 재샘플(record 0부터)** 대조. 감지-후≈상시면 detector 사슬 불요(= 가벼운 재시도
    정책으로 환원), 상시가 성공판을 깨면(OvenRack 상시위약 파손 5/23 전례) "감지-후"의
    존재 이유가 기준 ①에서 나옴. 여력 시 다른 seed-offset 재추첨 1회로 추첨 운 분산 추정.
- 무학습 경쟁: 같은 sparse 로그 LoRA 미세조정 — 차별점은 절대 SR이 아니라
  무학습·즉시배포·가역성·cell 단위 안전 판정.
- 솔직한 현재 상태: 감지·게이팅·비회귀는 스펙 충족, **구제는 미달성**(42·44 — raw 연산자
  전 계열 + condg 처치≈위약). 주장의 무게중심은 "요구 일반화를 최소화하는 프레임 + 자동
  검증 게이트 체계"에 둔다.
