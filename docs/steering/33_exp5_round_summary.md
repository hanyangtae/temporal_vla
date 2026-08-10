# 33. exp5 라운드 정리 — 시도 · 결과 · 다음 카드 (2026-07-27~28)

> **사용자 방향 메모 (2026-07-30)**: 잔차화 규명 · scene 암기 여부 · SAE가 정말 필요한지는
> **여러 task를 보면서 더 깊게 파볼 것**. 지금 결론은 drawer/mixer 등 소수 task 기준이라
> task 일반성이 확인되지 않았다. 특히 §2 "잔차화는 방향을 개선하지 못함"과 G2의 scene
> 암기 해석은 [`32_g2_scene_residual_results.md`](32_g2_scene_residual_results.md) §0-3 정정
> (rank 상한 때문에 산술적으로 강제되는 결과)과 함께 읽을 것.

exp5 세션 종합. 상세 근거: 31(G1)·32(G2·차원 지도)·`g3_precalc_*.json`·`g3_directions/`.
브랜치 feat/scene-sae. 협업: codex 리뷰 루프 3라운드(지적 20건 처리), exp5-3(데이터·기준선),
대시보드(재검증·판정 개정), opus 에이전트 9개(구현·분석).

## 1. 무엇을 시도했나 (시간순)

1. **G1 — SAE가 scene을 읽는가** (drawer_left fit30): 동료 top-k SAE 코어 lift(src/sae, 테스트
   37종) + per-token 빌더 + scene probe. 15 run(5 layer × k{16,32,64}, m=6144).
2. **probe 결함 수정** (대시보드 지적): 표준화·수렴·Pipeline CV·순열 null 정합 — G1 재판정.
3. **scene-matched 파이프라인** (exp5-3 수집 활용): drawer_right·mixer·beer 각 160판
   (scene 20 × denoise seed 8) → scan-dir 빌더(라벨 대조·scene-held-out split·fingerprint).
4. **G2 — scene을 지우면 outcome이 남는가**: 선형 arm(between/logreg 부분공간 r-sweep) +
   SAE arm(selective 제거) + raw, exp5-3 within-scene LOSO 프로토콜 문자 이식, 창 [0,38).
5. **SAE 반론 2회**: layout-기반 → scenario_seed-기반 selectivity 재산정.
6. **G1 재판정 @ 진짜 해상도**: scenario_seed 20클래스 probe (drawer_right, L8/10/12).
7. **G3 사전계산** (계산 A 방향 cos / B 차원 / C 세그먼트): drawer 5층 × mixer 5층 × beer 5층
   격자 + layer간 r̂ cos.
8. **G3 재료 준비**: r̂ fit 스크립트(setM NPZ 계약·fold 분리·fit_identities), 연산자 사전 등록
   결정표, armO 설계.

## 2. 결과 (확정된 것)

### 읽기(read) 계열 — 전부 결론남
- **G1 PASS ×2**: SAE feature는 scene을 읽는다 — drawer_left layout(회복 0.84~0.92, z 5.8~6.5),
  drawer_right **scenario_seed 20클래스**(L12 0.898, 회복 0.93, null_z 38.6). 라벨 해상도 한계 해소.
- **G2 PASS (drawer 단독, 개정 기준)**: seen-scene 성분을 지워도(식별 0.81→0.16) 실패축
  read는 유지+상승(정답지 자: 0.878→**0.921, 부호 13/13**, 위약 400회 p<0.0025).
  → **outcome 신호는 scene 성분이 아니다.** t=0 0.729(관측 동일·denoise 추첨만)도 독립 증거.
- **발견 A — scene은 암기다**: 새 scene의 식별 방향은 기존 scene들이 span하는 공간 밖
  (r19 전부 제거해도 unseen 0.91~0.99). 공유 "scene-coding" 축 없음 → **온라인 잔차화 불가**,
  scene 제거의 새 scene 일반화는 구조적으로 막혀 있음.
- **발견 B — SAE는 removal에서 선형에 완패**: selective 제거(layout·scenario_seed 기반 2회)
  모두 scene을 못 지움(0.77→0.70), 선형 between r12+는 0.16까지 지움. 읽기는 되고 지우기는
  안 되는 비대칭. → **SAE 존폐 결론 = (b) 선형 전환** (반론 소진, exp5 기획 전제의 반증).
- **3-task 유형론** (창 내 실패축): drawer=**강한 공통축**(read 0.80~0.85, 1축이 새 scene
  방향의 ~50% + 얇은 잔여 2~3축) / mixer=**약한 분산**(0.65~0.73, 공통축 4~17%) /
  beer=**무신호**(z<2, 결판 phase가 공통 창 밖) — 조작이 길수록 창-풀링 분석이 무너지는 기울기.
- **layer 구조**: 깊은 층 L8~12는 같은 축의 전파(cos 0.73~0.94) → L12 하나로 충분.
  **L0은 독립축**(깊은 층과 cos 0.05~0.25) — 초기조건형 신호 후보로 기록.
- **세그먼트**: task별 상이 — drawer=action 토큰(0.845·cos 0.45), mixer=state 토큰(0.72).
- **방법론 산출물**: 사다리식 차원검사는 산술적으로 강제 붕괴(합성 증명) → 비퇴화 포착률
  지표로 대체. 통계량 이원(풀링 0.847 vs scene별 평균 0.878) 규명 — 앵커 대조 시 자 명시 필수.

### 부정·유보로 결론난 것
- 잔차화는 **방향을 개선하지 못함**(scene간 cos 0.34→0.35) — read 상승은 nuisance 제거 효과.
- 잔차화 r̂ ≈ 전역 난이도 축(cos 0.993) — r16 arm은 exp4-1 null 재시도와 구분 곤란.
- 성분 제거 연산자는 산수상 탈락(실패판 1.8gap 이동 = 파국 조건 초과).
- mixer·beer는 G2/G3 대상에서 제외(각각 표현 근거 부족·창 구조 문제).

## 3. 앞으로 시도할 수 있는 것 (우선순위순)

1. **G3 (write, drawer 단독)** — 재료 완비, 입력 2개 대기:
   - 연산자 = **armO** h′=h−γ·max(0,(h·r̂)−s)·r̂ (단방향 문턱 — 보존율 붕괴 차단),
     **k=1(지배축) vs k=3(잔여축 포함)** 두 arm. serve에 max(0,·) 게이트 소규모 배선 필요.
   - 대기: exp5-3 β(γ) sweep 결과, 개입 시점 결정 — oracle t0 수동 주석(exp4-1 인프라 재사용,
     실패 105판, srv50 머신 바인딩) vs 무주석 대안(고정 지연·phase-오라클 latch) 먼저.
   - 판정: (scene,seed) 셀 paired **구제율/보존율 분리** (전체 SR 금지). 기대치는 낮게 —
     읽기 3무 1패 + t=0 증거는 "결과의 흔적" 가능성을 남김. 양방향 실험(null이어도 서사 종결 근거).
2. **task×phase별 실패축 지도** (~1.5h, CPU만): phase-matched 규약 이식 — G3 개입 시점의
   근거 + **beer 재심**(exp5-3 phase-matched insert-settle 0.810 → 창 밖 결판을 phase 앵커로
   잡으면 살아날 가능성). 선택 효과(실패는 후반 phase 미도달) 표본수 병기 필수.
3. **L0 독립축 규명**: 깊은 층과 직교하는 초기층 실패 신호 — 초기조건형(지각·기하 기원)
   가설과 연결. 읽기 분석은 저비용, write 대상으론 미검증.
4. **beer 이벤트-앵커 라운드** (exp5-3 §6 설계): grasp_timeline 재구성·실패 모드 층화·
   물체 pose 로깅 추가 수집 — 별도 라운드 규모.
5. **mixer 공식 G2**: 기준선 통계량 합의(scene별 평균 자에선 raw p<0.0001 성립) 후
   L2/L12 재판정 — 예비 결과는 L12 잔존(p<0.0025)·L2 감쇠(p=0.01).
6. **(보류 카드)** Cosmos value best-of-N(25a §5), SAE는 해석 용도 기록 보관.

## 4. 운영 기록 (다음 세션 함정 방지)

- 승준 공유 checkout은 exp5-4(`exp/exp5-4-noise-select`)가 점유 — 승준 실행은 **단일 파일
  전송 + repo 밖 출력**(~/sae_build/) 무접촉 방식으로.
- rudxo_home(218.152.144.220:11111)은 승준 경유 hop만 가능. mixer 원본 수집분 위치.
- 대용량 npz는 생성·전송 후 **np.load 무결성 검사 필수** (X_L10 원본 손상 사건 — 빌더 산출
  직후 자가검증 추가는 TODO).
- GPU는 발사 직전 소유자 확인(가드가 run_g1_right.sh에 내장) — 이번 라운드 중 준형 점유가
  수시로 변동. pkill 자기매칭 함정(브래킷 트릭) 재발 주의.
- codex 0.145.0: scope 리뷰 × instructions 불가 — ask 레인에 focus+diff 인라인 (wrapper 가드 갱신됨).
