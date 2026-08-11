# S5 — serve + steering 배선 스테이지 카드 (2026-08-10)

기계 판독분: [`S5_files.tsv`](S5_files.tsv) · 판정 UI: `python3 scripts/review/ledger_ui.py`

## 규모 — 15파일 9.9k줄. 그중 3.7k 가 즉시 회수 가능

### 발견 1 — serve 사본 2개가 완전 중복 (3,738줄)

`exp42_serve.py`·`patchceil_serve.py` 는 라운드 중 serve 를 고정하려던 사본인데,
**지금은 lerobot.py 와 bit 동일**(diff 0) — 고정 효력이 없다. archive 하고 참조하는
.sh(S6 라운드 러너 6개)는 S6 판정 때 처리하면 3.7k 즉시 회수.

### 발견 2 — native_steered_serve 는 이미 없음

COAST 재현 정리 때(eb1bddf) 삭제 완료. 남은 serve 는 HTTP=lerobot.py, ZMQ=feature_server.py 둘.

## 판정 축 — 질문 3개

1. **타모델 serve 4개** (groot/openvla_oft/upvla/xvla, 1,325줄): steering 연구는 안 쓰지만
   벤치마크 인프라(통일 API 의 존재 이유). 유지 판단은 "다른 모델 벤치를 앞으로 돌리나".
   groot.py 는 lerobot.py 와 역할이 겹칠 수 있어 확인 필요.
2. **hook 4종의 정리 폭**: steering(693)·safe(527) 는 본류 keep. attn(277) 은 archive
   방침 확정 — lerobot.py 안의 cam-attn 분기 절제와 한 세트. patching(443) 은 patchceil
   은 죽었지만 perturb(donor 이식) 축이 keep 이라 남길 근거 있음.
3. **lerobot.py 1,869줄 본체**: 수집 세션 사용 중이라 **라운드 완주까지 동결**.
   지금 할 수 있는 건 지도 만들기(캡처/개입/gated/patch/health 구획)와 절제 계획뿐.
   S1 허브의 '수정' 이월과 같은 처지 — 완주 후 cam-attn 절제 + 인자 정리 일괄.

## 예약된 부수 작업 (판정 무관, 이미 합의·발견된 것)

- `serve_provenance()` GPU 제거 (docs/38 §4-1) — machine 열 오염 방지. **동결 예외
  후보**: serving.py 는 수집이 import 하지만 반환값 형태만 바뀜 — 수집 세션과 협의 필요.
- `test_serve_lerobot.py` 16 실패 진단 (기존 baseline — 수정 자체는 동결 대상일 수 있음)
- `test_serve_groot.py` — groot 컨테이너 httpx 부재로 collection 불가 (env 문제)
- `vla_client.py` 의 src 이동 (RENAME_PLAN 후보 — src→scripts 역방향 의존 해소)

## 참고

- serve 는 지도 2(steering)의 "개입" 노드이자 지도 1의 오른쪽 레인 전체다.
- capture 와 steering 이 같은 hook 지점 → 개입 후 캡처값(자기서술) 성질은 검증 자산.
