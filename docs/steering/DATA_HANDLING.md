# 데이터 취급 규약 — 수집 · 아카이브 · 삭제

**대용량 rollout/activation을 만들고 원격(승준)으로 보내고 로컬에서 지우는 모든 작업에 적용된다.**
특정 라운드용 통지가 아니라 상시 규약이다. 사고 사례 §4는 근거로 남겨둔다.

## 1. 삭제 전 보존 검증 — 이름 세기 금지

**파일 개수를 세서 "아카이브됐다"고 판정하면 안 된다.** 껍데기 심링크와 축소본이 같은 이름으로
같은 개수만큼 존재할 수 있다.

```bash
# ① 실물만 센다 (심링크 제외)
find <dir> -type f | wc -l

# ② 용량 대조 (원본 vs 아카이브)
du -sh <원본> <아카이브>

# ③ 평균 파일 크기 상식 체크  ← 가장 잘 걸리는 게이트
#    fit activation pkl 은 개당 수십 MB. 평균 ~1MB 가 나오면 껍데기를 세고 있다 → 즉시 중단.
find <dir> -type f -name '*.pkl' -printf '%s\n' | awk '{s+=$1;n++} END{print s/n/1e6, "MB avg,", n, "files"}'

# ④ 무결성 (npz 는 생성·전송 직후 반드시)
python3 -c "import numpy as np,sys; d=np.load(sys.argv[1]); [d[k] for k in d.files]" <파일>.npz
```

## 2. 심링크

- **fit 서브셋을 절대경로 심링크로 만들지 말 것.** 아카이브 rsync가 `-L` 없이 돌면 링크 껍데기만
  저장되고 겉보기엔 완료로 보인다.
- 반드시 **상대경로**(`ln -srf`). 절대경로는 컨테이너에서 깨진다.
- 아카이브 rsync는 `-L`(`--copy-links`) 또는 실물 복사.
- 컨테이너 호환 검증은 **컨테이너를 만들거나 재시작하지 말고** 호스트에서 문자열 검사로 한다
  (수집기·VNC 세션 끊김 사고 방지):

```bash
find <dir> -type l -exec readlink {} \; | grep -vE "^/temporal_vla|^[^/]"
# 출력이 있으면 그 링크는 컨테이너에서 깨진다 (절대경로가 /temporal_vla 밖)
```

이미 떠 있는 컨테이너가 있으면 read-only 확인만: `docker exec robocasa test -e <경로>`.

## 2.5 재수집 정책 — 덮어쓰기 금지, 내용 대조

**단일 출처는 [`../04_data_storage_convention.md`](../04_data_storage_convention.md) §2(쓰기 검사)·§8(금지 사항)이다.**
여기서는 현행 stem 레이아웃에 적용된 형태만 적는다 — sig 레이아웃 이관 후에는 그 문서로 일원화된다.

수집 산출물의 stem 은 `task{id}--ep{idx}--succ{0|1}` 이고, **GPU·모델·캡처층 같은 조건은 stem 에
안 들어가고 pkl 내부 필드**(`serve_gpu`, `feature_kind`, `capture_layers`, …)로만 남는다.
따라서 파일명만으로는 "같은 조건의 재실행"과 "조건이 바뀐 재수집"을 구분할 수 없다 →
**내용(sha256)으로 판정한다.**

`write_safe_triplet` 의 동작:

| 상황 | 처리 |
|---|---|
| 기존 pkl 없음 | 그대로 쓴다 |
| 있고 **내용 동일** | 중복. 다시 쓰지 않고 **skip** (`[collect] … 이미 동일 내용으로 존재 — skip`) |
| 있고 **내용 상이** | **에러 중단.** 절대 덮어쓰지 않는다 |
| succ 가 뒤집힌 다른 stem 이 있음 | 덮어쓰기가 아니므로 지우지도 막지도 않고 **경고**만 |

> 구 배선은 이 자리에서 `succ*.*` 를 전부 unlink 했다. 발동할 자리가 없거나(중간 사망은
> pkl 자체가 없어 안 걸린다) 발동하면 안 되는 경우(succ 반전 = 조건 변경 신호)뿐이라
> 2026-08-04 에 제거하고 규약 §2 쓰기 검사로 교체했다.

**조건을 바꿔 재수집할 때는 출력 트리를 분리한다** — `phase_event_strict`, `phase_event_6p`,
`phase_event_exp3` 처럼 `RUN_ID` 를 바꾸거나 `cell_id` 에 접미사를 단다(`ppcc_potato_s2`).
그러면 충돌 자체가 안 나고, 에러로 막히는 일도 없다.

## 3. 아카이브 배치 규칙

- 승준 아카이브는 **HDD로만** (NVMe 금지). workspace에는 심링크.
- **종류를 골라 include하지 말 것** — pkl·csv·mp4 전부. (mp4 누락 사고 이력)
- pkl은 zstd 압축률 ~4%라 압축 이득이 없다.
- 원격 경로 역할 구분: 코드 repo = `~/workspace/temporal_vla`(git checkout),
  데이터 아카이브 = `~/datasets/temporal_vla_outputs/`. 섞지 말 것.

## 4. 사고 사례 — exp2 fit activation 유실 (2026-07-16 확인)

exp2 seed-변형 5 cell의 fit 원료 pkl(각 60판)이 세 호스트 어디에도 남지 않았다.
단일 실수가 아니라 **3단 연쇄**였고, 각 단계가 §1~§2 규약 중 하나를 어겼다.

| 시점 | 무슨 일 | 어긴 규약 |
|---|---|---|
| 07-06 | fit 서브셋이 **로컬 절대경로 심링크**로 생성 → 승준 rsync가 `-L` 없이 돌아 링크 껍데기만 저장 (겉보기 완료) | §2 |
| 07-10 | exp3 킥오프가 구 `phase_event_6p` 트리를 제거 ("아카이브됨" 전제). 승준 workspace 사본은 그 주 디스크 정리에서 소멸 | §1 |
| 07-14 | eval purge의 "fit 보존" 검증이 **이름 개수만** 셈 → 껍데기+신트리를 세고 통과. probe 기록에 이미 이상신호가 있었다: "pkl 1,605개, **평균 1.0MB**" (fit pkl은 개당 수십MB) | §1-③ |

- **무사했던 것**: conceptor NPZ 전체, fitlog, exp2 manifest, 판정 sidecar, mp4,
  구 3-scene cell fit 원료 180판.
- **복구 가능성**: 수집이 `(scenario_seed, inference_seed)` 결정적이라 재수집으로 동일 재생 가능
  (5 cell × 60판 ≈ GPU 2-worker ~5h). 사용자 결정으로 보류 — 필요 시에만.
- 교훈이 §1-③에 있다. **평균 파일 크기가 상식과 다르면 그 자리에서 멈춘다.**

## 5. 관련

- eval activation 전 호스트 삭제(2026-07-14, ~172GB): 판정 sidecar·fit·conceptor는 보존.
  exp3부터 eval 캡처 OFF.
- 원격 오케스트레이션: `scripts/utils/remote_compute.sh` (단일 출처), `.claude/agents/remote-compute.md`.
- 결과·데이터 위치 대조표: [`RESULTS.md`](RESULTS.md) §5.
