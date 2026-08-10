# S1 직접 확인 카드 (도구 불신 전제, 한 장)

**원칙: 내(agent) 도구의 "통과"는 증거가 아니다.** 아래는 ① 도구를 거치지 않는 독립 경로와
② 도구 자체를 검증하는 음성 대조다. 각 명령은 5줄 이하 — 직접 읽고 직접 친다.

```bash
P=/home/dongkyu/miniconda3/envs/lerobot_safe/bin/python
CELL=outputs/eval/robocasa/groot_n15/phase_event_6p/raw_rollouts/PickPlaceCounterToStove/ppcs_apple_s100395
```

## 1. 눈으로 — 영상 vs 라벨 (도구 완전 무관)

mp4는 수집 때 env 렌더러가 쓴 것이고 라벨은 파일명에 있다. 서로 다른 코드가 쓴 두 산출물.

```bash
ls $CELL/*succ0*.mp4 | head -3   # 실패 라벨 영상 3개
ls $CELL/*succ1*.mp4 | head -3   # 성공 라벨 영상 3개
```

각 3개씩 열어 본다. **succ1은 apple이 pan에 들어가고, succ0은 못 들어가야 한다.**
(apple 채점 임계 이력이 있으니 경계 애매하면 pan 중심거리 0.10 기준 — RESULTS.md §6 참조.)

## 2. 손으로 — pkl 필드 3개만 (5줄, 전부 읽고 치기)

```bash
$P -c "
import pickle
d = pickle.load(open('$CELL/task1--ep28--succ1.pkl','rb'))
print(d['cell_id'], d['scenario_seed'], d['episode_idx'], d['episode_success'])
print(d['canonical_instruction'])
print(len(d['hidden_states']), 'records, first shape', tuple(d['hidden_states'][0].shape))"
```

읽는 법: `cell_id` 끝 숫자 == `scenario_seed` / `episode_idx`·`episode_success` == 파일명의
`ep28`·`succ1` / instruction이 이 task 문장이 맞는지 / shape `(7, 4, 1536)` = (layer 7개, denoise 4, D).

다른 파일 2~3개로 반복. **파일은 아무거나 본인이 고를 것** (내가 고른 파일만 정상일 가능성 차단).

## 3. 길이 confound — 원라이너로 재확인 (내 집계를 믿지 말고)

```bash
for f in $CELL/*.pkl; do
  n=$($P -c "import pickle;print(len(pickle.load(open('$f','rb'))['hidden_states']))")
  echo "$n  $(basename $f)"
done | sort -n
```

내 주장: succ0은 전부 144, succ1은 41~78. **이 출력에서 직접 세어보면 된다.**

## 4. 도구 자체 검증 — 음성 대조 (검증자를 검증)

일부러 망가뜨린 pkl을 도구가 잡는지. 이미 1회 실행했고(2026-07-30) 결과는 아래 —
**직접 재현하려면**:

```bash
$P scripts/review/selfcheck_inspect.py   # 오염 6종 심고 검출 여부 출력
```

| 심은 오염 | 검출 |
|---|---|
| 성공 라벨 뒤집기 (파일명↔pkl) | ✓ |
| hidden_states 5개 절단 (축 불일치) | ✓ |
| record 하나에 NaN 주입 | ✓ |
| scenario_seed ↔ cell_id 불일치 | ✓ |
| instruction ↔ task_description 불일치 | ✓ |
| record 통째 0 (캡처 실패) | ✓ |

6/6 검출 + 무오염 대조는 통과. **이게 도구의 "통과"를 믿을 수 있는 근거**다.

## 5. 도구가 못 보는 것 (한계 — 알고 쓸 것)

- **의미 검증 불가**: activation 값이 "그 layer의 진짜 residual인지"는 pkl만으로 모른다.
  → serve 쪽 hook 검증(S5)에서 별도로 본다 (self-donor bitwise 이식 같은 방법).
- **영상↔라벨 일치는 §1 사람 눈**이 유일한 경로다.
- 도구는 **한 파일 안의 자기모순**만 잡는다. 수집 자체가 일관되게 틀렸다면(같은 버그로
  라벨과 데이터가 같이 틀림) 못 잡는다.

---

상세 해설(파일 구조 지도·dtype 감사·bias 실측)은 [`S1_verify_collect.md`](S1_verify_collect.md) —
참고용이고, **판정에 필요한 건 이 카드만으로 충분**하다.
