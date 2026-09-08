# k8 길이 조정 재fit 및 baseline 없는 5-arm 평가

2026-09-08 최신 사용자 지시: GT phase와 baseline replay를 이번 라운드에서 제외한다.
선정 target j의 성공/실패 n0..9 전판을 plain .8, jfair .9, reseed,
reseed→plain .8, reseed→jfair .9로 평가한다. detector 준비 시 reseed 먼저 시작한다.

## 대상과 가능한 범위

정본 `configs/experiments/v6_ck8_dwell_20260908/{targets,episodes}.tsv`.
원래 singleton 지정표는 없었으므로 이번 사용자 기준으로 새로 선정했다.
조건: target 성공≥1·실패≥1, target 외4j pool에 성공·실패 모두 존재,
instruction의 scene0/1/2 모두 충족. 기존 intervention 성공률은 선정에 쓰지 않았다.
후보 중 target 성공판 수 우선, 동률은 j 번호 오름차순. 새 성공≥3 기준 없음.

| instruction | s0 | s1 | s2 | arm당 | 5arm |
|---|---|---|---|---:|---:|
| OpenDrawer/left | j1 (9S/1F) | j3 (4S/6F) | j4 (8S/2F) | 30 | 150 |
| PPCC/candle | j0 (8S/2F) | j1 (9S/1F) | j2 (7S/3F) | 30 | 150 |

다른 10 instruction은 일부 scene에서 위 조건을 만족하는 j가 없다. 부분 scene만
섞어 instruction별 판수를 다르게 만들지 않는다. 상세 후보/부족 사유는 main repo
`outputs/analysis/v6_preflight_audit_20260908/target_candidates.tsv`.

## 길이와 모델 계약

- feature: L12, denoise index3, all49 token mean. compact shard의 segment 축은
  원본 `segment_names`에서 all만 선택한다. state/future/action 평균의 평균은 금지.
- `prepare_ck8_scene.py`: zip streaming, 실제 serve ClusterPhaseAssigner와 동일한 배정,
  원 GT label 별도 보존, AE bundle SHA256, row/episode 좌표 및 finite 검사.
- detector: actual LOKO fit source를 safe-length-ablation에서 통합했다. target 외4j의
  40판 전체를 사용, target 성공/실패 모두 제외. cluster별 성공 dwell ceil(mean+std)
  cap을 train pool에서 계산하고 각 episode cluster의 앞쪽 record를 유지한다.
  성공 pool에 없는 cluster는 삭제하지 않고 유지하며 짧은 정상 episode도 유지한다.
  시간순 recurrent 시퀀스와 masked per-record BCE를 사용한다. inference는 full record다.
  따라서 이 통제만으로 모든 길이 단서가 제거됐다고 주장하지 않는다.
- operator: target 외4j 전체+target 실패 허용, target 성공 제외. 같은 cap 후 cluster별
  episode 평균을 동일 가중해 긴 rollout의 과도한 기여를 줄인다. jfair는 기존 지터 내
  대비 평균, plain은 성공-실패 평균. 기존 min20 record/mixed2 등록 규칙은 유지한다.
- 각 실패 rollout을 하나씩 뺀 signed direction cosine, setpoint 변화, full target 실패
  anchor에서 공통 shift 차이를 기록한다. abs를 쓰거나 LOO cutoff로 셀을 제거하지 않는다.
  제거 후 정의 불가능한 연산자는 확인불가 사유를 남긴다.
- missing cluster → pure reseed. 전 cluster 연산자가 없으면 fallback_only.json으로
  명시하며 실제 hook은 0개다. reseed→setM도 2차 pass는 한 번만 수행한다.
- scalar setM v2 common delta 유지, 적용 layer L12만, denoise global 유지.

## 실행

코드는 `eval_whole_pipe_1` git으로 원격 별도 worktree에 전달한다. raw/compact shard는
승준 노드에 남긴다. CPU thread8, build는 순차 실행한다.

`build_v6_ck8_dwell.py --store ~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v6
--targets configs/experiments/v6_ck8_dwell_20260908/targets.tsv
--episodes configs/experiments/v6_ck8_dwell_20260908/episodes.tsv
--out outputs/analysis/grid_phase/v6_ck8_dwell_build`

작은 공개 산출물은 build output의 `released/`만 remote_compute.sh로 회수한다.
`dispatch_v6_ck8_dwell.py --main-root /home/dongkyu/pkt_ws/temporal_vla
--analysis-repo /home/kimseungjun/workspace/temporal_vla_ck8_dwell`을 setsid/nohup으로
실행한다. instruction의 3 detector가 준비되면 reseed30판을 먼저 실행하고, 연산자
준비·reseed 완료 후 나머지120판을 시작한다. GPU가 점유돼 있으면 대기한다.
kanu는 GPU5/6/7, srv50은 빈 GPU2를 로컬 lease로 관리한다. 오류 자동 재시도 없음.

새 output `og_v6_ck8_dwell_20260908/{reseed,operators}`. base arm을 만들지 않는다.
reference는 원 수집의 success label이며 새 baseline replay 대비 효과라고 부르지 않는다.
GT detector/GT operator를 사용한 이전145개 v2 후보는 이번 모델 계약과 불일치하므로
새 라운드에 재사용할 수 없다. 이전 잘못된 평가 output만 정리하고 학습원본/타 실험은 보존한다.

## 검증 및 판정 범위

Docker 관련 테스트113개와 detector 합성 self-test PASS. fallback-only 실제 등록 경로와
학습 노드 prepare/fit 및 평가 실제 실행 상태도 별도로 확인한다.
A/B 독립 audit을 저렴한 agent에게 배정했고 root가 source/segment 축 오독을 수정했다.
기존 NPZ274개 실제 finite/shape검사는 통과했으나 이를 실패 LOO 검증으로 대신하지 않는다.
새 LOO 통계는 실제 compact shard에서 재fit할 때 생성한다.

이 문서는 실행 계약이며 연구 효과 보고서가 아니다. 결과 해석 전 confound audit:
length=부분통제, instruction/scene=층별보고, operator target실패fit=허용된 in-sample,
detector=targetj holdout, phase=ck8, causal effect=baseline재실행없음 제한을 명시한다.
