# exp4-2 induced-failure 파이프라인 runbook

계획: `docs/steering/24b_exp4-2_perturb_conceptor_plan.md` (+ 공유 `24_exp4_shared_plan.md`).
섭동 메뉴(07-22 개정): **C1 카메라 pose / G1 그리퍼 초기위치 / P1 displace / P2 force**.
모든 실행은 worktree(`.claude/worktrees/exp4-2-induced-failures`) 코드 + main-tree `outputs/`.
serve = `scripts/serve/exp42_serve.py`(lerobot.py 심링크), GPU 는 실행 시 빈 것 선점(GPU당 serve 2).

## 0. Smoke (하드 게이트 — 본수집 전 전부 PASS)

```bash
cd scripts/safe/groot_n15/robocasa/steer/induced/smoke
bash s4_fit_truncation.sh                      # CPU (passB 기성 자산)
GPU=<idx> PORT=8470 bash s1_perturb.sh         # Track P sham/실효/결정론
GPU=<idx> PORT=8471 bash s2_dit_patch.sh       # DiT self-donor W=3
GPU=<idx> PORT=8472 bash s3_vl_patch.sh        # VL self/trim donor
GPU=<idx> PORT=8473 bash s5_npz_serve.sh       # exp4-1 NPZ 계약 (s4 산출 필요)
```

## 1. P0 파일럿 (ppcc_bread, scene 100084)

```bash
cd scripts/safe/groot_n15/robocasa/collect
P0=outputs/eval/robocasa/groot_n15/exp42_induced/p0   # main-tree 기준
# Phase A: baseline 18ep (--no-features 사이드카 = 앵커 원천)
PHASE=baseline GPUS_L="7 7" PORTS_L="8480 8481" bash collect_perturb_grid.sh
# Phase B: grid 생성 (성공 12ep 앵커 → spec json + grid.tsv)
python3 build_perturb_grid.py --baseline-dir <P0>/baseline/raw_rollouts/PickPlaceCounterToCabinet/ppcc_bread \
  --out-dir <P0>
# Phase C: grid 실행 (실패율 측정)
PHASE=grid GPUS_L="..." PORTS_L="..." bash collect_perturb_grid.sh
# Track I (B2/B4): noise 생성 → plan → 러너
docker exec lerobot python .../induced/make_noise_npz.py --src <donors/epX_L15.npz> --scale {0.5,1,2} --out <P0>/noise/...
python3 .../induced/build_patch_plan.py --baseline-dir ... --donor-glob '.../patchceil/*/donors/ep*_L15.npz' \
  --noise-dir <P0>/noise --out <P0>/arm_plan.tsv
PATHWAY=dit GPUS_L="..." PORTS_L="..." PLAN=<P0>/arm_plan.tsv bash .../induced/run_induced_patch.sh
# Phase D: 집계 → 40–70% 게이트로 config 채택
python3 .../induced/aggregate_p0.py --p0-dir <P0>
# 채택 config 캡처 재실행 (fit 용, 결정론 재수집)
PHASE=capture CONFIGS="c1_s100 g1_x010 ..." GPUS_L="..." PORTS_L="..." bash collect_perturb_grid.sh
```

## 2. fit + 게이트

```bash
# manifest (창끝+2 절단, P1 은 +4; 3분할 계약 누적)
docker exec lerobot python .../induced/build_induced_fit_manifest.py \
  --perturb-capture-dir <P0>/capture --patch-rollout-dir <P0>/trackI/<variant> \
  --role calibration --out-dir <P0>/manifests/cal
# conceptor fit (primary: perturbed-fail vs perturbed-succ)
docker exec lerobot python .../steer/fit_phase_conceptor_n15.py --manifest .../fit_manifest.tsv \
  --record-start-manifest .../record_start.tsv --layers 8,12,VL --groups global,... --quota-floor 0.01
# 비퇴화 진단 (= H1 1차 검정; exp3 기준 0.006~0.007 대비)
docker exec lerobot python .../steer/diag_conceptor_nondegen.py --npz .../dit_L8/conceptors.npz \
  --capture-layer 8 --fit-manifest ... --held-manifest ... --out diag.json
# bridge 게이트 (유도축↔자연축; 자연측 = patchceil passB 7fail/9succ)
docker exec lerobot python .../induced/bridge_axis_check.py --induced-manifest ... \
  --natural-manifest ... --layers 8,12 --out bridge.json
```

- **참조선** (bridge_sanity/nat_vs_nat.json, 07-22): 자연 s400020 vs 자연 s300033
  (cross-scene) cos 0.43~0.59, cross-AUROC(ep) 0.81~1.0 — 유도축 정렬의 자연 상한.
- exp4-1 전달: fit 산출 `global/dit_L{L}` → `<base_B>/steer/dit_L{L}` 복사 (s5 스크립트 참조).

## 함정 (실측)

- worktree submodule 은 심링크로 해결 (`lerobot -> ../../../lerobot` 등, patchceil 선례).
- gymnasium Wrapper 는 인스턴스 hasattr 로 실env 를 찾으면 안 됨 (type-레벨 hasattr 필수).
- `docker exec` heredoc 은 `-i` 필수.
- 6p proximity 라벨 어휘에 "transport" 없음 — carry 구간은 "place".
- `--expect-chunk-len` 은 serve chunk 길이(16), 실행 단위(NAS=5)와 별개.
- **VL self-donor 는 bitwise 불가**: VL 캡처는 fp16 저장인데 원값이 fp32 (DiT 는 bf16→fp16
  무손실이라 S2 는 bitwise). S3 실측 record0 편차 9.77e-04 → 판정 기준 = 첫 편차 ≤ 5e-3
  (`smoke_judge csv-first-diff`). B1 해석에는 무영향 (의도 주입이 거시적).
- **bash 특수변수 `GROUPS` 대입 금지**: bash 내장(사용자 gid 배열)이라 대입이 조용히 무시되고
  `$GROUPS`→gid(예: 1004)로 확장 — fit `--groups` 가 통째로 오염된 실사고(07-23). 러너
  변수명은 FIT_GROUPS 처럼 비예약어로. (UID/EUID/HOSTNAME/RANDOM/SECONDS 등도 동일 주의)
- **kitchen lang 은 ep_meta 소비가 아니라 task 재생성** — 타 instruction 주입은 collector
  `--instruction-override` 로 (B1 실사고 07-23).
