# Fork 소유 신규 scene 수집

사용자 09-09 직접 지시: fork가 수집 실행을 소유하고 메인은 기존 eval만 관리. CoffeeSetupMug/PPCC apple은 추가 수집 제외. 잘못 발사된 coffee plan aa4d86162226 collector/shipper 및 ports9500/9501은 중단, 산출물 보존.

첫 수집 plan 4fa6496cd684: canonical08f1c9df8207 전체1800좌표를 그대로 보존한 superset. 신규250판만실행. target10주방목록불변,각scene5j×n10. 원래PPCC3scene뒤sid3부터추가.

| 키 | sid | layout/style | env seed |
|---|---:|---|---:|
| PPCC/bread | 3 | 10/10 | 100114 |
| PPCC/bread | 4 | 2/2 | 101893 |
| PPCC/jug | 3 | 8/8 | 100386 |
| PPCC/marshmallow | 3 | 1/1 | 100498 |
| PPCC/marshmallow | 4 | 6/6 | 100165 |

기존target10 seed scan에서문장일치후대표spawn을선정했다. 정책성공결과를선정에사용하지않는다. 성공·실패분포는수집후판정한다. 새scene수집/향후replay머신은kanu.

`build_topup_ppcc_20260909.py`로재생성하며기존3scene instructions/scenes/jitters및env_kwargs동일성을assert한다. `probe_topup_ppcc_20260909.py --first`는bread s3의5j reset/문장/접촉을검사한다. no-arg는신규25j검사. collector는기존_v6_apply_jitter를사용한다.

실행은main의with_gpu_lease.sh kanu5아래 `scripts/collect/launch_topup_ppcc_20260909.sh`. GPU당2serve,ports9500/9501,수집서버profile ckpt120000,all_token_full,DiT layers0/2/4/8/10/12/15,4denoise,action5,max720. 별도staging `outputs/collect/grid_staging_topup_ppcc_20260909`. 런처는기존1800skip목록+실제shipped목록을합친DONE_LIST를쓴다.

Shipper는같은GRID_ROOT,INTERVAL60,STAGING_CAP_GB20,PARALLEL6으로분리detached실행. 실제pkl도착후QA/strictindex단계가필요하며수집발사를완료로표기하지않는다. 원격보관기존루트및기존eval은변경하지않는다.

후속부족13scene중이번5scene이후Drawer1,Dishwasher3,Oven최대2(layout2양측;추가2부족근거검토)후속조사. Claude데이터추가수집세션에CPUfeasibility/kscan자료만요청했고GPU발사하지않도록전달했다. 수집진행중plan/collector/runner파일수정금지.

## 2026-09-09 04:27 UTC GPU 5/6/7 재개

사용자 요청으로 kanu GPU 5,6,7 각 2 serve로 PPCC 추가 수집 재개. 동일 plan 4fa6496cd684, 기존 29셀(archive 27 + local 2)을 건너뛰어 잔여 221셀. Collector wrapper PID 3640447, shipper PID 3640448; 로그 outputs/collect/topup_ppcc_20260909/collector_resume_567.log 및 shipper_resume_567.log. 이전 paused.json은 중지 당시 이력이고 현재 상태는 resume_567.json 참조. Coffee/apple 제외 유지.

## A100 추가 수집 (2026-09-09)

PPCC250판 수집 완료 후 새 plan `4ab360df2b71` (기존 canonical1800+PPCC250 보존, 신규 pull250)으로 발사. canonical 신 키 기준:
- srv48 GPU1, 6serve, DishwasherRack/out-right s3 L7 seed100421, s4 L2 seed100280; out-left s3 L8 seed100031. 총150판.
- srv50 GPU2, 6serve, OvenRack/out-right s3 L2 seed100040; out-left s3 L2 seed100518. 총100판.
- 각 scene 5j×n10. 양 머신 ports9560..9565. 해당 신규 scene은 이후 worker1/worker2 원 수집 머신에서 eval.
- Oven/out-left 후보100691은 j4충돌,100451은 j3충돌로 제외. 최종100518은5j 통과. target10내 Oven/out-left 부족2scene은 여전히 불가.
- 실행 코드 worktree `.claude/worktrees/collect-topup`, remoteHEAD d9f4894. 메인 eval checkout 유지. `lerobot/src` 는 main의 동일 패키지에 심볼릭 링크 (worktree 빈 dependency 디렉토리 때문에 최초 serve import 실패해 수집 시작 전 중단·연결 후 재발사).
- 실제 collector `_v6_apply_jitter` 최신 fixture-side left=>+l/right=>-l, production reset/contact/base 검사. srv48 15/15, srv50 right5/5 및 최종left5/5 통과.
- launcher `scripts/collect/launch_topup_remaining_20260909.sh`, 환경 GPUS/INSTRUCTIONS/COLLECTION_SHARD, 원격 hostconda serve + Docker robocasa collector.
- 로컬 durable leasewrapper/전송 SSH PID 및 로그 `outputs/collect/topup_remaining_20260909/{srv48,srv50}_pids.json`, *_collector.log, *_shipper.log.
- 원격 staging `outputs/collect/grid_staging_topup_remaining_20260909_{srv48,srv50}`, shipper 동일 루트→승준 archive. 초기 빈 디렉토리 sent_tally 경고 후 다음 cycle 정상 생성.
- PPCC의 coffee/apple 제외는 그대로. 서랍 새 scene은 reset k-scan 별도 검증 중으로 이 plan에는 없음.
