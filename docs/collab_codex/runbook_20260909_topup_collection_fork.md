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
