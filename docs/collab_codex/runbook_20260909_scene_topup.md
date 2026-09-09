# 新scene 수집 시작

사용자 09-09 지시: eval 미착수작업이 없으면 kanu GPU5에서2serve로 신규scene j5×n10수집.
유효17scene을각instruction3scene로보충하는최초부족19scene(950판)은 후보수집량이며성공조건충족보장아님.
부족표는 `configs/collect/n15_scene_topup_20260909/requests.json`.

첫별도plan `configs/collect/n15_scene_topup_coffee_20260909/collection_plan.json` (aa4d86162226):
Coffee 3scene layout/style 6/6 seed100396,8/8 seed100229,1/1 seed100161,각50판=150판.
기존4/4·9/9·5/5와불중복. 원target10 scan과중앙spawn순서로선정; policy라벨로선정하지않음.
새plan의s0/1/2는기존v6 s0/1/2와다른물리scene다. 기존plan/raw/index불변.

GR00T N1.5 ckpt120000,캡처L0/2/4/8/10/12/15×denoise4×all49token,VL포함,
16예측/5실행,noise1300000..1300009,reset j0..4,env_kwargs10주방을명시.
기존수집재현gate는머신/파이프라인단위통과이력을보존하며baseline eval을추가하지않음.
새scene reset/문장/base/접촉검사는 `probe_v6_jitter_cells.py`로정책없이수행.
QA(실제layout/style,유효한라벨/좌표,VL정지·영상freeze)는수집후확인;정상timeout/실패는제외하지않음.
새수집자료는검증전fit/eval의입력에자동합치지않음.

발사: `with_gpu_lease.sh kanu 5 codex-topup-coffee collect -- bash scripts/collect/launch_scene_topup_coffee_20260909.sh`
수집staging `outputs/collect/grid_staging_topup_coffee_20260909`,포트9500/9501,backpressure30GB.
기존rolling shipper(PARALLEL2,INTERVAL60)를같이돌려원격
`~/datasets/temporal_vla_store/groot/n15/grid/<새plan>/kanu/...`로보관한다.
전송개수/바이트대조통과후staging을비움.로컬89GB/원격992GB여유확인(수집전).
로그/PID: `outputs/collect/topup_coffee_20260909/`.

완료후각50좌표,n0..9×j0..4,plan/model/env/inference seed/layout/style/문장검증,
`verify_grid.py --require-complete`, `build_grid_index.py --strict`와영상/VL검사를진행한다.
Detector의targetj전체제외other40양클래스,연산자의target성공제외계약을충족하는j를재선정한다.
원자료를옛v6의50판에합쳐학습량을늘리지않는다.
