"""재수집 라운드 collection_plan.json 빌더 — docs/steering/38 · plan v9.

scene seed 는 2026-08-06 확정값을 **코드에 박아** 둔다 — 계획의 단일 출처는 이
스크립트가 아니라 산출된 `collection_plan.json`(plan_id 로 동결)이지만, 어떤
스캔·필터 근거로 seed 가 뽑혔는지 재현 가능해야 하므로 출처를 주석으로 남긴다.

근거 (전부 outputs/analysis/seed_scan/):
- OpenDrawer left/right: OpenDrawer.tsv (300 seed 스캔) 의 variant 별 앞 10개.
  drawer_feasibility.json 300 건 전수 BLOCKED 0 → 필터 탈락 없음.
- PPCC bread/apple/candle: ppcc_candidates.tsv (100000–101000, 1001 seed 스캔)
  의 앞 10개. PnP 라 관절 스윕 필터 비적용.
- CoffeeSetupMug: 300 seed 전부 동일 instruction → 100000..100009.
- DishwasherRack out: SlideDishwasherRack.tsv 의 out variant 앞 10개.
  dishwasher_feasibility.json 154 건 BLOCKED 0.
- OvenRack out(층 구문 없음): SlideOvenRack.tsv 앞 10개.
  ovenrack_feasibility.json 53 건 BLOCKED 0. ⚠ layout 이 단단 오븐으로
  편향(300 중 104)돼 scene 다양성이 다른 7종과 다르다 — 분석 각주 필수.

noise_seeds 는 40 개 전량 선언, 1라운드 수집은 앞 10개만(러너 NOISE_LIMIT=10).
m 연장은 계획 수정이 아니라 결손 셀 채우기라 plan_id 가 유지된다.
eval 용 1400000+ 대역은 비워둔다 (fit-seed 분리, in-sample rescue 방지).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from src.collect.plan import CollectionPlan  # noqa: E402

SCENES = {
    "PPCC/bread":       [100084, 100197, 100201, 100228, 100317, 100435, 100441, 100459, 100471, 100485],
    "PPCC/apple":       [100025, 100069, 100192, 100195, 100243, 100359, 100374, 100389, 100408, 100489],
    "PPCC/candle":      [100058, 100107, 100154, 100166, 100214, 100218, 100382, 100402, 100676, 100741],
    "OpenDrawer/left":  [100001, 100002, 100004, 100007, 100008, 100013, 100014, 100015, 100017, 100019],
    "OpenDrawer/right": [100000, 100003, 100005, 100006, 100009, 100010, 100011, 100012, 100016, 100018],
    "CoffeeSetupMug":   [100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100008, 100009],
    "DishwasherRack/out": [100000, 100003, 100008, 100010, 100011, 100012, 100013, 100015, 100016, 100017],
    "OvenRack/out":     [100006, 100014, 100016, 100017, 100020, 100023, 100028, 100030, 100037, 100042],
}

_PPCC_ENV = "robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env"
ENV_NAMES = {
    "PPCC/bread": _PPCC_ENV,
    "PPCC/apple": _PPCC_ENV,
    "PPCC/candle": _PPCC_ENV,
    "OpenDrawer/left": "robocasa_panda_omron/OpenDrawer_PandaOmron_Env",
    "OpenDrawer/right": "robocasa_panda_omron/OpenDrawer_PandaOmron_Env",
    "CoffeeSetupMug": "robocasa_panda_omron/CoffeeSetupMug_PandaOmron_Env",
    "DishwasherRack/out": "robocasa_panda_omron/SlideDishwasherRack_PandaOmron_Env",
    "OvenRack/out": "robocasa_panda_omron/SlideOvenRack_PandaOmron_Env",
}

INSTRUCTION_TEXT = {
    "PPCC/bread": "Pick the bread from the counter and place it in the cabinet.",
    "PPCC/apple": "Pick the apple from the counter and place it in the cabinet.",
    "PPCC/candle": "Pick the candle from the counter and place it in the cabinet.",
    "OpenDrawer/left": "Open the left drawer.",
    "OpenDrawer/right": "Open the right drawer.",
    "CoffeeSetupMug": "Pick the mug from the counter and place it under the coffee machine dispenser.",
    "DishwasherRack/out": "Fully slide the top dishwasher rack out.",
    "OvenRack/out": "Fully slide the oven rack out.",
}

# 머신 배정 (docs/steering/38 §3.2 — base·arm 동일 머신, instruction 단위 통째 할당).
# 좌표(plan_id)에는 안 들어가고 러너 INSTRUCTIONS 인자의 참조용이다.
MACHINE_ASSIGNMENT = {
    "kanu": ["OpenDrawer/left", "PPCC/apple"],
    "srv50": ["OpenDrawer/right", "PPCC/candle"],
    "srv48": ["PPCC/bread", "CoffeeSetupMug", "DishwasherRack/out"],
    "pdk_external": ["OvenRack/out"],
}


def build() -> CollectionPlan:
    assert set(SCENES) == set(ENV_NAMES) == set(INSTRUCTION_TEXT)
    assigned = [i for v in MACHINE_ASSIGNMENT.values() for i in v]
    assert sorted(assigned) == sorted(SCENES), "머신 배정이 instruction 집합과 다름"
    for k, v in SCENES.items():
        assert len(v) == 10 and len(set(v)) == 10, k
    return CollectionPlan(
        name="n15_grid_v1",
        model="groot",
        version="n15",
        ckpt="lerobot_groot_n15__robocasa365_ckpt120000",
        capture_layers=[0, 2, 4, 8, 10, 12, 15],
        denoise_k=4,
        token_mode="all_token_full",
        instructions=SCENES,
        noise_seeds=list(range(1300000, 1300040)),  # 40 선언, 1라운드는 앞 10만 수집
        note=(
            "2026-08-06 재수집 라운드 (docs/steering/38). machine 좌표 기록이 목적. "
            "eval inference_seed 는 1400000+ 대역 사용 (fit 분리)."
        ),
        extra={
            "env_names": ENV_NAMES,
            "instruction_text": INSTRUCTION_TEXT,
            "machine_assignment": MACHINE_ASSIGNMENT,
            "round1_noise_limit": 10,
            "seed_provenance": "outputs/analysis/seed_scan/ (스캔 TSV + feasibility JSON 3종)",
        },
    )


def main() -> None:
    out_dir = REPO_ROOT / "configs" / "collect" / "n15_grid_v1"
    plan = build()
    path = plan.save(out_dir)
    print(f"plan_id = {plan.plan_id}")
    print(f"n_cells = {plan.n_cells} (선언; 1라운드 = {plan.n_cells // 4})")
    print(f"estimated full = {plan.estimate_bytes() / 2**40:.2f} TiB")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
