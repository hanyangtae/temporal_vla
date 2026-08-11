"""보충 계획 v1b — PPCC apple(SR 1.00, 실패 0) 교체 후보 2종 (2026-08-06).

v1(`n15_grid_v1`, plan_id b8054b5e7258)의 `instructions` 를 고치면 plan_id 가 바뀌어
기수집 트리 전체가 다른 좌표가 되므로, 교체 instruction 은 **별도 계획**으로 추가한다
(plan_id 가 좌표에 포함되어 같은 store 에 충돌 없이 쌓임). apple 30판은 v1 에 보존
(고SR 성공 분포 용도, m 연장 없음).

- 물체 선정: 사용자 확정 marshmallow(17 scene)·jug(15 scene). beer 기각.
- SR 미지 → m3 로 먼저 재고(SR 파일럿 겸용) 극단이면 재교체.
- scene seed 출처: seed_scan PPCC TSV 2장 (100000–101000) 오름차순 앞 10개.
- noise·캡처·ckpt 는 v1 과 동일 (같은 noise_seeds 40 선언, 1라운드 m3).
- 머신 배정: 둘 다 srv50(worker2) — kanu 는 타 세션 GPU 점유로 보류(2026-08-06).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from src.collect.plan import CollectionPlan  # noqa: E402

_PPCC_ENV = "robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env"

SCENES = {
    "PPCC/marshmallow": [100007, 100179, 100222, 100424, 100553, 100571, 100607, 100645, 100659, 100665],
    "PPCC/jug":         [100095, 100128, 100137, 100360, 100387, 100443, 100447, 100574, 100590, 100591],
}

INSTRUCTION_TEXT = {
    "PPCC/marshmallow": "Pick the marshmallow from the counter and place it in the cabinet.",
    "PPCC/jug": "Pick the jug from the counter and place it in the cabinet.",
}


def build() -> CollectionPlan:
    for k, v in SCENES.items():
        assert len(v) == 10 and len(set(v)) == 10, k
    return CollectionPlan(
        name="n15_grid_v1b",
        model="groot",
        version="n15",
        ckpt="lerobot_groot_n15__robocasa365_ckpt120000",
        capture_layers=[0, 2, 4, 8, 10, 12, 15],
        denoise_k=4,
        token_mode="all_token_full",
        instructions=SCENES,
        noise_seeds=list(range(1300000, 1300040)),
        note=(
            "v1 보충 — apple(SR 1.00) 교체 후보 marshmallow·jug. SR 미지라 m3 파일럿 겸용. "
            "eval inference_seed 는 1400000+ 대역 (v1 과 동일 규약)."
        ),
        extra={
            "env_names": {k: _PPCC_ENV for k in SCENES},
            "instruction_text": INSTRUCTION_TEXT,
            "machine_assignment": {"srv50": list(SCENES)},
            "supplements": "n15_grid_v1 (b8054b5e7258) — apple 대체 후보",
            "seed_provenance": "outputs/analysis/seed_scan/PickPlaceCounterToCabinet{,_ext}.tsv",
        },
    )


def main() -> None:
    plan = build()
    path = plan.save(REPO_ROOT / "configs" / "collect" / "n15_grid_v1b")
    print(f"plan_id = {plan.plan_id}  n_cells = {plan.n_cells}  wrote {path}")


if __name__ == "__main__":
    main()
