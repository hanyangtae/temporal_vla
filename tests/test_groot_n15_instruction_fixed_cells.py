from __future__ import annotations

import csv
from pathlib import Path


CONFIG_PATH = Path("configs/robocasa/n15_instruction_fixed_cells.tsv")


EXPECTED_CELLS = {
    "ppcs_onion": (
        0,
        "PickPlaceCounterToStove",
        "Pick the onion from the plate and place it in the pan.",
    ),
    "ppcs_apple": (
        1,
        "PickPlaceCounterToStove",
        "Pick the apple from the plate and place it in the pan.",
    ),
    "ppdc_tongs": (
        2,
        "PickPlaceDrawerToCounter",
        "Pick the tongs from the drawer and place it on the counter.",
    ),
    "ppdc_wooden_spoon": (
        3,
        "PickPlaceDrawerToCounter",
        "Pick the wooden spoon from the drawer and place it on the counter.",
    ),
    "ppcc_potato": (
        4,
        "PickPlaceCounterToCabinet",
        "Pick the potato from the counter and place it in the cabinet.",
    ),
    "ppcc_bread": (
        5,
        "PickPlaceCounterToCabinet",
        "Pick the bread from the counter and place it in the cabinet.",
    ),
    "open_cabinet_door": (6, "OpenCabinet", "Open the cabinet door."),
    "open_drawer_right": (7, "OpenDrawer", "Open the right drawer."),
    "open_drawer_left": (8, "OpenDrawer", "Open the left drawer."),
    "close_toaster_oven_door": (
        9,
        "CloseToasterOvenDoor",
        "Close the toaster oven door.",
    ),
    "turn_on_microwave": (
        10,
        "TurnOnMicrowave",
        "Press the start button on the microwave.",
    ),
    "turn_on_sink_faucet": (
        11,
        "TurnOnSinkFaucet",
        "Turn on the sink faucet.",
    ),
    "navigate_fridge": (12, "NavigateKitchen", "Navigate to the fridge."),
    "navigate_coffee_machine": (
        13,
        "NavigateKitchen",
        "Navigate to the coffee machine.",
    ),
    "coffee_setup_mug": (
        14,
        "CoffeeSetupMug",
        "Pick the mug from the counter and place it under the coffee machine dispenser.",
    ),
}

def test_instruction_fixed_cell_config_is_the_single_source_of_truth():
    with CONFIG_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    assert len(rows) == 15
    assert set(row["cell_id"] for row in rows) == set(EXPECTED_CELLS)
    assert sorted(int(row["cell_index"]) for row in rows) == list(range(15))
    assert all(int(row["target_episodes"]) == 50 for row in rows)
    assert all(int(row["seed_search_start"]) == 100000 for row in rows)
    assert "env_kwargs_json" not in rows[0]

    for row in rows:
        expected_index, expected_task, expected_instruction = EXPECTED_CELLS[row["cell_id"]]
        assert int(row["cell_index"]) == expected_index
        assert row["task"] == expected_task
        assert row["env_name"] == f"robocasa_panda_omron/{expected_task}_PandaOmron_Env"
        assert row["canonical_instruction"] == expected_instruction
