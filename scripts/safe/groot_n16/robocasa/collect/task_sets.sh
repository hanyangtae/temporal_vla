#!/usr/bin/env bash
# Shared GR00T N1.6 RoboCasa rollout collection task sets.

set -euo pipefail

normalize_robocasa_collection_env_source() {
    local source="${1:-robocasa_v02}"

    case "${source}" in
        robocasa_v02)
            echo "robocasa_v02"
            ;;
        robocasa365)
            echo "robocasa365"
            ;;
        *)
            echo "ERROR: unknown RoboCasa env source=${source}" >&2
            echo "Known env sources: robocasa_v02, robocasa365" >&2
            return 2
            ;;
    esac
}

load_robocasa_collection_task_set() {
    local task_set="${1:-safe_seen6}"

    case "${task_set}" in
        safe_seen6)
            ROBOCASA_ENV_SOURCE_FOR_TASK_SET="robocasa_v02"
            ROBOCASA_SEED_START_FOR_TASK_SET=241
            TASKS=(
                # RoboCasa v0.2 names. robocasa365 mappings:
                # CoffeeSetupMug -> CoffeeSetupMug
                # OpenSingleDoor -> OpenCabinet
                # PnPCounterToCab -> PickPlaceCounterToCabinet
                # PnPSinkToCounter -> PickPlaceSinkToCounter
                # PnPCounterToStove -> PickPlaceCounterToStove
                # OpenDrawer -> OpenDrawer
                "CoffeeSetupMug"
                "OpenSingleDoor"
                "PnPCounterToCab"
                "PnPSinkToCounter"
                "PnPCounterToStove"
                "OpenDrawer"
            )
            ;;
        target_atomic_seen18 | robocasa365_atomic_seen18)
            ROBOCASA_ENV_SOURCE_FOR_TASK_SET="robocasa365"
            ROBOCASA_SEED_START_FOR_TASK_SET=100000
            TASKS=(
                "CloseBlenderLid"
                "CloseFridge"
                "CloseToasterOvenDoor"
                "CoffeeSetupMug"
                "NavigateKitchen"
                "OpenCabinet"
                "OpenDrawer"
                "OpenStandMixerHead"
                "PickPlaceCounterToCabinet"
                "PickPlaceCounterToStove"
                "PickPlaceDrawerToCounter"
                "PickPlaceSinkToCounter"
                "PickPlaceToasterToCounter"
                "SlideDishwasherRack"
                "TurnOffStove"
                "TurnOnElectricKettle"
                "TurnOnMicrowave"
                "TurnOnSinkFaucet"
            )
            ;;
        *)
            echo "ERROR: unknown TASK_SET=${task_set}" >&2
            echo "Known task sets: safe_seen6, target_atomic_seen18" >&2
            exit 2
            ;;
    esac

    if [[ -n "${TASKS_OVERRIDE:-}" ]]; then
        # shellcheck disable=SC2206
        TASKS=( ${TASKS_OVERRIDE} )
    fi
}
