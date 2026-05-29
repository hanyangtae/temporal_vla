# RoboCasa Task Name Mapping: GR00T Fork (v0.2) vs Our robocasa365 (v1.0)

> GR00T fork: `src/policies/Isaac-GR00T/external_dependencies/robocasa/`
> Our robocasa365: `src/benchmarks/robocasa/`
>
> GR00T fork 구조: `environments/kitchen/{single_stage,multi_stage}/`
> Our robocasa365 구조: `environments/kitchen/{atomic,composite}/`

## Eval Batch 태스크 (12개, `eval_groot_robocasa.sh` local-batch)

| # | GR00T gym env name | GR00T class | Our robocasa365 class | 이름 변경? |
|---|---|---|---|---|
| 1 | `robocasa_panda_omron/PnPCounterToMicrowave_PandaOmron_Env` | `PnPCounterToMicrowave` | `PickPlaceCounterToMicrowave` | **Yes** |
| 2 | `robocasa_panda_omron/PnPMicrowaveToCounter_PandaOmron_Env` | `PnPMicrowaveToCounter` | `PickPlaceMicrowaveToCounter` | **Yes** |
| 3 | `robocasa_panda_omron/CoffeeSetupMug_PandaOmron_Env` | `CoffeeSetupMug` | `CoffeeSetupMug` | No |
| 4 | `robocasa_panda_omron/TurnOffStove_PandaOmron_Env` | `TurnOffStove` | `TurnOffStove` | No |
| 5 | `robocasa_panda_omron/OpenDoubleDoor_PandaOmron_Env` | `OpenDoubleDoor` | -- (없음) | **N/A** |
| 6 | `robocasa_panda_omron/OpenDrawer_PandaOmron_Env` | `OpenDrawer` | `OpenDrawer` | No |
| 7 | `robocasa_panda_omron/PnPCounterToCab_PandaOmron_Env` | `PnPCounterToCab` | `PickPlaceCounterToCabinet` | **Yes** |
| 8 | `robocasa_panda_omron/PnPCabToCounter_PandaOmron_Env` | `PnPCabToCounter` | `PickPlaceCabinetToCounter` | **Yes** |
| 9 | `robocasa_panda_omron/PnPCounterToSink_PandaOmron_Env` | `PnPCounterToSink` | `PickPlaceCounterToSink` | **Yes** |
| 10 | `robocasa_panda_omron/PnPSinkToCounter_PandaOmron_Env` | `PnPSinkToCounter` | `PickPlaceSinkToCounter` | **Yes** |
| 11 | `robocasa_panda_omron/PnPCounterToStove_PandaOmron_Env` | `PnPCounterToStove` | `PickPlaceCounterToStove` | **Yes** |
| 12 | `robocasa_panda_omron/PnPStoveToCounter_PandaOmron_Env` | `PnPStoveToCounter` | `PickPlaceStoveToCounter` | **Yes** |

**요약**: 7/12 이름 변경 (PnP→PickPlace, Cab→Cabinet), 4/12 동일, 1/12 (`OpenDoubleDoor`) v1.0에 없음

---

## Atomic / Single-Stage 전체 비교

### Pick and Place

| GR00T fork (v0.2) | Our robocasa365 (v1.0) | 비고 |
|---|---|---|
| `PnPCounterToCab` | `PickPlaceCounterToCabinet` | **[EVAL]** |
| `PnPCabToCounter` | `PickPlaceCabinetToCounter` | **[EVAL]** |
| `PnPCounterToSink` | `PickPlaceCounterToSink` | **[EVAL]** |
| `PnPSinkToCounter` | `PickPlaceSinkToCounter` | **[EVAL]** |
| `PnPCounterToMicrowave` | `PickPlaceCounterToMicrowave` | **[EVAL]** |
| `PnPMicrowaveToCounter` | `PickPlaceMicrowaveToCounter` | **[EVAL]** |
| `PnPCounterToStove` | `PickPlaceCounterToStove` | **[EVAL]** |
| `PnPStoveToCounter` | `PickPlaceStoveToCounter` | **[EVAL]** |
| -- | `PickPlaceCounterToOven` | v1.0 only |
| -- | `PickPlaceCounterToDrawer` | v1.0 only |
| -- | `PickPlaceDrawerToCounter` | v1.0 only |
| -- | `PickPlaceCounterToBlender` | v1.0 only |
| -- | `PickPlaceToasterToCounter` | v1.0 only |
| -- | `PickPlaceCounterToToasterOven` | v1.0 only |
| -- | `PickPlaceToasterOvenToCounter` | v1.0 only |
| -- | `PickPlaceCounterToStandMixer` | v1.0 only |
| -- | `PickPlaceFridgeShelfToDrawer` | v1.0 only |
| -- | `PickPlaceFridgeDrawerToShelf` | v1.0 only |

### Doors

| GR00T fork (v0.2) | Our robocasa365 (v1.0) | 비고 |
|---|---|---|
| `OpenDoor` | `OpenDoor` | 동일 (base class) |
| `CloseDoor` | `CloseDoor` | 동일 (base class) |
| `OpenSingleDoor` | `OpenCabinet` | 이름 변경 |
| `CloseSingleDoor` | `CloseCabinet` | 이름 변경 |
| `OpenDoubleDoor` | -- | **[EVAL]** v0.2 only |
| `CloseDoubleDoor` | -- | v0.2 only |
| -- | `OpenMicrowave` / `CloseMicrowave` | v1.0 only |
| -- | `OpenFridge` / `CloseFridge` | v1.0 only |
| -- | `OpenOven` / `CloseOven` | v1.0 only |
| -- | `OpenDishwasher` / `CloseDishwasher` | v1.0 only |
| -- | `OpenToasterOvenDoor` / `CloseToasterOvenDoor` | v1.0 only |

### Drawers

| GR00T fork (v0.2) | Our robocasa365 (v1.0) | 비고 |
|---|---|---|
| `OpenDrawer` | `OpenDrawer` | **[EVAL]** 동일 |
| `CloseDrawer` | `CloseDrawer` | 동일 |
| -- | `OpenFridgeDrawer` / `CloseFridgeDrawer` | v1.0 only |
| -- | `SlideDishwasherRack` | v1.0 only |

### Stove / Microwave / Coffee / Sink

| GR00T fork (v0.2) | Our robocasa365 (v1.0) | 비고 |
|---|---|---|
| `TurnOnStove` | `TurnOnStove` | 동일 |
| `TurnOffStove` | `TurnOffStove` | **[EVAL]** 동일 |
| `TurnOnMicrowave` | `TurnOnMicrowave` | 동일 |
| `TurnOffMicrowave` | `TurnOffMicrowave` | 동일 |
| `CoffeeSetupMug` | `CoffeeSetupMug` | **[EVAL]** 동일 |
| `CoffeeServeMug` | `CoffeeServeMug` | 동일 |
| `CoffeePressButton` | `StartCoffeeMachine` | 이름 변경 |
| `TurnOnSinkFaucet` | `TurnOnSinkFaucet` | 동일 |
| `TurnOffSinkFaucet` | `TurnOffSinkFaucet` | 동일 |
| `TurnSinkSpout` | `TurnSinkSpout` | 동일 |
| `NavigateKitchen` | `NavigateKitchen` | 동일 |
| -- | `LowerHeat` | v1.0 only |
| -- | `AdjustWaterTemperature` | v1.0 only |

### v1.0 신규 가전 태스크

- Oven: `PreheatOven`, `SlideOvenRack`
- Blender: `OpenBlenderLid`, `CloseBlenderLid`, `TurnOnBlender`
- Toaster: `TurnOnToaster`
- Toaster Oven: `AdjustToasterOvenTemperature`, `TurnOnToasterOven`, `SlideToasterOvenRack`
- Stand Mixer: `OpenStandMixerHead`, `CloseStandMixerHead`
- Electric Kettle: `TurnOnElectricKettle`, `OpenElectricKettleLid`, `CloseElectricKettleLid`

---

## 이름 변경 규칙 요약

| 패턴 | v0.2 | v1.0 |
|---|---|---|
| Pick & Place 접두사 | `PnP` | `PickPlace` |
| Cabinet 약어 | `Cab` | `Cabinet` |
| 단일 문 열기 | `OpenSingleDoor` | `OpenCabinet` |
| 커피 버튼 | `CoffeePressButton` | `StartCoffeeMachine` |

## 통계

| | GR00T fork (v0.2) | Our robocasa365 (v1.0) |
|---|---|---|
| Atomic 태스크 클래스 | ~25 | ~75+ |
| Composite 카테고리 | 20 | 60+ |
| 신규 가전 타입 | -- | Oven, Blender, Toaster, ToasterOven, StandMixer, ElectricKettle, Dishwasher |
