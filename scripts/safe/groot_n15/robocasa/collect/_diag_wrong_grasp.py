"""wrong-grasp 술어 런타임 진단 (robocasa 컨테이너): env.objects 키·distractor 발견·술어 무예외 확인."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/temporal_vla/scripts/safe/groot_n16/robocasa/collect")

import gymnasium as gym  # noqa: E402
import robocasa  # noqa: F401, E402
import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401, E402
from robocasa_event_labeler import make_robocasa_event_labeler, find_robocasa_env  # noqa: E402

env = gym.make(
    "robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env",
    enable_render=True,
    seed=100084,
)
env.reset(seed=100084)
lab = make_robocasa_event_labeler(env, "PickPlaceCounterToCabinet", proximity_phases=True)
det = lab.probe
kenv = find_robocasa_env(env)
print("env.objects keys:", sorted(kenv.objects.keys()))
names = det._distractor_names()
print("distractor names:", names)
print("wrong_grasped() (reset 직후, False 기대):", det.wrong_grasped())
print("near_obj():", det.near_obj(), "| near_target():", det.near_target())
# 예외가 조용히 먹히는지 직접 확인: raw 호출
import robocasa.utils.object_utils as OU  # noqa: E402
for n in names:
    try:
        print(f"  raw check_obj_grasped({n!r}) =", bool(OU.check_obj_grasped(kenv, n)))
    except Exception as e:  # noqa: BLE001
        print(f"  raw check_obj_grasped({n!r}) EXCEPTION: {type(e).__name__}: {e}")
print("DIAG_OK")
