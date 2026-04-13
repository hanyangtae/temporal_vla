"""Test pi0.5 inference with dataset sample directly (model card approach)."""
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors

device = torch.device("cuda")
policy = PI05Policy.from_pretrained("lerobot/pi05_libero_base").to(device).eval()
preprocessor, postprocessor = make_pre_post_processors(
    policy.config, "lerobot/pi05_libero_base",
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

ds = LeRobotDataset("lerobot/libero")

# Test on first few frames
for idx in [0, 50, 100]:
    frame = dict(ds[idx])
    gt_action = frame["action"]
    task_str = frame["task"]
    state = frame["observation.state"]

    batch = preprocessor(frame)

    policy.reset()
    with torch.inference_mode():
        pred_action = policy.select_action(batch)
    pred_action = postprocessor(pred_action)

    print(f"--- Frame {idx} ---")
    print(f"  Task: {task_str}")
    print(f"  State (raw): {state.numpy()}")
    print(f"  State (preprocessed): {batch['observation.state'].cpu().numpy()}")
    print(f"  GT action:   {gt_action.numpy()}")
    print(f"  Pred action: {pred_action[0].cpu().numpy()}")
