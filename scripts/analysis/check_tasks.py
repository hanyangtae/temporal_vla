"""Compare model predictions with/without normalization on dataset samples."""
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import hotswap_stats
from lerobot.processor.normalize_processor import _NormalizationMixin

device = torch.device("cuda")
policy = PI05Policy.from_pretrained("lerobot/pi05_libero_base").to(device).eval()
ds = LeRobotDataset("lerobot/libero")

# Without stats
pre_raw, post_raw = make_pre_post_processors(
    policy.config, "lerobot/pi05_libero_base",
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

# With stats
pre_norm, post_norm = make_pre_post_processors(
    policy.config, "lerobot/pi05_libero_base",
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)
pre_norm = hotswap_stats(pre_norm, ds.meta.stats)
post_norm = hotswap_stats(post_norm, ds.meta.stats)
features = {**policy.config.input_features, **policy.config.output_features}
for step in pre_norm.steps:
    if isinstance(step, _NormalizationMixin) and not step.features:
        step.features = features
for step in post_norm.steps:
    if isinstance(step, _NormalizationMixin) and not step.features:
        step.features = features

for idx in [0, 50, 100, 200]:
    frame_raw = dict(ds[idx])
    frame_norm = dict(ds[idx])
    gt = frame_raw["action"]

    batch_raw = pre_raw(frame_raw)
    policy.reset()
    with torch.inference_mode():
        pred_raw = policy.select_action(batch_raw)
    pred_raw = post_raw(pred_raw)[0].cpu()

    batch_norm = pre_norm(frame_norm)
    policy.reset()
    with torch.inference_mode():
        pred_norm = policy.select_action(batch_norm)
    pred_norm = post_norm(pred_norm)[0].cpu()

    err_raw = (pred_raw - gt).abs().mean().item()
    err_norm = (pred_norm - gt).abs().mean().item()

    print(f"Frame {idx:4d} | GT: {gt.numpy()} ")
    print(f"           | no_norm: {pred_raw.numpy()} (MAE={err_raw:.4f})")
    print(f"           | w_norm:  {pred_norm.numpy()} (MAE={err_norm:.4f})")
    print()
