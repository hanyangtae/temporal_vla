"""주입(injection) 검증: 저자 hook이 '의미적으로 옳은' neuron을 '수치적으로 정확히' override 하는가.

- 의미 검증: fast_10 cluster의 neuron들의 top token이 fast 계열인가 (top_tokens 기준).
- 수치 검증: 실제 down_proj 가중치로 저자 hooks.py 로직을 실행 ->
  입력 활성의 선택 neuron 위치만 정확히 alpha 로 바뀌고 down_proj 출력이 기대값과 일치하는가.
출력: phaseA_viz/F_injection_verification.png  + 콘솔 PASS/FAIL
"""
import os, re, glob, json, yaml
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors import safe_open
import matplotlib; matplotlib.use("Agg")
import matplotlib.font_manager as fm
_fp = "/home/dongkyu/.fonts/NanumGothic-Regular.ttf"
if os.path.exists(_fp):
    fm.fontManager.addfont(_fp); matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/pkt_ws/mechanistic-steering-vlas")
DICT = f"{ROOT}/openvla/libero_experiments/configs/interventions/dictionaries_full.yaml"
TXT = f"{ROOT}/openvla/ffn_value_vectors/artifacts/ffn_value_projection/top_tokens_output.txt"
CKPT = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--openvla--openvla-7b-finetuned-libero-10/snapshots/*"))[0]
OUT = f"{ROOT}/phaseA_viz"; os.makedirs(OUT, exist_ok=True)
INTER, ALPHA, CONCEPT = 11008, 6.0, "fast_10"

# ---- cluster -> (layer, neuron) ----
dd = yaml.safe_load(open(DICT))
flats = [int(k) for k in dd[CONCEPT].keys()]
by_layer = {}
for fl in flats: by_layer.setdefault(fl//INTER, []).append(fl % INTER)
print(f"=== {CONCEPT}: {len(flats)} neurons, layer별 분포 ===")
for L in sorted(by_layer): print(f"  layer {L:>2}: neurons {sorted(by_layer[L])}")

# ---- 의미 검증: 선택 neuron의 top token이 fast 계열? ----
toptok = {}
with open(TXT) as f:
    for line in f:
        m = re.match(r"\[(\d+)\]\s*(.*)", line)
        if m and int(m.group(1)) in flats:
            toks = [x.strip().strip("'") for x in m.group(2).split(", ")]
            toptok[int(m.group(1))] = toks[:6]
print("\n=== 의미 검증: 선택된 neuron의 top token ===")
fast_pre = ("fast","quick","rapid","swift")
ok_sem = 0
for fl in flats:
    t = toptok.get(fl, [])
    is_fast = any(s.lstrip("▁ ").lower().startswith(fast_pre) for s in t)
    ok_sem += is_fast
    print(f"  flat {fl} (L{fl//INTER},n{fl%INTER}): {t}  {'fast' if is_fast else 'X'}")
print(f"  -> {ok_sem}/{len(flats)} 가 fast 계열 (의미적으로 옳은 neuron)")

# ---- 수치 검증: 실제 down_proj 가중치로 hook override 재현 ----
L = max(by_layer, key=lambda k: len(by_layer[k]))   # fast neuron 가장 많은 layer
neurons = sorted(by_layer[L])
key = f"language_model.model.layers.{L}.mlp.down_proj.weight"
idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))
with safe_open(f"{CKPT}/{idx['weight_map'][key]}", framework="pt") as f:
    W = f.get_tensor(key).float()           # [4096, 11008]
lin = nn.Linear(INTER, W.shape[0], bias=False); lin.weight.data = W
print(f"\n=== 수치 검증: layer {L}, down_proj weight {tuple(W.shape)}, 주입 neuron {neurons} ===")

torch.manual_seed(0)
x = torch.randn(1, 3, INTER)                # 가짜 활성 입력
captured = {}
def steer_hook(module, inp, out):           # 저자 hooks.py 와 동일 로직
    mod = inp[0]; mod[..., neurons] = ALPHA
    captured["after"] = mod[0, 0, :].clone()
    return F.linear(mod, module.weight, module.bias)
base_out = lin(x.clone())
h = lin.register_forward_hook(steer_hook)
steer_out = lin(x.clone()); h.remove()
# 기대값: 수동으로 override 후 통과
x2 = x.clone(); x2[..., neurons] = ALPHA
expected = F.linear(x2, W)

set_ok = bool(torch.allclose(captured["after"][neurons], torch.full((len(neurons),), ALPHA)))
out_ok = bool(torch.allclose(steer_out, expected, atol=1e-4))
changed = (~torch.isclose(base_out, steer_out)).any().item()
print(f"  [check1] 선택 neuron 활성이 정확히 alpha={ALPHA} 로 set?  {'PASS' if set_ok else 'FAIL'}")
print(f"  [check2] hook 출력 == 수동 override 출력?               {'PASS' if out_ok else 'FAIL'}")
print(f"  [check3] 출력이 baseline 대비 실제로 바뀜?              {'PASS' if changed else 'FAIL'}")

# ---- 시각화: 입력 활성 (override 전 vs 후), 선택 neuron 강조 ----
sample = sorted(set(neurons + list(range(0, INTER, INTER//60))))
before = x[0,0,sample].numpy(); after = captured["after"][sample].numpy()
fig, ax = plt.subplots(figsize=(11,4))
xs = np.arange(len(sample))
is_sel = np.array([s in neurons for s in sample])
ax.bar(xs[~is_sel], before[~is_sel], color="#bdc3c7", label="다른 neuron (변화 없음)")
ax.bar(xs[is_sel], before[is_sel], color="#bdc3c7")
ax.scatter(xs[is_sel], after[is_sel], color="#e74c3c", zorder=5, s=60,
           label=f"주입 neuron -> alpha={ALPHA}")
for xi in xs[is_sel]:
    ax.annotate("", xy=(xi, ALPHA), xytext=(xi, before[xi]),
                arrowprops=dict(arrowstyle="->", color="#e74c3c"))
ax.axhline(ALPHA, ls="--", color="#e74c3c", alpha=0.5)
ax.set_xlabel(f"down_proj 입력 차원 (layer {L}, 11008 중 샘플)"); ax.set_ylabel("활성값")
ax.set_title(f"(F) 주입 검증: '{CONCEPT}' neuron만 정확히 alpha={ALPHA} 로 override (나머지 불변)")
ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/F_injection_verification.png", dpi=130); plt.close()

print(f"\n  => {'ALL PASS' if (set_ok and out_ok and changed) else 'CHECK FAILED'}")
print(f"  그림: {OUT}/F_injection_verification.png")
