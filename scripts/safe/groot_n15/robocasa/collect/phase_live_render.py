"""라이브 롤아웃 산출물(frames.npz + meta.pkl) → 참고영상 스타일 mp4.

레이아웃: 위=로봇 3뷰 프레임, 오른쪽=현재 phase 현황(AE/SAE cluster→phase), 아래=시간축 phase 리본.
t-SNE 없음. AE/SAE 는 serve --phase-readout 가 라이브로 산출한 값.
"""
from __future__ import annotations
import argparse, pickle
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

# 13-phase 팔레트 (task_classification meta phase_vocab 순서)
PALETTE = ["#2a78d6","#1baf7a","#eda100","#008300","#4a3aa7","#e87ba4","#e34948",
           "#17a2b8","#8e6fd0","#8c564b","#2bb5a0","#b5179e","#7f8c00"]
NAMES = ["reach-to-object","grasp","transport","place","insert-settle","terminal",
         "wrong-grasp","reach-to-handle","grasp-handle","pull","open-done","disengage","push-back"]
NAME2ID = {n: i for i, n in enumerate(NAMES)}
GREY = "#c9c8c4"

def hexrgb(h): return tuple(int(h[i:i+2],16) for i in (1,3,5))
def font(sz, bold=False):
    for p in (("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),):
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()

def pid_of(entry):
    """live phase dict -> phase_id (색/리본용). None 이면 -1."""
    if not isinstance(entry, dict): return -1
    return NAME2ID.get(entry.get("phase"), -1)

def gt_pid(name):
    """GT phase 이름 -> phase_id. NAME2ID 에 없으면 별칭 매핑 시도, 그래도 없으면 -1."""
    if not name: return -1
    if name in NAME2ID: return NAME2ID[name]
    alias = {"reach-obj": "reach-to-object", "reach": "reach-to-object",
             "insert": "insert-settle", "reach-handle": "reach-to-handle"}
    return NAME2ID.get(alias.get(name, ""), -1)

def _live(p, key):
    """phases 항목에서 live AE/SAE dict 추출 ({live:{ae,sae},gt:...} 또는 구형 {ae,sae})."""
    live = p.get("live", p) if isinstance(p, dict) else {}
    return live.get(key, {}) if isinstance(live, dict) else {}

def render(base: Path, out: Path, fps=20):
    frames = np.load(base/"frames.npz")["frames"]
    meta = pickle.load(open(base/"meta.pkl","rb"))
    phases = meta["phases"]
    N = len(frames)
    ae_ids = [pid_of(_live(p, "ae")) for p in phases]
    sae_ids = [pid_of(_live(p, "sae")) for p in phases]
    gt_ids = [gt_pid(p.get("gt") if isinstance(p, dict) else None) for p in phases]
    has_gt = any(g >= 0 for g in gt_ids)

    fh, fw = frames[0].shape[:2]
    PANEL_W = 300            # 오른쪽 현황 패널
    n_rib = 3 if has_gt else 2
    RIBBON_H = 30 + n_rib * 40   # GT/AE/SAE 리본
    W = fw + PANEL_W
    H = max(fh, 190) + RIBBON_H
    f_hd, f_row, f_sm = font(22, True), font(20, True), font(14)
    x0, x1 = 12, fw - 12     # 리본 x 범위 (프레임 폭 기준)

    def ribbon(d, y, ids, label):
        d.text((12, y - 18), label, font=f_sm, fill=(60,60,60))
        for i in range(N):
            xa = x0 + int(i/N*(x1-x0)); xb = x0 + int((i+1)/N*(x1-x0))
            pid = ids[i]
            col = hexrgb(PALETTE[pid]) if pid>=0 else hexrgb(GREY)
            d.rectangle([xa, y, max(xa,xb-1), y+18], fill=col)

    writer = imageio.get_writer(str(out), fps=fps, codec="libx264",
                                quality=8, macro_block_size=None)
    instr = meta.get("instruction","")
    tag = "success" if meta.get("success") else "running/fail"
    for i in range(N):
        img = Image.new("RGB", (W, H), (252,252,251))
        img.paste(Image.fromarray(frames[i]), (0, 0))
        d = ImageDraw.Draw(img)
        # instruction (프레임 위 오버레이)
        d.rectangle([0,0,fw,26], fill=(0,0,0)); d.text((8,4), instr, font=f_sm, fill=(255,255,255))

        # 오른쪽 현황 패널
        px = fw + 16; d.text((px, 12), f"t = {i} / {N-1}", font=f_hd, fill=(11,11,11))
        d.text((px, 40), f"LIVE gr00t · robocasa", font=f_sm, fill=(120,120,120))
        d.text((px, 58), tag, font=f_sm, fill=(120,120,120))
        yy = 84
        if has_gt:
            g = gt_ids[i]; gcol = hexrgb(PALETTE[g]) if g>=0 else hexrgb(GREY)
            gname = (phases[i].get("gt") if isinstance(phases[i],dict) else None) or "—"
            d.text((px, yy), "GT", font=f_row, fill=(82,81,78))
            d.rectangle([px+52, yy+4, px+74, yy+22], fill=gcol, outline=(11,11,11))
            d.text((px+84, yy), f"{gname}", font=f_row, fill=(11,11,11))
            yy += 46
        for lab, ids in (("AE", ae_ids), ("SAE", sae_ids)):
            entry = _live(phases[i], lab.lower())
            pid = ids[i]; col = hexrgb(PALETTE[pid]) if pid>=0 else hexrgb(GREY)
            nm = entry.get("phase") or "—"; cl = entry.get("cluster")
            d.text((px, yy), lab, font=f_row, fill=(82,81,78))
            d.rectangle([px+52, yy+4, px+74, yy+22], fill=col, outline=(11,11,11))
            d.text((px+84, yy), f"{nm}", font=f_row, fill=(11,11,11))
            d.text((px+84, yy+24), f"cluster {cl}", font=f_sm, fill=(120,120,120))
            yy += 52

        # 아래 리본 (시간축 phase): GT / AE / SAE
        ry = H - RIBBON_H + 20
        tracks = ([("GT phase (env label)", gt_ids)] if has_gt else []) + \
                 [("AE phase (live)", ae_ids), ("SAE phase (live)", sae_ids)]
        for ti, (lab, ids) in enumerate(tracks):
            ribbon(d, ry + ti*40, ids, lab)
        # 재생 헤드
        hx = x0 + int((i+0.5)/N*(x1-x0))
        d.line([hx, ry-4, hx, ry + len(tracks)*40 - 6], fill=(11,11,11), width=2)
        writer.append_data(np.asarray(img, np.uint8))
    writer.close()
    print(f"rendered {N} frames -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=20)
    a = ap.parse_args()
    render(Path(a.base), Path(a.out), a.fps)
