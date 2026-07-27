"""실제 http_feature_collect 산출물(pkl+mp4) → 참고영상 스타일 3트랙 영상.

- 프레임: http VideoRecordingWrapper mp4 (3뷰 + 상단 캡션)
- GT: pkl feature_phases (record 단위, proximity 13-class 이벤트 라벨)
- AE/SAE: pkl hidden_states[record] ([denoise,token,dim]) → 마지막 denoise / 전체 토큰 mean
          → OnlinePhaseClassifier (PCA→encode→kmeans) → phase  (라이브 실행분의 활성값)
- 정렬: frame f ↔ env step (steps_per_render*f) ↔ record floor(env_step / n_action_steps)
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO))
from src.phase_online import OnlinePhaseClassifier

PALETTE = ["#2a78d6","#1baf7a","#eda100","#008300","#4a3aa7","#e87ba4","#e34948",
           "#17a2b8","#8e6fd0","#8c564b","#2bb5a0","#b5179e","#7f8c00"]
NAMES = ["reach-to-object","grasp","transport","place","insert-settle","terminal",
         "wrong-grasp","reach-to-handle","grasp-handle","pull","open-done","disengage","push-back"]
NAME2ID = {n:i for i,n in enumerate(NAMES)}
GREY = "#c9c8c4"
def hexrgb(h): return tuple(int(h[i:i+2],16) for i in (1,3,5))
def font(sz,b=False):
    try: return ImageFont.truetype(("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if b
                                     else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), sz)
    except Exception: return ImageFont.load_default()
def pid(name): return NAME2ID.get(name, -1)


def reduce_record(hs_r):
    """hidden_states[record] → clf 입력 [1536]. shape [denoise,token,dim] 또는 [layer,denoise,token,dim]."""
    a = np.asarray(hs_r, dtype=np.float32)
    if a.ndim == 4:      # [L,K,T,D] (다층 캡처) → layer 0 가정(단일 layer12 캡처)
        a = a[0]
    # a: [K, T, D] → 마지막 denoise, 전체 토큰 mean
    return a[-1].mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--mp4", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=str(_REPO / "task_classification"))
    ap.add_argument("--fps", type=int, default=20)
    a = ap.parse_args()

    root = Path(a.root)
    clfs = {k: OnlinePhaseClassifier.from_run(
                run_dir=root / f"runs/{v}",
                pca_path=root / "datasets_local/phase_cls_pq3/derived/L12-D3-pca64w/pca.npz",
                map_path=root / "runs/cluster_phase_map.json", device="cpu")
            for k, v in {"ae":"ae-log_likelihood-s0","sae":"sae-log_likelihood-s0"}.items()}

    m = pickle.load(open(a.pkl, "rb"))
    hs = m["hidden_states"]; gt_rec = m.get("feature_phases", [])
    nrec = len(hs); n_as = m.get("n_action_steps", 5); spr = m.get("steps_per_render", 2)
    # record 별 AE/SAE
    H = np.stack([reduce_record(hs[r]) for r in range(nrec)])   # [nrec, 1536]
    ae = clfs["ae"].infer(H); sae = clfs["sae"].infer(H)
    if isinstance(ae, dict): ae=[ae]; sae=[sae]

    _rd = imageio.get_reader(a.mp4)
    try: N = _rd.count_frames()
    except Exception: N = m.get("n_records", nrec) * n_as // spr
    frames = [np.asarray(_rd.get_data(i)) for i in range(N)]
    N = len(frames)
    def rec_of(f):   # frame → record
        return min(int((spr*f)//n_as), nrec-1)

    fh, fw = frames[0].shape[:2]
    PANEL_W = 300; RIBBON_H = 150
    W = fw + PANEL_W; Hh = max(fh, 190) + RIBBON_H
    f_hd,f_row,f_sm = font(22,True), font(20,True), font(14)
    x0,x1 = 12, fw-12
    tag = "SUCCESS" if m.get("episode_success") else "FAIL"

    def ribbon(d,y,ids,label):
        d.text((12,y-18),label,font=f_sm,fill=(60,60,60))
        for f in range(N):
            xa=x0+int(f/N*(x1-x0)); xb=x0+int((f+1)/N*(x1-x0))
            p=ids[f]; col=hexrgb(PALETTE[p]) if p>=0 else hexrgb(GREY)
            d.rectangle([xa,y,max(xa,xb-1),y+18],fill=col)

    gt_ids=[pid(gt_rec[rec_of(f)]) if rec_of(f)<len(gt_rec) else -1 for f in range(N)]
    ae_ids=[pid(ae[rec_of(f)].get("phase")) for f in range(N)]
    sae_ids=[pid(sae[rec_of(f)].get("phase")) for f in range(N)]

    w = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8, macro_block_size=None)
    for f in range(N):
        r=rec_of(f)
        img=Image.new("RGB",(W,Hh),(252,252,251)); img.paste(Image.fromarray(frames[f]),(0,0))
        d=ImageDraw.Draw(img)
        px=fw+16; d.text((px,12),f"t = {f} / {N-1}",font=f_hd,fill=(11,11,11))
        d.text((px,40),"LIVE gr00t · robocasa · execute-5",font=f_sm,fill=(120,120,120))
        d.text((px,58),tag,font=f_row,fill=((0,131,0) if tag=="SUCCESS" else (200,40,40)))
        yy=92
        rows=[("GT",gt_rec[r] if r<len(gt_rec) else None,None,gt_ids[f]),
              ("AE",ae[r].get("phase"),ae[r].get("cluster"),ae_ids[f]),
              ("SAE",sae[r].get("phase"),sae[r].get("cluster"),sae_ids[f])]
        for lab,nm,cl,pi in rows:
            col=hexrgb(PALETTE[pi]) if pi>=0 else hexrgb(GREY)
            d.text((px,yy),lab,font=f_row,fill=(82,81,78))
            d.rectangle([px+52,yy+4,px+74,yy+22],fill=col,outline=(11,11,11))
            d.text((px+84,yy),f"{nm or '—'}",font=f_row,fill=(11,11,11))
            if cl is not None: d.text((px+84,yy+24),f"cluster {cl}",font=f_sm,fill=(120,120,120))
            yy+=52
        ry=Hh-RIBBON_H+20
        for ti,(lab,ids) in enumerate([("GT phase (env label)",gt_ids),
                                       ("AE phase (live)",ae_ids),("SAE phase (live)",sae_ids)]):
            ribbon(d,ry+ti*40,ids,lab)
        hx=x0+int((f+0.5)/N*(x1-x0)); d.line([hx,ry-4,hx,ry+3*40-6],fill=(11,11,11),width=2)
        w.append_data(np.asarray(img,np.uint8))
    w.close()
    print(f"rendered {N} frames ({tag}) -> {a.out}")

if __name__ == "__main__":
    main()
