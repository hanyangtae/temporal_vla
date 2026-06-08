"""VL/DiT LDA 기반 anomaly 분석 + 사분면 분류 + 비디오 목록.

Mahalanobis scalar 대신 PCA→LDA 방향으로 투영한 score를 사용.
이 방향이 cv-AUROC=0.931을 만든 실제 discriminant direction이므로
더 정확하게 성공/실패를 구분하는 축을 따라 anomaly를 측정한다.

출력:
  - 사분면별 scatter plot (LDA-VL vs LDA-DiT)
  - step별 LDA anomaly trajectory (사분면별 평균)
  - 사분면별 ep 개별 trajectory
  - 카테고리별 비디오 경로 목록 (HTML + txt)

instruction 단위 분석:
  --instruction-filter "slide in"   : ep_meta.lang 에 해당 문자열 포함 rollout만 분석
  --auto-split-instruction          : task 내 instruction 변형을 자동 발견 후 각각 실행
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from collections import Counter
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "scripts/safe/groot_n16/robocasa/vis"))
from core.io import load_pkl  # noqa: E402

CAPTURE_LAYERS = [0, 2, 4, 8, 16, 24, 31]
# watch_video.sh 경로 (HTML 명령어 생성용)
_WATCH_SH = "scripts/safe/groot_n16/robocasa/watch_video.sh"


# ── PCA + LDA (numpy/scipy, sklearn 불필요) ────────────────────────────────

def pca_fit(X: np.ndarray, n: int = 30):
    mu = X.mean(axis=0)
    Xc = X - mu
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    n = min(n, len(s))
    return mu, Vt[:n]          # [n, D]

def pca_transform(X, mu, Vt):
    return (X - mu) @ Vt.T    # [N, n]

def lda_direction(Xtr: np.ndarray, ytr: np.ndarray):
    """Binary LDA: 1D discriminant direction (numpy)."""
    mu1 = Xtr[ytr == 1].mean(axis=0)
    mu0 = Xtr[ytr == 0].mean(axis=0)
    diff = mu1 - mu0
    norm = np.linalg.norm(diff)
    if norm < 1e-12:
        return np.zeros_like(diff)
    return diff / norm         # [d]

def fit_lda_pipeline(X_succ: np.ndarray, X_fail: np.ndarray, n_pc: int = 30):
    """성공/실패 feature → PCA → LDA direction 학습."""
    X_all = np.vstack([X_succ, X_fail])
    y_all = np.array([1] * len(X_succ) + [0] * len(X_fail))
    mu, Vt = pca_fit(X_all, n_pc)
    Xp = pca_transform(X_all, mu, Vt)
    direction = lda_direction(Xp, y_all)   # [n_pc]
    # 성공이 높은 방향으로 부호 맞추기
    score_succ = Xp[y_all == 1] @ direction
    score_fail = Xp[y_all == 0] @ direction
    if score_succ.mean() < score_fail.mean():
        direction = -direction
    return mu, Vt, direction   # 투영: (X - mu) @ Vt.T @ direction → scalar

def project(X: np.ndarray, mu, Vt, direction) -> np.ndarray:
    return pca_transform(X, mu, Vt) @ direction


def _instr_tag(lang: str) -> str:
    """instruction 문자열 → 파일명 안전 태그."""
    return re.sub(r"[^a-z0-9]+", "_", lang.lower()).strip("_")


# ── 분석 본체 ──────────────────────────────────────────────────────────────

def run_analysis(args: argparse.Namespace, task_dir: Path, out: Path) -> None:
    """instruction_filter / 전체 데이터 한 번 실행."""
    out.mkdir(parents=True, exist_ok=True)

    local_idx = int(np.argmin([abs(b - args.dit_block) for b in CAPTURE_LAYERS]))
    dit_block_actual = CAPTURE_LAYERS[local_idx]
    T = args.early_t
    MAX = args.max_steps

    instr_label = f" [instr: {args.instruction_filter}]" if args.instruction_filter else ""
    title = task_dir.name + instr_label
    print(f"[{title}] DiT block={dit_block_actual}  early_t={T}  n_pc={args.n_pc}")

    # ── 1. feature 로드 (instruction filter 적용) ──
    records = []
    for p in sorted(task_dir.glob("*.pkl")):
        d = load_pkl(p)
        # instruction filter
        if args.instruction_filter:
            lang = d.get("ep_meta", {}).get("lang", "")
            if args.instruction_filter.lower() not in lang.lower():
                continue
        vl = d.get("vl_hidden_states")
        hs = d.get("hidden_states")
        if vl is None or hs is None:
            continue
        vl_arr  = np.stack([np.asarray(v, dtype=np.float32) for v in vl])
        hs_arr  = np.stack([np.asarray(h, dtype=np.float32) for h in hs])
        dit_arr = hs_arr[:, local_idx, 1:17, :].mean(axis=1)   # valid16 mean
        records.append({
            "mp4": p.with_suffix(".mp4").name,
            "success": int(d.get("episode_success", 0)),
            "length": len(vl),
            "vl": vl_arr,    # [n_steps, Dvl]
            "dit": dit_arr,  # [n_steps, Ddit]
        })

    succ_r = [r for r in records if r["success"] == 1]
    fail_r = [r for r in records if r["success"] == 0]
    print(f"  success={len(succ_r)}  fail={len(fail_r)}")

    if len(succ_r) < 2 or len(fail_r) < 2:
        print(f"  [skip] 샘플 부족 (succ={len(succ_r)}, fail={len(fail_r)})")
        return

    # ── 2. LDA 파이프라인 학습 (early window mean feature) ──
    def early_mean(r, key):
        arr = r[key]
        t = min(T, r["length"])
        return arr[:t].mean(axis=0)

    Xvl_s  = np.stack([early_mean(r, "vl")  for r in succ_r])
    Xvl_f  = np.stack([early_mean(r, "vl")  for r in fail_r])
    Xdit_s = np.stack([early_mean(r, "dit") for r in succ_r])
    Xdit_f = np.stack([early_mean(r, "dit") for r in fail_r])

    vl_mu,  vl_Vt,  vl_dir  = fit_lda_pipeline(Xvl_s,  Xvl_f,  args.n_pc)
    dit_mu, dit_Vt, dit_dir = fit_lda_pipeline(Xdit_s, Xdit_f, args.n_pc)

    succ_vl_scores  = project(Xvl_s,  vl_mu,  vl_Vt,  vl_dir)
    succ_dit_scores = project(Xdit_s, dit_mu, dit_Vt, dit_dir)
    succ_vl_ref  = succ_vl_scores.mean()
    succ_dit_ref = succ_dit_scores.mean()

    def anomaly_score_early(r):
        vl_s  = float(succ_vl_ref  - project(early_mean(r,"vl" ).reshape(1,-1),  vl_mu,  vl_Vt,  vl_dir)[0])
        dit_s = float(succ_dit_ref - project(early_mean(r,"dit").reshape(1,-1), dit_mu, dit_Vt, dit_dir)[0])
        return vl_s, dit_s

    for r in records:
        r["vl_score"], r["dit_score"] = anomaly_score_early(r)

    # ── 3. step별 LDA score 계산 ──
    def step_lda(r, key, mu, Vt, direction, ref):
        arr = r[key]
        scores = []
        for t in range(MAX):
            vec = arr[t] if t < r["length"] else arr[-1]
            s = float(ref - project(vec.reshape(1,-1), mu, Vt, direction)[0])
            scores.append(s)
        return np.array(scores)

    for r in records:
        r["vl_steps"]  = step_lda(r, "vl",  vl_mu,  vl_Vt,  vl_dir,  succ_vl_ref)
        r["dit_steps"] = step_lda(r, "dit", dit_mu, dit_Vt, dit_dir, succ_dit_ref)

    # ── 4. 사분면 분류 (median split on fail) ──
    vl_med  = float(np.median([r["vl_score"]  for r in fail_r]))
    dit_med = float(np.median([r["dit_score"] for r in fail_r]))

    QUAD = {
        "A_both":  ("A: VL+ DiT+  (둘다 혼란)",   "red"),
        "B_goal":  ("B: VL+ DiT-  (goal-type)",  "orange"),
        "C_motor": ("C: VL- DiT+  (motor-type)", "royalblue"),
        "D_none":  ("D: VL- DiT-  (예측불가)",   "gray"),
    }

    def get_quad(r):
        hi_vl  = r["vl_score"]  >= vl_med
        hi_dit = r["dit_score"] >= dit_med
        if hi_vl  and hi_dit:        return "A_both"
        if hi_vl  and not hi_dit:    return "B_goal"
        if not hi_vl and hi_dit:     return "C_motor"
        return "D_none"

    quads = {q: [] for q in QUAD}
    for r in fail_r:
        quads[get_quad(r)].append(r)

    print(f"\n  VL median={vl_med:.3f}  DiT median={dit_med:.3f}")
    for qk, (ql, _) in QUAD.items():
        print(f"  {ql}: n={len(quads[qk])}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(MAX)

    # ── 그림 1: scatter (LDA-VL vs LDA-DiT) ──
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter([r["vl_score"] for r in succ_r],
               [r["dit_score"] for r in succ_r],
               c="green", alpha=0.4, s=35, label="success", marker="o")
    for qk, (ql, qc) in QUAD.items():
        eps = quads[qk]
        if eps:
            ax.scatter([r["vl_score"] for r in eps],
                       [r["dit_score"] for r in eps],
                       c=qc, alpha=0.8, s=55, label=f"{ql} (n={len(eps)})", marker="x")
    ax.axvline(vl_med,  color="orange",    linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axhline(dit_med, color="royalblue", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xlabel(f"VL LDA score (failure-like ↑, early t={T})")
    ax.set_ylabel(f"DiT-b{dit_block_actual} LDA score (failure-like ↑, early t={T})")
    ax.set_title(f"{title} — LDA-based failure quadrant")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "lda_scatter.png", dpi=130)
    plt.close()

    # ── 그림 2: step별 trajectory (VL / DiT 나란히) ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, ylabel in [
        (axes[0], "vl_steps",  "VL LDA anomaly"),
        (axes[1], "dit_steps", f"DiT-b{dit_block_actual} LDA anomaly"),
    ]:
        succ_traj = np.stack([r[key] for r in succ_r]).mean(axis=0)
        ax.plot(steps, succ_traj, color="green", linewidth=2.5, label="success (mean)")
        for qk, (ql, qc) in QUAD.items():
            eps = quads[qk]
            if eps:
                traj = np.stack([r[key] for r in eps]).mean(axis=0)
                ax.plot(steps, traj, color=qc, linewidth=2, label=f"{ql} (n={len(eps)})")
        ax.axvline(T, color="black", linestyle=":", linewidth=1)
        ax.set_xlabel("inference step"); ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "lda_trajectory.png", dpi=130)
    plt.close()

    # ── 그림 3: 사분면별 개별 ep (대표 4개씩) ──
    n_rep = 4
    fig, axes = plt.subplots(len(QUAD), 2, figsize=(12, 4 * len(QUAD)))
    for qi, (qk, (ql, qc)) in enumerate(QUAD.items()):
        recs = sorted(quads[qk], key=lambda r: r["vl_score"], reverse=True)[:n_rep]
        for ax, key, title_sfx in [(axes[qi, 0], "vl_steps", "VL"),
                                    (axes[qi, 1], "dit_steps", f"DiT-b{dit_block_actual}")]:
            succ_traj = np.stack([r[key] for r in succ_r]).mean(axis=0)
            ax.fill_between(steps, succ_traj.min(), succ_traj,
                            alpha=0.12, color="green")
            ax.plot(steps, succ_traj, color="green", linewidth=1.5,
                    linestyle="--", alpha=0.7, label="success mean")
            for r in recs:
                ep_label = r["mp4"].replace("succ0.mp4","").rstrip("-")
                ax.plot(steps, r[key], linewidth=1.3, alpha=0.85, label=ep_label)
            ax.axvline(T, color="black", linestyle=":", linewidth=0.8)
            ax.set_title(f"{ql}  [{title_sfx}]", fontsize=9)
            ax.set_xlabel("inference step")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "lda_per_quad.png", dpi=130)
    plt.close()

    print(f"\n[plots] {out}/")

    # ── 5. 비디오 목록 (카테고리별) ──
    txt_lines = [f"# {title} — failure 비디오 카테고리별 목록",
                 f"# (LDA early_t={T}, DiT block={dit_block_actual})", ""]
    for qk, (ql, _) in QUAD.items():
        eps = sorted(quads[qk], key=lambda r: r["vl_score"], reverse=True)
        txt_lines += [f"\n## {ql}  (n={len(eps)})", "# VL_score  DiT_score  파일명"]
        for r in eps:
            txt_lines.append(f"  {r['vl_score']:+.3f}  {r['dit_score']:+.3f}  {r['mp4']}")
    (out / "video_list.txt").write_text("\n".join(txt_lines))

    # HTML (watch_video.sh 명령어 포함)
    QCOLORS = {"A_both":"#f88","B_goal":"#fc8","C_motor":"#88f","D_none":"#aaa"}
    html_lines = [
        "<html><head><meta charset=\"utf-8\"><style>",
        "body{font-family:monospace;background:#1a1a1a;color:#ddd;padding:20px;line-height:1.7}",
        "h1{color:#fff} h2{color:#aef;margin-top:24px}",
        ".ep{margin:3px 0;padding:6px 10px;background:#2a2a2a;border-radius:4px}",
        ".score{color:#888;font-size:.85em;margin-right:10px}",
        ".cmd{display:block;background:#333;padding:3px 8px;border-radius:3px;font-size:.82em;color:#8df;margin-top:3px}",
        "</style></head><body>",
        f"<h1>{title}</h1>",
        f"<p style=\"color:#888\">LDA early_t={T} / DiT block={dit_block_actual}</p>",
        "<p style=\"color:#fc8\">아래 명령어를 터미널(또는 ! 커맨드)에 복붙해서 재생하세요.</p>",
    ]
    for qk, (ql, _) in QUAD.items():
        eps = sorted(quads[qk], key=lambda r: r["vl_score"], reverse=True)
        html_lines.append(
            f"<h2 style=\"color:{QCOLORS[qk]}\">{ql} &nbsp; (n={len(eps)})</h2>"
        )
        for r in eps:
            cmd = f"bash {_WATCH_SH} {r['mp4']} {task_dir.name}"
            html_lines.append(
                f"<div class=\"ep\">"
                f"<span class=\"score\">VL={r['vl_score']:+.3f}  DiT={r['dit_score']:+.3f}</span> {r['mp4']}"
                f"<code class=\"cmd\">{cmd}</code>"
                f"</div>"
            )
    # 성공 ep 목록 조회 명령어
    html_lines += [
        "<h2 style=\"color:#8f8\">Success 영상 (비교용)</h2>",
        "<p style=\"color:#888\">성공 영상 목록 조회:</p>",
        "<div class=\"ep\"><code class=\"cmd\">"
        f"ssh -p 11112 kimseungjun@166.104.146.37 "
        f"'ls /home/kimseungjun/workspace/temporal_vla/outputs/eval/robocasa/groot_n16/"
        f"target_atomic_moderate10_pathway_pertoken_100ep/raw_rollouts/{task_dir.name}/*succ1.mp4 | head -10'"
        "</code></div>",
        "</body></html>",
    ]
    (out / "video_list.html").write_text("\n".join(html_lines))

    print(f"[video list] {out}/video_list.txt")
    print(f"[video html] {out}/video_list.html")

    # JSON
    (out / "lda_summary.json").write_text(json.dumps({
        "task": task_dir.name,
        "instruction_filter": args.instruction_filter,
        "early_t": T, "dit_block": dit_block_actual,
        "vl_median": vl_med, "dit_median": dit_med,
        "n_success": len(succ_r), "n_failure": len(fail_r),
        "quadrant_counts": {q: len(v) for q, v in quads.items()},
        "quadrants": {q: [{"mp4": r["mp4"], "vl": round(r["vl_score"], 3),
                           "dit": round(r["dit_score"], 3)} for r in v]
                     for q, v in quads.items()},
    }, indent=2))
    print(f"[done] {out}/")


# ── 메인 ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--early-t", type=int, default=8,
                    help="LDA 학습 + 사분면 분류에 쓸 early window (inference steps)")
    ap.add_argument("--dit-block", type=int, default=31)
    ap.add_argument("--max-steps", type=int, default=45)
    ap.add_argument("--n-pc", type=int, default=30)
    ap.add_argument(
        "--instruction-filter",
        default=None,
        metavar="SUBSTR",
        help="ep_meta.lang 에 해당 문자열이 포함된 rollout 만 분석 (대소문자 무시). "
             "예: 'slide in', 'right', 'left'.",
    )
    ap.add_argument(
        "--auto-split-instruction",
        action="store_true",
        help="task 내 모든 instruction 변형을 자동 발견 후 --min-instruction-rollouts "
             "이상인 것만 각각 별도 output 디렉토리에 분석.",
    )
    ap.add_argument(
        "--min-instruction-rollouts",
        type=int,
        default=5,
        help="--auto-split-instruction 에서 분석 대상 최소 rollout 수 (기본 5).",
    )
    args = ap.parse_args()

    task_dir = Path(args.task_dir)
    default_out = task_dir.parent.parent / "analysis" / "lda_quadrant" / task_dir.name

    if args.auto_split_instruction:
        if args.instruction_filter:
            ap.error("--auto-split-instruction 과 --instruction-filter 는 동시에 사용 불가.")
        # pkl 헤더만 스캔해서 instruction 분포 파악
        lang_counts: Counter[str] = Counter()
        for p in sorted(task_dir.glob("*.pkl")):
            d = load_pkl(p)
            lang_counts[d.get("ep_meta", {}).get("lang", "")] += 1
        print(f"[auto-split] {task_dir.name}: {len(lang_counts)} instruction variants")
        for lang, cnt in lang_counts.most_common():
            if cnt < args.min_instruction_rollouts or not lang:
                print(f"  [skip] '{lang}' n={cnt}")
                continue
            tag = _instr_tag(lang)
            base = Path(args.out) if args.out else default_out
            sub_out = base.parent / (base.name + f"__instr_{tag}")
            print(f"  [run]  '{lang}' n={cnt} -> {sub_out.name}")
            args_copy = copy.copy(args)
            args_copy.instruction_filter = lang
            args_copy.auto_split_instruction = False
            args_copy.out = str(sub_out)
            run_analysis(args_copy, task_dir, sub_out)
        return

    # 단일 실행 (instruction_filter 있을 수도 없을 수도)
    if args.instruction_filter and not args.out:
        tag = _instr_tag(args.instruction_filter)
        out = default_out.parent / (default_out.name + f"__instr_{tag}")
    else:
        out = Path(args.out) if args.out else default_out
    run_analysis(args, task_dir, out)


if __name__ == "__main__":
    main()
