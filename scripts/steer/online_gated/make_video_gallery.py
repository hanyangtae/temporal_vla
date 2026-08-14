#!/usr/bin/env python3
"""replay eval 영상 갤러리 생성 — instruction별 페이지 (셀 행 × arm 열).

per_episode.tsv(collect_results 후처리본)의 quadrant/collection_success/trigger 를
셀 주석으로 달고, 수집 대비 뒤집힘(구제=초록/파손=빨강)을 하이라이트한다.
비디오 경로는 갤러리 html 위치 기준 상대경로 — http.server 를 갤러리 상위에서 띄울 것.
"""
from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

STYLE = """
body{font-family:sans-serif;background:#111;color:#ddd;margin:16px}
table{border-collapse:collapse} td,th{border:1px solid #333;padding:4px;vertical-align:top}
th{background:#222;position:sticky;top:0}
video{width:240px;display:block}
.cell-head{background:#1a1a1a;font-size:12px;white-space:nowrap}
.rescue{outline:3px solid #2e7} .break{outline:3px solid #e33}
.succ{color:#2e7} .fail{color:#e55}
.meta{font-size:11px;color:#999}
a{color:#8cf}
"""


def load_rows(job_dir: Path) -> dict[int, dict]:
    pe = job_dir / "per_episode.tsv"
    if not pe.exists():
        return {}
    out = {}
    for r in csv.DictReader(pe.open(), delimiter="\t"):
        out[int(r["ep"])] = r
    return out


def find_video(job_dir: Path, ep: int) -> Path | None:
    hits = list(job_dir.glob(f"raw_rollouts/*/*/task0--ep{ep}--succ*.mp4"))
    return hits[0] if hits else None


def build_page(title: str, root: Path, slug: str, arms: list[str], out_dir: Path) -> str:
    per_arm = {a: load_rows(root / slug / a) for a in arms}
    eps = sorted({e for rows in per_arm.values() for e in rows})
    body = [f"<h1>{html.escape(title)}</h1>",
            "<p class=meta>행=셀(scene,noise) · 열=arm · 테두리: <span class=succ>초록=구제"
            "(수집실패→성공)</span> / <span class=fail>빨강=파손(수집성공→실패)</span> · "
            "trig=발화 record(phase)</p>",
            "<table><tr><th>cell</th>" + "".join(f"<th>{a}</th>" for a in arms) + "</tr>"]
    for ep in eps:
        ref = next((per_arm[a][ep] for a in arms if ep in per_arm[a]), None)
        if ref is None:
            continue
        csucc = ref.get("collection_success", "")
        head = (f"s{ref['scene_idx']} n{ref.get('noise_idx','?')}<br>"
                f"{ref.get('quadrant','')}<br>수집: "
                f"<span class={'succ' if csucc == '1' else 'fail'}>"
                f"{'성공' if csucc == '1' else '실패'}</span>")
        tds = [f"<td class=cell-head>{head}</td>"]
        for a in arms:
            r = per_arm[a].get(ep)
            if r is None:
                tds.append("<td>—</td>")
                continue
            v = find_video(root / slug / a, ep)
            succ = r["success"] == "1"
            flip = ""
            if csucc == "0" and succ:
                flip = "rescue"
            elif csucc == "1" and not succ:
                flip = "break"
            trig = r.get("trigger_step", "")
            trig_txt = (f" trig={trig}({r.get('phase_at_trigger','')})"
                        if trig not in ("", "NA", "None") else "")
            label = (f"<span class={'succ' if succ else 'fail'}>"
                     f"{'성공' if succ else '실패'}</span>"
                     f"<span class=meta>{trig_txt} steps={r.get('steps','')}</span>")
            if v is None:
                tds.append(f"<td>{label}<br>(영상 없음)</td>")
            else:
                rel = Path(v).resolve().relative_to(out_dir.resolve().parent)
                rel = Path("..") / rel
                tds.append(f"<td class='{flip}'><video src='{rel}' controls "
                           f"preload=none></video>{label}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    body.append("</table>")
    return f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title><style>{STYLE}</style>" + "\n".join(body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", action="append", required=True,
                    help="title:root:slug:arm1,arm2,... (반복 지정)")
    ap.add_argument("--out", type=Path, required=True, help="갤러리 출력 디렉토리")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    links = []
    for spec in args.spec:
        title, root, slug, arms = spec.split(":")
        page = build_page(title, Path(root), slug, arms.split(","), args.out)
        fn = title.replace("/", "_").replace(" ", "_") + ".html"
        (args.out / fn).write_text(page, encoding="utf-8")
        links.append(f"<li><a href='{fn}'>{html.escape(title)}</a></li>")
        print(f"[gallery] {args.out/fn}")
    (args.out / "index.html").write_text(
        f"<!doctype html><meta charset=utf-8><title>eval 영상 갤러리</title>"
        f"<style>{STYLE}</style><h1>replay eval 영상</h1><ul>{''.join(links)}</ul>",
        encoding="utf-8")
    print(f"[gallery] {args.out/'index.html'}")


if __name__ == "__main__":
    main()
