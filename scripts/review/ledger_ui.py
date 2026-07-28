#!/usr/bin/env python3
"""레포 검토 판정 UI — 브라우저에서 keep/merge/archive 판정을 LEDGER.tsv 에 기록한다.

설계: docs/superpowers/specs/2026-07-28-repo-review-design.md

입력: docs/review/S*_files.tsv  (스테이지 카드의 기계 판독 부분)
출력: docs/review/LEDGER.tsv    (판정 원장, 클릭 즉시 기록)

사용:
    python3 scripts/review/ledger_ui.py            # http://127.0.0.1:8777
    python3 scripts/review/ledger_ui.py --port 9000

소스 코드 뷰어는 없다. 소스는 별도 에디터에서 보는 것을 전제로 한다.
"""
from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REVIEW = REPO / "docs" / "review"
LEDGER = REVIEW / "LEDGER.tsv"

LEDGER_COLS = ["파일", "스테이지", "판정", "사유", "적용여부", "삭제커밋"]
FILES_COLS = ["파일", "스테이지", "LOC", "최종수정", "역할", "플래그", "비고"]
VERDICTS = ["keep", "merge", "archive", "미정"]


def read_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    rows = []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        cells = ln.split("\t")
        cells += [""] * (len(header) - len(cells))
        rows.append(dict(zip(header, cells)))
    return rows


def write_tsv(path: Path, cols: list[str], rows: list[dict]) -> None:
    """원자적 교체 — 부분 기록 상태로 남지 않게 한다."""
    body = ["\t".join(cols)]
    for r in rows:
        body.append("\t".join((r.get(c, "") or "").replace("\t", " ") for c in cols))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(body) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_files() -> list[dict]:
    """모든 스테이지의 S*_files.tsv 를 스테이지 번호순으로 합친다."""
    out = []
    for p in sorted(REVIEW.glob("S*_files.tsv"), key=_stage_key):
        out.extend(read_tsv(p))
    return out


def _stage_key(p: Path):
    m = re.match(r"S(\d+)", p.name)
    return (int(m.group(1)) if m else 999, p.name)


def build_state() -> dict:
    files = load_files()
    ledger = {r["파일"]: r for r in read_tsv(LEDGER)}
    items = []
    for f in files:
        j = ledger.get(f["파일"], {})
        items.append({**f, "판정": j.get("판정", ""), "사유": j.get("사유", ""),
                      "적용여부": j.get("적용여부", "")})
    return {"items": items, "verdicts": VERDICTS}


def save_judgement(path: str, verdict: str, reason: str) -> None:
    rows = read_tsv(LEDGER)
    by_path = {r["파일"]: r for r in rows}
    files = {f["파일"]: f for f in load_files()}
    stage = files.get(path, {}).get("스테이지", "")
    if path in by_path:
        by_path[path].update({"판정": verdict, "사유": reason, "스테이지": stage})
    else:
        rows.append({"파일": path, "스테이지": stage, "판정": verdict,
                     "사유": reason, "적용여부": "", "삭제커밋": ""})
    write_tsv(LEDGER, LEDGER_COLS, rows)


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>레포 검토 판정</title>
<style>
:root{--bg:#fff;--fg:#111;--mut:#666;--line:#ddd;--card:#fafafa;--accent:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#e8e8e8;--mut:#999;--line:#333;--card:#1d1d1d;--accent:#60a5fa}}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 ui-sans-serif,system-ui,"Noto Sans KR",sans-serif;background:var(--bg);color:var(--fg)}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;z-index:5}
h1{font-size:15px;margin:0 0 6px}
#bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden}
#bar>i{display:block;height:100%;background:var(--accent);width:0}
#meta{color:var(--mut);font-size:12px;margin-top:5px}
#filters{margin-top:8px;display:flex;gap:6px;flex-wrap:wrap}
#filters button{font-size:12px;padding:3px 9px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:12px;cursor:pointer}
#filters button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
main{padding:12px 16px;max-width:1100px}
.row{border:1px solid var(--line);border-radius:8px;margin-bottom:8px;background:var(--card);padding:10px 12px}
.row.sel{border-color:var(--accent);box-shadow:0 0 0 2px color-mix(in srgb,var(--accent) 25%,transparent)}
.row.done{opacity:.55}
.p{font-family:ui-monospace,monospace;font-size:13px;word-break:break-all}
.facts{color:var(--mut);font-size:12px;margin-top:3px}
.flag{display:inline-block;font-size:11px;padding:1px 6px;border-radius:9px;background:#b45309;color:#fff;margin-right:4px}
.acts{margin-top:7px;display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.acts button{font-size:12px;padding:3px 10px;border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:5px;cursor:pointer}
.acts button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.acts input{flex:1;min-width:180px;font-size:12px;padding:3px 7px;border:1px solid var(--line);border-radius:5px;background:var(--bg);color:var(--fg)}
kbd{font:11px ui-monospace,monospace;border:1px solid var(--line);border-radius:3px;padding:0 4px;color:var(--mut)}
.tblwrap{overflow-x:auto}
</style></head><body>
<header>
  <h1>레포 검토 판정 <span id="stage" style="color:var(--mut);font-weight:400"></span></h1>
  <div id="bar"><i></i></div>
  <div id="meta"></div>
  <div id="filters"></div>
  <div id="meta2" style="color:var(--mut);font-size:11px;margin-top:6px">
    <kbd>j</kbd>/<kbd>k</kbd> 이동 · <kbd>1</kbd> keep <kbd>2</kbd> merge <kbd>3</kbd> archive <kbd>4</kbd> 미정 · 판정 즉시 LEDGER.tsv 기록
  </div>
</header>
<main><div class="tblwrap" id="list"></div></main>
<script>
let S={items:[],verdicts:[]}, cur=0, filt='all';
const $=s=>document.querySelector(s);

async function load(){ S=await (await fetch('/api/data')).json(); render(); }

function shown(){
  return S.items.filter(it=>{
    if(filt==='all') return true;
    if(filt==='todo') return !it['판정'];
    if(filt==='done') return !!it['판정'];
    return it['스테이지']===filt;
  });
}

function render(){
  const items=shown();
  const done=S.items.filter(i=>i['판정']).length, tot=S.items.length;
  $('#bar>i').style.width=(tot?done/tot*100:0)+'%';
  $('#meta').textContent=`판정 ${done} / ${tot}  ·  남은 ${tot-done}건  ·  화면 ${items.length}건`;

  const stages=[...new Set(S.items.map(i=>i['스테이지']))].filter(Boolean);
  $('#filters').innerHTML=['all','todo','done',...stages].map(f=>
    `<button data-f="${f}" class="${f===filt?'on':''}">${{all:'전체',todo:'미판정',done:'판정완료'}[f]||f}</button>`).join('');
  $('#filters').onclick=e=>{const b=e.target.closest('button'); if(!b)return; filt=b.dataset.f; cur=0; render();};

  if(cur>=items.length) cur=Math.max(0,items.length-1);
  $('#list').innerHTML=items.map((it,i)=>{
    const flags=(it['플래그']||'').split(/[,;]/).filter(Boolean)
      .map(f=>`<span class="flag">${esc(f.trim())}</span>`).join('');
    const btns=S.verdicts.map(v=>
      `<button data-v="${v}" class="${it['판정']===v?'on':''}">${v}</button>`).join('');
    return `<div class="row ${i===cur?'sel':''} ${it['판정']?'done':''}" data-i="${i}">
      <div class="p">${esc(it['파일'])}</div>
      <div class="facts">${flags}${esc(it['스테이지'])} · ${esc(it['LOC'])}줄 · ${esc(it['최종수정'])} · ${esc(it['역할']||'')}
        ${it['비고']?' · '+esc(it['비고']):''}</div>
      <div class="acts">${btns}
        <input placeholder="사유 / merge 대상" value="${esc(it['사유']||'')}"></div>
    </div>`;}).join('') || '<p style="color:var(--mut)">해당 항목 없음</p>';

  $('#list').onclick=e=>{
    const row=e.target.closest('.row'); if(!row) return;
    cur=+row.dataset.i;
    const b=e.target.closest('button');
    if(b) judge(shown()[cur], b.dataset.v, row.querySelector('input').value);
    else render();
  };
  $('#list').onchange=e=>{
    const row=e.target.closest('.row'); if(!row||e.target.tagName!=='INPUT') return;
    const it=shown()[+row.dataset.i];
    if(it['판정']) judge(it, it['판정'], e.target.value);
  };
  const sel=$('.row.sel'); if(sel) sel.scrollIntoView({block:'nearest'});
}

async function judge(it,v,reason){
  await fetch('/api/judge',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({파일:it['파일'],판정:v,사유:reason||''})});
  it['판정']=v; it['사유']=reason||'';
  render();
}

document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT') return;
  const items=shown(); if(!items.length) return;
  if(e.key==='j'){cur=Math.min(cur+1,items.length-1);render();}
  else if(e.key==='k'){cur=Math.max(cur-1,0);render();}
  else if('1234'.includes(e.key)){
    const v=S.verdicts[+e.key-1];
    const inp=document.querySelectorAll('.row')[cur]?.querySelector('input');
    judge(items[cur], v, inp?inp.value:'');
  }
});
function esc(s){return String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        raw = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.startswith("/api/data"):
            self._send(200, json.dumps(build_state(), ensure_ascii=False), "application/json; charset=utf-8")
        elif self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if not self.path.startswith("/api/judge"):
            return self._send(404, "not found", "text/plain; charset=utf-8")
        n = int(self.headers.get("Content-Length", 0))
        d = json.loads(self.rfile.read(n) or b"{}")
        verdict = d.get("판정", "")
        if verdict not in VERDICTS:
            return self._send(400, json.dumps({"error": "bad verdict"}), "application/json; charset=utf-8")
        save_judgement(d.get("파일", ""), verdict, d.get("사유", ""))
        self._send(200, json.dumps({"ok": True}, ensure_ascii=False), "application/json; charset=utf-8")

    def log_message(self, format, *args):  # noqa: A002 - 콘솔 조용히
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8777)
    args = ap.parse_args()
    cards = sorted(REVIEW.glob("S*_files.tsv"))
    if not cards:
        print(f"경고: {REVIEW}/S*_files.tsv 가 없다. 스테이지 카드를 먼저 만들어야 한다.")
    else:
        print(f"카드 {len(cards)}개: {', '.join(p.name for p in cards)}")
    print(f"원장: {LEDGER}")
    print(f"열기: http://127.0.0.1:{args.port}   (Ctrl-C 종료)")
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
