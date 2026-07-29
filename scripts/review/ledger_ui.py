#!/usr/bin/env python3
"""레포 검토 판정 UI — 브라우저에서 keep/merge/archive 판정을 LEDGER.tsv 에 기록한다.

설계: docs/superpowers/specs/2026-07-28-repo-review-design.md

입력: docs/review/S*_files.tsv  (스테이지 카드의 기계 판독 부분)
출력: docs/review/LEDGER.tsv    (판정 원장, 클릭 즉시 기록)

사용:
    python3 scripts/review/ledger_ui.py            # 0.0.0.0:8777, 토큰 자동 생성
    python3 scripts/review/ledger_ui.py --port 9000
    python3 scripts/review/ledger_ui.py --host 127.0.0.1 --no-token   # 로컬 전용

외부 IP 접근을 전제로 하므로 기본 바인드는 0.0.0.0 이고, 쓰기 API 보호를 위해
토큰을 자동 생성한다. 기동 시 출력되는 URL(`?t=...`)로 접속해야 한다.

소스 코드 뷰어는 없다. 소스는 별도 에디터에서 보는 것을 전제로 한다.
"""
from __future__ import annotations

import argparse
import json
import re
import secrets
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[2]
REVIEW = REPO / "docs" / "review"
LEDGER = REVIEW / "LEDGER.tsv"

LEDGER_COLS = ["파일", "스테이지", "판정", "사유", "적용여부", "삭제커밋"]
FILES_COLS = ["파일", "스테이지", "LOC", "최종수정", "역할", "플래그", "비고"]
VERDICTS = ["keep", "수정", "merge", "archive", "미정"]

# 사유칸에 무엇을 적어야 하는지 — UI 범례로 그대로 노출된다.
VERDICT_HELP = {
    "keep": "이대로 둔다. 사유 비워도 됨",
    "수정": "파일은 남기고 내용을 고친다. 사유칸에 <b>무엇을 고칠지</b>",
    "merge": "다른 파일로 흡수시키고 이 파일은 없앤다. 사유칸에 <b>대상 파일 경로</b>",
    "archive": "필요 없다. git rm (이력에는 남음). 사유칸에 <b>왜 불필요한지</b>",
    "미정": "지금은 판단 불가. 사유칸에 <b>무엇을 알아야 정할 수 있는지</b>",
}


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
    return {"items": items, "verdicts": VERDICTS, "help": VERDICT_HELP}


def save_judgement(path: str, verdict: str, reason: str) -> None:
    """판정 기록. verdict 가 빈 문자열이면 해당 행을 원장에서 제거한다(판정 취소)."""
    rows = read_tsv(LEDGER)
    if not verdict:
        rows = [r for r in rows if r["파일"] != path]
        write_tsv(LEDGER, LEDGER_COLS, rows)
        return
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
#legend{margin-top:8px;font-size:11px;color:var(--mut);display:flex;gap:14px;flex-wrap:wrap}
#legend b{color:var(--fg);font-weight:600}
#legend em{font-style:normal;font-weight:700;color:var(--fg)}
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
  <div id="legend"></div>
  <div id="meta2" style="color:var(--mut);font-size:11px;margin-top:6px">
    <kbd>j</kbd>/<kbd>k</kbd> 이동 · <kbd>1</kbd>~<kbd>5</kbd> 판정 · <kbd>Enter</kbd> 사유칸 · <b>같은 판정 재클릭 = 취소</b> ·
    사유는 타이핑 중 자동 저장 · 저장 버튼 없음
  </div>
</header>
<main><div class="tblwrap" id="list"></div></main>
<script>
let S={items:[],verdicts:[],help:{}}, cur=0, filt='all', VIEW=[], T=null;
const $=s=>document.querySelector(s);
const TOK=new URLSearchParams(location.search).get('t')||'';
const H=TOK?{'X-Review-Token':TOK}:{};

async function load(){
  const r=await fetch('/api/data',{headers:H});
  if(!r.ok){document.body.innerHTML='<p style="padding:20px">접근 거부 — 기동 시 출력된 URL(?t=...)로 접속할 것.</p>';return;}
  S=await r.json(); render();
}

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

  $('#legend').innerHTML=S.verdicts.map((v,i)=>
    `<span><em>${i+1} ${v}</em> — ${S.help[v]||''}</span>`).join('');

  const stages=[...new Set(S.items.map(i=>i['스테이지']))].filter(Boolean);
  $('#filters').innerHTML=['all','todo','done',...stages].map(f=>
    `<button data-f="${f}" class="${f===filt?'on':''}">${{all:'전체',todo:'미판정',done:'판정완료'}[f]||f}</button>`).join('');
  $('#filters').onclick=e=>{const b=e.target.closest('button'); if(!b)return; filt=b.dataset.f; cur=0; render();};

  if(cur>=items.length) cur=Math.max(0,items.length-1);
  VIEW=items;
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
    const i=+row.dataset.i;
    const b=e.target.closest('button');
    if(b){
      const it=VIEW[i];
      // 같은 판정을 다시 누르면 취소
      const v = (it['판정']===b.dataset.v) ? '' : b.dataset.v;
      judge(i, v, row.querySelector('input').value);
    } else { setCur(i); }
  };
  // 사유는 타이핑 중 디바운스 저장 + 포커스 이탈 시 확정 저장.
  // addEventListener 대신 프로퍼티 대입 — render() 마다 리스너가 쌓이는 것을 막는다.
  $('#list').oninput=e=>{
    if(e.target.tagName!=='INPUT') return;
    const i=+e.target.closest('.row').dataset.i, v=e.target.value;
    clearTimeout(T); T=setTimeout(()=>saveReason(i, v), 500);
  };
  $('#list').onfocusout=e=>{
    if(e.target.tagName!=='INPUT') return;
    clearTimeout(T);
    saveReason(+e.target.closest('.row').dataset.i, e.target.value);
  };
  const sel=$('.row.sel'); if(sel) sel.scrollIntoView({block:'nearest'});
}

// 목록을 다시 그리지 않고 해당 행만 갱신한다 — innerHTML 재작성은 입력칸 포커스와
// 타이핑 중인 값을 날려버린다(사유 입력 불가 버그의 원인).
function paint(i){
  const it=VIEW[i], row=document.querySelectorAll('.row')[i]; if(!row) return;
  row.classList.toggle('done', !!it['판정']);
  row.querySelectorAll('.acts button').forEach(b=>b.classList.toggle('on', b.dataset.v===it['판정']));
  const done=S.items.filter(x=>x['판정']).length, tot=S.items.length;
  $('#bar>i').style.width=(tot?done/tot*100:0)+'%';
  $('#meta').textContent=`판정 ${done} / ${tot}  ·  남은 ${tot-done}건  ·  화면 ${VIEW.length}건`;
}

function setCur(i){
  document.querySelectorAll('.row').forEach((r,n)=>r.classList.toggle('sel', n===i));
  cur=i;
  document.querySelectorAll('.row')[i]?.scrollIntoView({block:'nearest'});
}

async function post(it,v,reason){
  await fetch('/api/judge',{method:'POST',headers:{'Content-Type':'application/json',...H},
    body:JSON.stringify({파일:it['파일'],판정:v,사유:reason||''})});
}

async function judge(i,v,reason){
  const it=VIEW[i];
  it['판정']=v; it['사유']=reason||'';
  setCur(i); paint(i);
  await post(it,v,reason);
}

// 판정이 없는 상태의 사유는 원장에 쓰지 않는다(판정 원장이므로). 판정 시점에 함께 실린다.
async function saveReason(i,val){
  const it=VIEW[i]; if(!it || it['사유']===val) return;
  it['사유']=val;
  if(it['판정']) await post(it, it['판정'], val);
}

document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT') return;
  if(!VIEW.length) return;
  if(e.key==='j'){setCur(Math.min(cur+1,VIEW.length-1));}
  else if(e.key==='k'){setCur(Math.max(cur-1,0));}
  else if(e.key==='Enter'){  // 선택된 행의 사유칸으로 바로 진입
    e.preventDefault();
    document.querySelectorAll('.row')[cur]?.querySelector('input')?.focus();
  }
  else if('12345'.includes(e.key) && +e.key<=S.verdicts.length){
    const pick=S.verdicts[+e.key-1];
    const inp=document.querySelectorAll('.row')[cur]?.querySelector('input');
    judge(cur, VIEW[cur]['판정']===pick ? '' : pick, inp?inp.value:'');
  }
});
function esc(s){return String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    token: str = ""  # main() 에서 주입. 빈 문자열이면 검사하지 않는다.

    def _authorized(self) -> bool:
        if not self.token:
            return True
        q = parse_qs(urlparse(self.path).query)
        given = self.headers.get("X-Review-Token") or (q.get("t") or [""])[0]
        return secrets.compare_digest(given, self.token)

    def _send(self, code, body, ctype):
        raw = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = urlparse(self.path).path
        if not self._authorized():
            return self._send(403, "토큰이 필요하다. 기동 시 출력된 URL(?t=...)로 접속할 것.",
                              "text/plain; charset=utf-8")
        if path == "/api/data":
            self._send(200, json.dumps(build_state(), ensure_ascii=False), "application/json; charset=utf-8")
        elif path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if urlparse(self.path).path != "/api/judge":
            return self._send(404, "not found", "text/plain; charset=utf-8")
        if not self._authorized():
            return self._send(403, json.dumps({"error": "forbidden"}), "application/json; charset=utf-8")
        n = int(self.headers.get("Content-Length", 0))
        d = json.loads(self.rfile.read(n) or b"{}")
        verdict = d.get("판정", "")
        if verdict and verdict not in VERDICTS:  # "" = 판정 취소
            return self._send(400, json.dumps({"error": "bad verdict"}), "application/json; charset=utf-8")
        save_judgement(d.get("파일", ""), verdict, d.get("사유", ""))
        self._send(200, json.dumps({"ok": True}, ensure_ascii=False), "application/json; charset=utf-8")

    def log_message(self, format, *args):  # noqa: A002 - 콘솔 조용히
        pass


def local_ips() -> list[str]:
    """외부에서 접속할 때 쓸 만한 IPv4 목록 (loopback 제외)."""
    ips = set()
    try:  # 기본 경로로 나가는 인터페이스 주소
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = str(info[4][0])
            if not ip.startswith("127."):
                ips.add(ip)
    except OSError:
        pass
    return sorted(ips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--host", default="0.0.0.0", help="바인드 주소 (기본 0.0.0.0 = 외부 접근 허용)")
    ap.add_argument("--token", default=None, help="접근 토큰. 미지정 시 자동 생성")
    ap.add_argument("--no-token", action="store_true", help="토큰 검사 비활성 (로컬 전용일 때만)")
    args = ap.parse_args()

    Handler.token = "" if args.no_token else (args.token or secrets.token_urlsafe(12))

    cards = sorted(REVIEW.glob("S*_files.tsv"))
    if not cards:
        print(f"경고: {REVIEW}/S*_files.tsv 가 없다. 스테이지 카드를 먼저 만들어야 한다.")
    else:
        print(f"카드 {len(cards)}개: {', '.join(p.name for p in cards)}")
    print(f"원장: {LEDGER}")

    q = f"?t={Handler.token}" if Handler.token else ""
    print("\n접속 URL:")
    hosts = ["127.0.0.1"] if args.host == "127.0.0.1" else ["127.0.0.1", *local_ips()]
    for h in hosts:
        print(f"  http://{h}:{args.port}/{q}")
    if not Handler.token:
        print("\n⚠ 토큰 없이 기동됨 — 같은 네트워크의 누구나 판정을 쓸 수 있다.")
    elif args.host != "127.0.0.1":
        print("\n외부에서 안 열리면 방화벽을 확인할 것:")
        print(f"  sudo ufw allow {args.port}/tcp        # 필요 시")
        print(f"  ss -ltn | grep {args.port}")
    print("\n(Ctrl-C 종료)", flush=True)  # 로그 리다이렉트 시에도 URL 이 바로 보이게
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
