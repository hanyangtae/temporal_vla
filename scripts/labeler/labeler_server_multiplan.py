#!/usr/bin/env python3
"""v6 실패 시점 라벨러 — 승준 자체 서빙 (stdlib only).
GET /            → index.html   GET /video/<rel>/video.mp4 → 아카이브 영상(Range 지원)
GET /api/labels  → 저장된 라벨 JSON   POST /api/label → 한 셀 저장(upsert, TSV 원자적 재기록)
GET /api/export.csv → CSV
사용: python3 labeler_server.py --root <grid>/08f1c9df8207 --labels <tsv> --port 8767 (localhost 바인드)
"""
import argparse, json, os, csv, io, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import unquote
from label_events import validate_record, event_rows, EVENT_COLS
COLS=["rel","key","s","j","n","succ","no_failure","t_fail","frame","fps","frames_total","type","n_marks","marks","note","by","updated_at","sig"]
LOCK=threading.Lock()
def load(p):
    d={}
    if os.path.exists(p):
        for r in csv.DictReader(open(p,newline='',encoding='utf-8'),delimiter='\t'):
            for k in ("s","j","n","succ","frame","frames_total","fps"):
                if r.get(k) not in (None,""): r[k]=int(float(r[k]))
                else: r[k]=None
            r["t_fail"]=float(r["t_fail"]) if r.get("t_fail") not in (None,"") else None
            r["no_failure"]=r.get("no_failure") in ("1","True","true")
            try: r["marks"]=json.loads(r.get("marks") or "[]")
            except Exception: r["marks"]=[]
            r["n_marks"]=len(r["marks"])
            d[r["rel"]]=r
    return d
def dump(p,d):
    tmp=p+".tmp"
    with open(tmp,"w",newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=COLS,delimiter='\t',extrasaction='ignore'); w.writeheader()
        for r in sorted(d.values(),key=lambda x:x["rel"]):
            rr=dict(r); rr["no_failure"]="1" if r.get("no_failure") else "0"
            rr["marks"]=json.dumps(r.get("marks") or [],ensure_ascii=False); rr["n_marks"]=len(r.get("marks") or [])
            for k in COLS:
                if rr.get(k) is None: rr[k]=""
            w.writerow(rr)
    os.replace(tmp,p)
class H(BaseHTTPRequestHandler):
    root=None; labels_path=None; index_html=None; labels=None; extra_roots={}
    def log_message(self,*a): pass
    def _send(self,code,body,ctype="application/json; charset=utf-8",extra=None):
        self.send_response(code); self.send_header("Content-Type",ctype); self.send_header("Content-Length",str(len(body)))
        for k,v in (extra or {}).items(): self.send_header(k,v)
        self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        p=unquote(self.path.split("?")[0])
        if p=="/" or p=="/index.html":
            return self._send(200,self.index_html,"text/html; charset=utf-8")
        if p=="/api/labels":
            with LOCK: body=json.dumps(list(self.labels.values()),ensure_ascii=False).encode()
            return self._send(200,body)
        if p in ("/api/events.csv", "/api/events.tsv"):
            with LOCK: rows=list(event_rows(self.labels.values()))
            sep="\t" if p.endswith(".tsv") else ","
            b=io.StringIO(); w=csv.DictWriter(b,fieldnames=EVENT_COLS,delimiter=sep); w.writeheader(); w.writerows(rows)
            ext="tsv" if sep=="\t" else "csv"
            return self._send(200,b.getvalue().encode(),"text/plain; charset=utf-8",{"Content-Disposition":f'attachment; filename="v6_label_events.{ext}"'})
        if p=="/api/export.csv":
            with LOCK: rows=sorted(self.labels.values(),key=lambda x:x["rel"])
            b=io.StringIO(); w=csv.DictWriter(b,fieldnames=COLS,extrasaction='ignore'); w.writeheader()
            for r in rows:
                rr={k:("" if r.get(k) is None else r.get(k)) for k in COLS}; rr["marks"]=json.dumps(r.get("marks") or [],ensure_ascii=False); w.writerow(rr)
            return self._send(200,b.getvalue().encode(),"text/csv; charset=utf-8",{"Content-Disposition":'attachment; filename="v6_failure_onset_labels.csv"'})
        if p.startswith("/video/"):
            rel=p[len("/video/"):]
            if ".." in rel or not rel.endswith("/video.mp4"): return self._send(404,b"bad path","text/plain")
            parts=rel.split("/",1)
            if parts[0] in self.extra_roots and len(parts)==2:
                fp=os.path.join(self.extra_roots[parts[0]],parts[1])
            else:
                fp=os.path.join(self.root,rel)
            if not os.path.isfile(fp): return self._send(404,b"not found","text/plain")
            size=os.path.getsize(fp); rng=self.headers.get("Range")
            start,end=0,size-1
            if rng and rng.startswith("bytes="):
                a,_,b=rng[6:].partition("-")
                if a: start=int(a)
                if b: end=min(int(b),size-1)
                if not a and b: start=max(0,size-int(b)); end=size-1
            length=end-start+1
            self.send_response(206 if rng else 200); self.send_header("Content-Type","video/mp4"); self.send_header("Accept-Ranges","bytes")
            self.send_header("Content-Length",str(length))
            if rng: self.send_header("Content-Range",f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(fp,"rb") as f:
                f.seek(start); left=length
                while left>0:
                    chunk=f.read(min(1<<20,left))
                    if not chunk: break
                    try: self.wfile.write(chunk)
                    except BrokenPipeError: return
                    left-=len(chunk)
            return
        return self._send(404,b"not found","text/plain")
    def do_POST(self):
        if self.path!="/api/label": return self._send(404,b"","text/plain")
        n=int(self.headers.get("Content-Length","0")); rec=json.loads(self.rfile.read(n).decode())
        if not rec.get("rel") or ".." in rec["rel"]: return self._send(400,b'{"err":"rel"}')
        try:
            normalized=validate_record(rec)
        except (ValueError, TypeError, KeyError, OverflowError) as e:
            return self._send(400,json.dumps({"err":str(e)}).encode())
        with LOCK:
            old=self.labels.get(rec["rel"],{})
            if any(m.get("kind")=="interval" for m in old.get("marks",[])) and rec.get("schema_version")!=2:
                return self._send(409,b'{"err":"Interval labels exist. Refresh the labeler before saving."}')
            updated=dict(self.labels);updated[rec["rel"]]=normalized
            dump(self.labels_path,updated)
            self.labels.clear();self.labels.update(updated)
        return self._send(200,b'{"ok":true}')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); ap.add_argument("--labels",required=True); ap.add_argument("--index",default=os.path.join(os.path.dirname(os.path.abspath(__file__)),"index.html")); ap.add_argument("--port",type=int,default=8767); ap.add_argument("--host",default="127.0.0.1")
    ap.add_argument("--extra-root",action="append",default=[],metavar="PLAN_ID=PATH")
    a=ap.parse_args()
    for item in a.extra_root:
        name,path=item.split("=",1)
        if not name or "/" in name or ".." in name: ap.error("invalid extra-root plan id")
        H.extra_roots[name]=os.path.abspath(path)
    H.root=os.path.abspath(a.root); H.labels_path=os.path.abspath(a.labels); H.index_html=open(a.index,"rb").read(); H.labels=load(H.labels_path)
    print(f"serving {H.root} on http://{a.host}:{a.port}  labels={H.labels_path} ({len(H.labels)} rows)",flush=True)
    ThreadingHTTPServer((a.host,a.port),H).serve_forever()
if __name__=="__main__": main()
