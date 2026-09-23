"""Temporary HTTP server tests; never writes production labels."""
import sys, json, tempfile, threading, urllib.request, urllib.error, csv, io, subprocess, re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from labeler_server_multiplan import H, ThreadingHTTPServer, dump, load
from label_events import validate_record

def run(html,chrome=None):
 with tempfile.TemporaryDirectory() as tmp:
  d=Path(tmp);labels=d/'labels.tsv'
  legacy=dict(rel='legacy',key='test',s=0,j=0,n=0,succ=0,fps=20,frames_total=360,no_failure=False,t_fail=1,frame=20,type='drop',marks=[dict(t=1,frame=20,type='drop',note='기존')])
  dump(str(labels),{'legacy':legacy});before=labels.read_bytes()
  H.root=str(d);H.labels_path=str(labels);H.labels=load(str(labels));H.index_html=html.encode()
  server=ThreadingHTTPServer(('127.0.0.1',0),H);threading.Thread(target=server.serve_forever,daemon=True).start();url='http://127.0.0.1:'+str(server.server_port)
  def get(p):return urllib.request.urlopen(url+p).read()
  def post(rec):
   req=urllib.request.Request(url+'/api/label',json.dumps(rec).encode(),{'Content-Type':'application/json'})
   try:r=urllib.request.urlopen(req);return r.status
   except urllib.error.HTTPError as e:return e.code
  assert json.loads(get('/api/labels'))[0]['marks']==legacy['marks'] and labels.read_bytes()==before
  mixed=dict(legacy,rel='mixed',schema_version=2,marks=[legacy['marks'][0],dict(kind='interval',t=2,frame=40,start_t=2,start_frame=40,end_t=4,end_frame=80,type='collision_stuck',note='구간'),dict(kind='interval',t=6,frame=120,start_t=6,start_frame=120,end_t=8,end_frame=160,type='no_reach',note='재발')])
  assert post(mixed)==200
  assert load(str(labels))['mixed']['marks']==mixed['marks']
  for ext,sep in [('csv',','),('tsv','\t')]:
   rows=list(csv.DictReader(io.StringIO(get('/api/events.'+ext).decode()),delimiter=sep));intervals=[r for r in rows if r['kind']=='interval'];assert len(intervals)==2 and intervals[1]['end_frame']=='160'
  bad=dict(mixed,marks=[dict(mixed['marks'][1],end_t=1,end_frame=20)]);previous=labels.read_bytes();assert post(bad)==400 and labels.read_bytes()==previous
  bad=dict(mixed,marks=[dict(mixed['marks'][1],end_t=None,end_frame=None)]);assert post(bad)==400
  bad=dict(mixed,marks=[dict(mixed['marks'][1],end_t=25,end_frame=500)]);assert post(bad)==400
  stale=dict(mixed);stale.pop('schema_version');assert post(stale)==409
  retry=dict(legacy,rel='retry',schema_version=2,marks=[dict(t=1,frame=20,kind='point',type='successful_retry')]);assert post(retry)==200 and load(str(labels))['retry']['t_fail'] is None
  print('PASS HTTP: legacy, multi-interval roundtrip, CSV/TSV endpoints, invalid bounds, old-client guard, retry exclusion')
  if chrome:
   testjs=Path(__file__).with_name('interval_browser_checks.js').read_text()
   H.index_html=html.replace('</body>','<script>'+testjs+'</script></body>').encode()
   r=subprocess.run([chrome,'--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--no-proxy-server','--user-data-dir='+str(d/'chrome'),'--virtual-time-budget=12000','--dump-dom',url],capture_output=True,text=True,timeout=40)
   (Path('/tmp')/'labeler_interval_browser_result.html').write_text(r.stdout)
   assert 'data-interval-tests="PASS"' in r.stdout, r.stdout[-1800:]+r.stderr[-1000:]
   print('PASS Chromium: point + interval creation, keyboard, boundary editing, reload, timeline, save failure retained')
  server.shutdown()

if __name__=='__main__':run(Path(sys.argv[1]).read_text(),sys.argv[2] if len(sys.argv)>2 else None)
