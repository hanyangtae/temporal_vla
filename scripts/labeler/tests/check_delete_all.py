"""Exercise destructive action exclusively against an isolated test TSV."""
import sys,tempfile,threading,subprocess,json,urllib.request,urllib.error
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from labeler_server_multiplan import H,ThreadingHTTPServer,dump,load

JS=r'''
(async()=>{
 const check=(ok,why)=>{if(!ok)throw new Error(why);};
 try{
  await new Promise(r=>setTimeout(r,600));check(state.labels.size===2,'initial labels');
  let confirmations=0;window.confirm=()=>{confirmations++;return false;};
  await $('btnDeleteAll').onclick();check(confirmations===1&&state.labels.size===2,'cancel');
  const originalFetch=window.fetch;window.confirm=()=>{confirmations++;return true;};
  window.fetch=async()=>({ok:false,status:500});await $('btnDeleteAll').onclick();
  check(confirmations===2&&state.labels.size===2,'failure preserves labels');window.fetch=originalFetch;
  state.dirty=true;$('note').value='unsaved';
  await $('btnDeleteAll').onclick();check(confirmations===3,'one confirmation per click');
  check(state.labels.size===0&&!state.dirty&&state.marks.length===0&&$('note').value==='','UI cleared');
  check((await(await fetch('/api/labels')).json()).length===0,'server cleared');
  document.body.dataset.deleteTests='PASS';
 }catch(e){document.body.dataset.deleteTests='FAIL';const p=document.createElement('pre');p.textContent=e.stack;document.body.append(p);}
})();
'''
with tempfile.TemporaryDirectory() as temp:
 d=Path(temp);f=d/'labels.tsv';records={str(i):dict(rel=str(i),no_failure=True,marks=[],note='keep until click') for i in range(2)};dump(str(f),records);before=f.read_bytes()
 H.labels_path=str(f);H.labels=load(str(f));H.root=str(d);H.index_html=Path(sys.argv[1]).read_text().replace('</body>','<script>'+JS+'</script></body>').encode()
 srv=ThreadingHTTPServer(('127.0.0.1',0),H);threading.Thread(target=srv.serve_forever,daemon=True).start();url='http://127.0.0.1:'+str(srv.server_port)
 try:urllib.request.urlopen(urllib.request.Request(url+'/api/labels/delete-all',b'{}',{'Content-Type':'application/json'}));raise AssertionError('confirmation missing')
 except urllib.error.HTTPError as e:assert e.code==400
 assert f.read_bytes()==before
 r=subprocess.run([sys.argv[2],'--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--no-proxy-server','--user-data-dir='+str(d/'chrome'),'--virtual-time-budget=12000','--dump-dom',url],capture_output=True,text=True,timeout=40)
 Path('/tmp/labeler_delete_test_result.html').write_text(r.stdout)
 assert 'data-delete-tests="PASS"' in r.stdout,r.stdout[-1200:]+r.stderr[-700:]
 assert load(str(f))=={} and H.labels=={}
 backups=list((d/'deleted_label_backups').glob('*.tsv'));assert len(backups)==1 and backups[0].read_bytes()==before
 srv.shutdown();print('PASS: missing confirmation rejected, cancel, failed request preserved, single confirm, full deletion, reload, byte-exact backup')
