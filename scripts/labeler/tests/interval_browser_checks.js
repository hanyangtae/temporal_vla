(async()=>{
 const check=(v,m)=>{if(!v)throw new Error(m);};
 try{
  await new Promise(resolve=>setTimeout(resolve,700));
  state.dirty=false;open(CELLS.find(c=>!c.succ));state.frames=360;
  Object.defineProperty(v,'duration',{configurable:true,value:18});
  const seek=t=>{v.currentTime=t;Object.defineProperty(v,'currentTime',{configurable:true,writable:true,value:t});};
  seek(1);$('bMark').click();setType('drop');check(state.marks.length===1,'point');
  seek(2);document.body.dispatchEvent(new KeyboardEvent('keydown',{key:'i',bubbles:true}));setType('collision_stuck');
  check(state.marks[1].kind==='interval','I key');
  await save(false);check(state.dirty&&!state.labels.has(state.cur.rel),'unfinished save blocked');
  seek(4);document.body.dispatchEvent(new KeyboardEvent('keydown',{key:'o',bubbles:true}));
  check(state.marks[1].end_frame===80,'O key');
  $('rangeEndFrame').value=100;$('rangeEndFrame').dispatchEvent(new Event('change'));check(state.marks[1].end_t===5,'end edit');
  $('rangeEndFrame').value=10;$('rangeEndFrame').dispatchEvent(new Event('change'));check(state.marks[1].end_t===5,'reject reversed');
  seek(6);$('bRangeStart').click();setType('no_reach');seek(8);$('bRangeEnd').click();
  check(document.querySelectorAll('.interval-band').length===2,'bands');
  await save(false);check(!state.dirty,'persisted');
  const loaded=(await (await fetch('/api/labels')).json()).find(r=>r.rel===state.cur.rel);state.marks=marksOf(loaded);renderTypes();updateReadout();
  check(state.marks.length===3&&state.marks[2].end_t===8,'reload mixed');
  const nativeFetch=window.fetch;window.fetch=async()=>({ok:false,status:503,json:async()=>({err:'test failure'})});
  state.dirty=true;await save(true);check(state.dirty,'failed save dirty retained');window.fetch=nativeFetch;
  document.body.dataset.intervalTests='PASS';
 }catch(e){document.body.dataset.intervalTests='FAIL';const p=document.createElement('pre');p.textContent=e.stack;document.body.append(p);}
})();
