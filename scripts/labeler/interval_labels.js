// Point labels retain t/frame. For intervals t/frame alias the start boundary.
function marksOf(lb){
  let ms=lb.marks;if(typeof ms==='string'){try{ms=JSON.parse(ms);}catch(e){ms=[];}}
  if(!Array.isArray(ms)||!ms.length) ms=lb.t_fail!=null&&lb.t_fail!==''?[{t:+lb.t_fail,frame:Math.round(+lb.t_fail*FPS),type:lb.type,note:''}]:[];
  return ms.map(m=>({...m,kind:m.kind||'point',t:+m.t,frame:m.frame==null?Math.round(+m.t*FPS):+m.frame,type:m.type||null,note:m.note||''}));
}
function intervalChanged(m){
  state.marks.sort((a,b)=>a.t-b.t);state.msel=state.marks.indexOf(m);state.none=false;state.dirty=true;renderTypes();updateReadout();
}
function currentFrame(){return Math.round((v.currentTime||0)*FPS);}
function mark(){
  if(!v.src)return;v.pause();const frame=currentFrame();
  if(state.marks.some(m=>m.kind!=='interval'&&m.frame===frame)){toast('같은 프레임에 점 마크가 있습니다');return;}
  const m={kind:'point',t:frame/FPS,frame,type:null,note:''};state.marks.push(m);intervalChanged(m);
}
function startInterval(){
  if(!v.src)return;
  if(state.marks.some(m=>m.kind==='interval'&&m.end_frame==null)){toast('작성 중인 구간을 먼저 끝내거나 삭제하세요');return;}
  v.pause();const frame=currentFrame();const m={kind:'interval',t:frame/FPS,frame,start_t:frame/FPS,start_frame:frame,end_t:null,end_frame:null,type:null,note:''};
  state.marks.push(m);intervalChanged(m);toast('구간 시작 — 종료 시점에서 O, 숫자키로 유형 선택');
}
function changeBoundary(which,frame){
  const m=curMark();if(!m)return;
  if(!Number.isInteger(frame)||frame<0||(state.frames!=null&&frame>state.frames)){toast('영상 범위 안의 정수 프레임을 입력하세요');return;}
  if(which==='end'){
    if(frame<=m.frame){toast('종료는 시작보다 뒤여야 합니다');return;}
    m.kind='interval';m.start_t=m.t;m.start_frame=m.frame;m.end_frame=frame;m.end_t=frame/FPS;
  }else{
    if(m.kind==='interval'&&m.end_frame!=null&&frame>=m.end_frame){toast('시작은 종료보다 앞이어야 합니다');return;}
    m.frame=frame;m.t=frame/FPS;
    if(m.kind==='interval'){m.start_frame=frame;m.start_t=m.t;}
  }
  intervalChanged(m);
}
function moveMark(){if(!curMark()||!v.src)return;v.pause();changeBoundary('start',currentFrame());}
function endInterval(){if(!curMark()){toast('종료할 구간 또는 시작 점 마크를 선택하세요');return;}v.pause();changeBoundary('end',currentFrame());}
function renderMarks(){
  const box=$('marks');box.replaceChildren();
  if(!state.marks.length){box.textContent='M: 점 마크 · I: 구간 시작 · O: 선택 구간 종료';}
  state.marks.forEach((m,i)=>{
    const row=document.createElement('div');row.className='mrow'+(i===state.msel?' sel':'');
    const text=document.createElement('span');text.className='event-label';
    const ty=TYPES.find(t=>t[0]===m.type)?.[1]||'유형 없음';
    text.textContent=m.kind==='interval'?`${i+1}. 구간 ${m.t.toFixed(2)}–${m.end_t==null?'미완료':m.end_t.toFixed(2)}s · f${m.frame}–${m.end_frame??'?'} · ${ty}`:`${i+1}. 점 ${m.t.toFixed(2)}s · f${m.frame} · ${ty}`;
    if(m.note)text.textContent+=' · '+m.note;
    const del=document.createElement('button');del.textContent='✕';del.title='삭제';del.onclick=e=>{e.stopPropagation();delMark(i);};
    row.append(text,del);row.onclick=()=>{state.msel=i;v.pause();v.currentTime=m.t;renderTypes();updateReadout();};box.append(row);
  });
  const cm=curMark();$('rangeStartFrame').disabled=!cm;$('rangeEndFrame').disabled=!cm;
  $('rangeStartFrame').value=cm?cm.frame:'';$('rangeEndFrame').value=cm?.end_frame??'';
  $('bRangeEnd').disabled=!cm;$('bRangeStartMove').disabled=!cm;
}
function updateReadout(){
  const t=v.currentTime||0,f=Math.round(t*FPS);$('rT').textContent=t.toFixed(2);$('rF').textContent=f;$('rR').textContent=state.frames?(f/2.5).toFixed(1):'—';
  $('rM').textContent=state.none?'실패 없음':`${state.marks.length}개`;
  const duration=v.duration||1;$('tlhead').style.left=100*t/duration+'%';$('tlfill').style.width=100*t/duration+'%';
  const tl=$('tl');tl.querySelectorAll('.mk,.interval-band').forEach(e=>e.remove());
  state.marks.forEach((m,i)=>{
    if(m.kind==='interval'){
      const band=document.createElement('div');band.className='interval-band'+(i===state.msel?' selected':'');
      band.style.left=100*m.t/duration+'%';band.style.width=100*Math.max(0,(m.end_t??Math.max(m.t,t))-m.t)/duration+'%';tl.append(band);
    }
    for(const boundary of m.kind==='interval'&&m.end_t!=null?[m.t,m.end_t]:[m.t]){
      const e=document.createElement('div');e.className='mk'+(i===state.msel?' sel':'');e.dataset.n=String(i+1);e.style.left=100*boundary/duration+'%';tl.append(e);
    }
  });$('bPlay').textContent=v.paused?'▶ 재생':'❚❚ 정지';
}
async function save(thenNext){
  const c=state.cur;if(!c||state.saving)return;
  if(!state.none&&!state.marks.length){toast('마크를 추가하거나 실패 없음을 선택하세요');return;}
  if(state.marks.some(m=>m.kind==='interval'&&(m.end_frame==null||m.end_frame<=m.frame))){toast('미완료 구간의 종료를 지정한 후 저장하세요');return;}
  const ms=state.none?[]:state.marks.map(m=>{
    const base={kind:m.kind||'point',t:m.frame/FPS,frame:m.frame,type:m.type,note:(m.note||'').trim()};
    return m.kind==='interval'?{...base,start_t:m.frame/FPS,start_frame:m.frame,end_t:m.end_frame/FPS,end_frame:m.end_frame}:base;
  });
  const first=ms.find(m=>m.type!=='successful_retry')||null;
  const rec={schema_version:2,rel:c.rel,key:c.key,s:c.s,j:c.j,n:c.n,succ:c.succ,sig:c.sig,no_failure:state.none,t_fail:first?.t??null,frame:first?.frame??null,fps:FPS,frames_total:state.frames,type:first?.type??null,marks:ms,n_marks:ms.length,note:$('note').value.trim(),by:$('who').value.trim(),updated_at:new Date().toISOString()};
  const snapshot=JSON.stringify({marks:state.marks,none:state.none,note:$('note').value});
  state.saving=true;
  try{
    const r=await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(rec)});
    if(!r.ok){let detail='';try{detail=(await r.json()).err||'';}catch(e){}throw new Error(`${r.status} ${detail}`);}
    state.labels.set(c.rel,rec);
    const unchanged=state.cur===c&&snapshot===JSON.stringify({marks:state.marks,none:state.none,note:$('note').value});
    if(unchanged)state.dirty=false;
    renderTypes();renderList();toast(unchanged?'저장됨(승준 TSV)':'저장 완료 · 이후 수정 사항은 다시 저장하세요');
    if(thenNext&&unchanged){const rest=state.filtered.filter(x=>!state.labels.has(x.rel));const i=idx();const n=state.filtered.slice(i+1).find(x=>!state.labels.has(x.rel))||rest[0];if(n)open(n);else applyFilters();}
  }catch(e){toast('저장 실패 — 변경 사항 유지: '+e.message);}finally{state.saving=false;}
}
$('bRangeStart').onclick=startInterval;$('bRangeEnd').onclick=endInterval;$('bRangeStartMove').onclick=moveMark;
$('rangeStartFrame').onchange=()=>changeBoundary('start',$('rangeStartFrame').value===''?NaN:Number($('rangeStartFrame').value));
$('rangeEndFrame').onchange=()=>changeBoundary('end',$('rangeEndFrame').value===''?NaN:Number($('rangeEndFrame').value));
$('btnEventsCSV').onclick=()=>window.open('/api/events.csv','_blank');$('btnEventsTSV').onclick=()=>window.open('/api/events.tsv','_blank');
document.addEventListener('keydown',e=>{
  if(/^(input|textarea|select)$/i.test(e.target.tagName))return;
  if(e.key.toLowerCase()==='i'){e.preventDefault();startInterval();}
  if(e.key.toLowerCase()==='o'){e.preventDefault();endInterval();}
});
window.addEventListener('beforeunload',e=>{if(state.dirty){e.preventDefault();e.returnValue='';}});
