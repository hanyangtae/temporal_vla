"""Add click-only, single-confirmation deletion of all annotation records."""
import argparse
from pathlib import Path
JS = r'''
$('btnDeleteAll').onclick=async()=>{
  if(state.saving){toast('저장 중입니다. 완료 후 다시 눌러주세요');return;}
  if(!confirm('모든 영상의 기존 라벨을 전체 삭제할까요?\n현재 필터와 관계없이 점·구간·메모 전체와 이 탭의 미저장 내용이 삭제됩니다. 영상과 원본 성공/실패 정보는 유지됩니다.'))return;
  const button=$('btnDeleteAll');button.disabled=true;state.saving=true;
  try{
    const response=await fetch('/api/labels/delete-all',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirmation:'DELETE_ALL_LABELS'})});
    if(!response.ok)throw new Error('HTTP '+response.status);
    const result=await response.json();
    state.labels.clear();state.marks=[];state.msel=-1;state.none=false;state.dirty=false;
    $('note').value='';$('mnote').value='';
    $('dbst').textContent='저장소 승준 TSV · 라벨 0';
    renderTypes();applyFilters();updateReadout();
    toast(result.deleted+'개 라벨 전체 삭제 완료');
  }catch(e){toast('전체 삭제 실패 — 화면의 라벨을 유지합니다: '+e.message);}
  finally{button.disabled=false;state.saving=false;}
};
'''
def extend(html):
    assert 'id="btnDeleteAll"' not in html
    anchor='<button id="btnEventsTSV">사건별 TSV</button>'
    assert anchor in html and '// ---- init' in html
    return html.replace(anchor,anchor+'<button id="btnDeleteAll" style="color:var(--mark);border-color:var(--mark)" title="현재 필터와 관계없이 저장된 모든 라벨 삭제">전체 라벨 삭제</button>').replace('// ---- init',JS+'\n// ---- init')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.write_text(extend(a.index.read_text()))
