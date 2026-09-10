"""Patch the standalone v6 labeler without changing its cells or stored labels."""
import argparse,json,re
from pathlib import Path

def patch(html, manifest):
    if 'id="fTarget"' in html:
        raise ValueError('Target filter already installed; refusing duplicate patch')
    cells=json.loads(re.search(r'const CELLS = (\[.*?\]);',html).group(1))
    target=set(manifest['target_rels']); available={c['rel'] for c in cells}
    assert target and target <= available
    def replace(old,new):
        nonlocal html
        assert html.count(old)==1,(old[:100],html.count(old))
        html=html.replace(old,new)
    replace('<div class="filters">','''<div class="filters">
    <div class="row"><select id="fTarget" aria-label="평가 대상 범위"><option value="target">대상 j만 · detector 평가</option><option value="all">전체 j</option></select></div>
    <div class="help">대상 j는 eval manifest와 원본 rollout 서명으로 대조했습니다. 현재 목록은 원본 v6이며 추가 수집분은 미포함입니다.</div>''')
    replace('// ---- filters/list','// ---- filters/list\nconst TARGET_RELS = new Set('+json.dumps(sorted(target),ensure_ascii=False)+');\n'+'''
const FILTER_IDS=["fTarget","fKey","fScene","fStatus","fSucc"];
let appliedFilters=null;
function baseMatches(c){
  const k=$("fKey").value,s=$("fScene").value,sc=$("fSucc").value;
  return ($("fTarget").value==="all"||TARGET_RELS.has(c.rel))&&(!k||c.key===k)&&(!s||String(c.s)===s)&&(sc==="all"||c.succ===0);
}
''')
    start=html.index('function applyFilters(){');end=html.index('function renderList(){',start)
    html=html[:start]+'''function applyFilters(){
  const st=$("fStatus").value;
  const next=CELLS.filter(c=>baseMatches(c)&&(st==="all"||(st==="done")===state.labels.has(c.rel)));
  const leaving=state.cur&&!next.some(c=>c.rel===state.cur.rel);
  if(leaving&&state.dirty){
    if(!confirm("저장하지 않은 라벨이 있습니다. 버리고 필터를 변경할까요?")){
      if(appliedFilters) FILTER_IDS.forEach(id=>$(id).value=appliedFilters[id]);
      return;
    }
    state.dirty=false;
  }
  appliedFilters=Object.fromEntries(FILTER_IDS.map(id=>[id,$(id).value]));
  state.filtered=next;
  if(leaving){
    if(next.length){open(next[0]);return;}
    state.cur=null;state.marks=[];state.msel=-1;state.none=false;state.dirty=false;
    v.pause();v.removeAttribute("src");v.load();
    $("tKey").textContent="조건에 맞는 영상 없음";$("tCoord").textContent="";$("tLang").textContent="";
    $("note").value="";renderTypes();
  } else if(!state.cur&&next.length){open(next[0]);return;}
  renderList();
}
'''+html[end:]
    replace('CELLS.filter(x=>x.key===c.key&&x.succ===0).length','CELLS.filter(x=>baseMatches(x)&&x.key===c.key).length')
    replace('CELLS.filter(x=>x.key===c.key&&x.succ===0&&state.labels.has(x.rel)).length','CELLS.filter(x=>baseMatches(x)&&x.key===c.key&&state.labels.has(x.rel)).length')
    replace('const total=CELLS.filter(c=>c.succ===0).length,done=CELLS.filter(c=>c.succ===0&&state.labels.has(c.rel)).length;','const total=CELLS.filter(baseMatches).length,done=CELLS.filter(c=>baseMatches(c)&&state.labels.has(c.rel)).length;')
    replace('(100*done/total)+"%"','(total?100*done/total:0)+"%"')
    replace('${done} / ${total} 실패 판 라벨됨','${done} / ${total} 판 라벨됨')
    replace('["fKey","fScene","fStatus","fSucc"].forEach(id=>$(id).onchange=applyFilters);','FILTER_IDS.forEach(id=>$(id).onchange=applyFilters);')
    replace('const c=CELLS.find(x=>x.rel===last)||state.filtered[0]||CELLS.find(x=>x.succ===0);','const c=state.filtered.find(x=>x.rel===last)||state.filtered[0];')
    assert json.loads(re.search(r'const CELLS = (\[.*?\]);',html).group(1))==cells
    return html

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--targets',type=Path,default=Path(__file__).with_name('eval_targets_20260910.json'));a=ap.parse_args()
    a.out.write_text(patch(a.index.read_text(),json.loads(a.targets.read_text())))
    print('Patched',a.out)
