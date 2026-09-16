"""Expose archived bread s4 and dishwasher-left s3 without changing labels."""
import argparse,json,re
from pathlib import Path

SPECS=[('4fa6496cd684','worker2','PPCC/bread',4,3,24),('4ab360df2b71','worker1','DishwasherRack/out-left',3,0,23)]
def extend(html,root):
    cm=re.search(r'const CELLS = (\[.*?\]);',html);assert cm
    old=json.loads(cm.group(1));new=[];targets=[];stats=[]
    for plan,machine,key,scene,target,expected_fail in SPECS:
        assert not any(c['key']==key and c['s']==scene for c in old),'scene already present'
        entries=[]
        for j in range(5):
            for n in range(10):
                rel=f'{plan}/{machine}/{key}/s{scene}/j{j}/n{n}/base';p=root/rel
                m=json.loads((p/'meta.json').read_text())
                assert (p/'video.mp4').is_file()
                assert (m['plan_id'],m['machine'],m['grid_instruction'],m['scene_idx'],m['jitter_idx'],m['noise_idx'])==(plan,machine,key,scene,j,n)
                entries.append(dict(rel=rel,key=key,s=scene,j=j,n=n,succ=int(m['success']),sig=m['sig'],lang=m['lang'],source=f'추가 baseline · {machine} · {plan}'))
                if j==target:targets.append(rel)
        failures=sum(c['succ']==0 for c in entries);assert failures==expected_fail
        new.extend(entries);stats.append(dict(key=key,scene=scene,total=len(entries),failures=failures,target_j=target))
    html=html[:cm.start(1)]+json.dumps(old+new,ensure_ascii=False)+html[cm.end(1):]
    tm=re.search(r'const TARGET_RELS = new Set\((\[.*?\])\);',html);assert tm
    prev=json.loads(tm.group(1));html=html[:tm.start(1)]+json.dumps(prev+targets,ensure_ascii=False)+html[tm.end(1):]
    before='<option value="target">대상 j만 · detector 평가</option><option value="all">전체 j</option>'
    assert before in html
    html=html.replace(before,'<option value="all">전체 j</option><option value="target">대상 j만 · detector 평가</option>')
    helptext='추가 bread s4·rack-L s3 등록 완료. 전체 j + 실패 판만: bread 24판, rack-L 23판. 성공 포함 시 각각 50판. 추가 자료는 plan·수집 머신별 원본으로 구분됩니다.'
    needle='<div class="help">기존 v6 + 마시멜로 추가 s3·s4.'
    pos=html.index(needle);end=html.index('</div>',pos)
    html=html[:pos]+'<div class="help">'+helptext+html[end:]
    assert json.loads(re.search(r'const CELLS = (\[.*?\]);',html).group(1))[:len(old)]==old
    assert 'successful_retry' in html
    return html,stats

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--grid-root',type=Path,required=True);a=ap.parse_args()
    html,stats=extend(a.index.read_text(),a.grid_root);a.out.write_text(html);print(json.dumps(stats))
