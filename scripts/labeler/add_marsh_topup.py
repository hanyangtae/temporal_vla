"""Append the correctly collected marshmallow scenes, preserving canonical cells."""
import argparse,json,re
from pathlib import Path

def extend(html, entries, target_j):
 match=re.search(r'const CELLS = (\[.*?\]);',html);assert match
 old=json.loads(match.group(1));existing={c['rel'] for c in old};new=[]
 for e in entries:
  m=e['meta'];assert e['video_exists'] and m['machine']=='worker2' and m['grid_instruction']=='PPCC/marshmallow'
  assert m['scene_idx'] in [3,4] and m['plan_id']=='4fa6496cd684'
  rel=m['plan_id']+'/'+e['rel'];assert rel not in existing
  new.append(dict(rel=rel,key=m['grid_instruction'],s=m['scene_idx'],j=m['jitter_idx'],n=m['noise_idx'],succ=int(m['success']),lang=m['lang'],sig=m['sig'],source='추가 재수집 baseline'))
 assert len(new)==100 and len({c['rel'] for c in new})==100
 assert {(c['s'],c['j'],c['n']) for c in new}=={(s,j,n) for s in [3,4] for j in range(5) for n in range(10)}
 html=html[:match.start(1)]+json.dumps(old+new,ensure_ascii=False)+html[match.end(1):]
 target=re.search(r'const TARGET_RELS = new Set\((\[.*?\])\);',html);assert target
 original=json.loads(target.group(1));added=[c['rel'] for c in new if c['j']==target_j[c['s']]]
 html=html[:target.start(1)]+json.dumps(original+added,ensure_ascii=False)+html[target.end(1):]
 oldselect='<select id="fScene"><option value="">s 전체</option><option>0</option><option>1</option><option>2</option></select>'
 assert oldselect in html
 html=html.replace(oldselect,'<select id="fScene"><option value="">s 전체</option>'+''.join('<option>'+str(x)+'</option>' for x in sorted({c['s'] for c in old+new}))+'</select>')
 oldhelp='대상 j는 eval manifest와 원본 rollout 서명으로 대조했습니다. 현재 목록은 원본 v6이며 추가 수집분은 미포함입니다.'
 assert oldhelp in html
 html=html.replace(oldhelp,'기존 v6 + 마시멜로 추가 s3·s4. 추가 scene은 올바른 머신에서 재수집한 baseline이며, 과거 eval 원본과 서명이 다릅니다. 대상 j: s3→j4, s4→j2; 전체 j에서 각 scene 50판을 볼 수 있습니다.')
 needle='$("tLang").textContent=c.lang;';assert needle in html
 html=html.replace(needle,'$("tLang").textContent=c.lang+(c.source?" · "+c.source:"");')
 assert json.loads(re.search(r'const CELLS = (\[.*?\]);',html).group(1))[:len(old)]==old
 return html,{'old_cells':len(old),'added':len(new),'total':len(old)+len(new),'added_target':len(added),'preserved_targets':len(original)}

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--index',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--entries',type=Path);ap.add_argument('--grid-root',type=Path);a=ap.parse_args()
 if a.entries:entries=json.loads(a.entries.read_text())
 else:
  assert a.grid_root;entries=[]
  for p in sorted((a.grid_root/'4fa6496cd684').glob('worker2/PPCC/marshmallow/s[34]/j*/n*/base/meta.json')):
   entries.append({'rel':str(p.parent.relative_to(a.grid_root/'4fa6496cd684')),'meta':json.loads(p.read_text()),'video_exists':(p.parent/'video.mp4').is_file()})
 result,stats=extend(a.index.read_text(),entries,{3:4,4:2});a.out.write_text(result);print(json.dumps(stats))
