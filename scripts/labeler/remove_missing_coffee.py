"""Remove only listed kanu Coffee cells whose video files are absent."""
import argparse,json,re
from pathlib import Path

def prune(html,root):
    cm=re.search(r'const CELLS = (\[.*?\]);',html);assert cm
    cells=json.loads(cm.group(1));removed=[];kept=[]
    for c in cells:
        rel=c['rel'];parts=rel.split('/')
        machine=parts[0] if parts[0] in ('kanu','worker1','worker2') else parts[1]
        path=root/'08f1c9df8207'/rel if parts[0] in ('kanu','worker1','worker2') else root/rel
        if c['key']=='CoffeeSetupMug' and machine=='kanu' and not (path/'video.mp4').is_file():removed.append(c)
        else:kept.append(c)
    assert removed,'No missing Coffee cells; no change needed'
    html=html[:cm.start(1)]+json.dumps(kept,ensure_ascii=False)+html[cm.end(1):]
    tm=re.search(r'const TARGET_RELS = new Set\((\[.*?\])\);',html);assert tm
    targets=json.loads(tm.group(1));gone={c['rel'] for c in removed}
    targets=[x for x in targets if x not in gone]
    html=html[:tm.start(1)]+json.dumps(targets,ensure_ascii=False)+html[tm.end(1):]
    return html,dict(before=len(cells),after=len(kept),removed=removed)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--grid-root',type=Path,required=True);ap.add_argument('--receipt',type=Path,required=True);a=ap.parse_args()
    html,receipt=prune(a.index.read_text(),a.grid_root);a.out.write_text(html);a.receipt.write_text(json.dumps(receipt,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='removed'}))
