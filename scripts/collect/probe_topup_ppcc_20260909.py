from pathlib import Path
import importlib.util,sys,csv
from multiprocessing import Pool
repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo))
from src.collect.plan import CollectionPlan
sp=importlib.util.spec_from_file_location('probe_cells',repo/'scripts/collect/probe_v6_jitter_cells.py');m=importlib.util.module_from_spec(sp);sys.modules['probe_cells']=m;sp.loader.exec_module(m)
p=str(repo/'configs/collect/n15_topup_ppcc_20260909/collection_plan.json')
plan=CollectionPlan.load(p);jobs=sorted({(p,c.instruction,c.scene_idx,c.jitter_idx) for c in plan.cells() if c.scene_idx>=3})
if '--first' in sys.argv:jobs=[j for j in jobs if j[1]=='PPCC/bread' and j[2]==3]
out=Path('/temporal_vla/outputs/collect/topup_ppcc_20260909');out.mkdir(parents=True,exist_ok=True)
with Pool(5) as pool,(out/('probe_first.tsv' if '--first' in sys.argv else 'probe_all.tsv')).open('w') as f:
 w=csv.DictWriter(f,fieldnames=['key','sid','jid','env_seed','lat','back','reset_idx','base','ok','err'],delimiter='\t');w.writeheader();bad=[]
 for r in pool.imap_unordered(m.probe,jobs):w.writerow(r);f.flush();print(r,flush=True);bad += [r] if not r['ok'] else []
assert not bad,bad
print('PROBE PASS',len(jobs),flush=True)
