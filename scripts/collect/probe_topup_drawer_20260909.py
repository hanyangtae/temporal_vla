from pathlib import Path
import importlib.util,sys,csv,argparse
from multiprocessing import Pool
repo=Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo))
from src.collect.plan import CollectionPlan
sp=importlib.util.spec_from_file_location('probe_cells',repo/'scripts/collect/probe_v6_jitter_cells.py');m=importlib.util.module_from_spec(sp);sys.modules['probe_cells']=m;sp.loader.exec_module(m)
a=argparse.ArgumentParser();a.add_argument('--keys',required=True);a.add_argument('--out',required=True);args=a.parse_args()
p=str(repo/'configs/collect/n15_topup_drawer_20260909/collection_plan.json');plan=CollectionPlan.load(p)
new=set((repo/'configs/collect/n15_topup_drawer_20260909/new_cells.txt').read_text().splitlines());keys=args.keys.split(',')
jobs=sorted({(p,c.instruction,c.scene_idx,c.jitter_idx) for c in plan.cells() if c.key in new and c.instruction in keys})
assert jobs
out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
with Pool(4,maxtasksperchild=1) as pool,out.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['key','sid','jid','env_seed','lat','back','reset_idx','base','ok','err'],delimiter='\t');w.writeheader();bad=[]
 for r in pool.imap_unordered(m.probe,jobs):w.writerow(r);f.flush();print(r,flush=True);bad += [r] if not r['ok'] else []
assert not bad,bad
print('PROBE PASS',len(jobs),flush=True)
