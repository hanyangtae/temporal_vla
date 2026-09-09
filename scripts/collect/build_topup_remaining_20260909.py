"""Append validated remaining kitchens; preserve all canonical and PPCC cells."""
from pathlib import Path
import sys,json,copy,importlib.util
repo=Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo))
from src.collect.plan import CollectionPlan
sp=importlib.util.spec_from_file_location('builder',repo/'scripts/collect/build_v6_plan.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
selection=json.loads(Path(sys.argv[1]).read_text())
base=CollectionPlan.load(repo/'configs/collect/n15_topup_ppcc_20260909/collection_plan.json')
plan=copy.deepcopy(base); extra=m.build(selection,name='remaining_topup')
for key in selection['keys']:
 assert key not in ['CoffeeSetupMug','PPCC/apple']
 for scene in extra.scenes[key]:assert scene['layout'] not in {s['layout'] for s in base.scenes[key]}
 for attr in ['instructions','scenes','jitters']:getattr(plan,attr)[key]+=getattr(extra,attr)[key]
for key in base.instructions:
 for attr in ['instructions','scenes','jitters']:assert getattr(plan,attr)[key][:len(getattr(base,attr)[key])]==getattr(base,attr)[key]
plan.name='n15_topup_remaining_20260909';plan.note='Canonical 1800 + PPCC250 preserved; only new remaining kitchens collected.'
plan.extra['topup_remaining']={'parent_plan_id':base.plan_id,'machine_contract':'collect and future eval on same machine','excluded':['CoffeeSetupMug','PPCC/apple']}
out=repo/'configs/collect/n15_topup_remaining_20260909';plan.save(out)
(out/'new_scene_selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
old={c.key for c in base.cells()};new=[c.key for c in plan.cells() if c.key not in old]
(out/'skip_existing.txt').write_text('\n'.join(sorted(old))+'\n');(out/'new_cells.txt').write_text('\n'.join(new)+'\n')
print(plan.plan_id,len(new),plan.n_cells)
