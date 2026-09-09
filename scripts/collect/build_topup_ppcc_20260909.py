"""Append new PPCC kitchens to the canonical v6 plan, preserving every old cell."""
from pathlib import Path
import sys, json, copy, importlib.util
ROOT=Path('/home/dongkyu/pkt_ws/temporal_vla')
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from src.collect.plan import CollectionPlan

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
select=module('scene_select',ROOT/'scripts/collect/select_v6_scenes.py')
build=module('scene_build',ROOT/'scripts/collect/build_v6_plan.py')
tasks=select.load_scan(ROOT/'outputs/analysis/seed_scan/fixture_groups');select.annotate_all(tasks)
plan=CollectionPlan.load(ROOT/'configs/collect/n15_grid_v6_scene_jitter/collection_plan.json')
assert plan.plan_id=='08f1c9df8207'
base=copy.deepcopy(plan)
requests={'PPCC/bread':[10,2],'PPCC/jug':[8],'PPCC/marshmallow':[1,6]}
selection={'env_kwargs':plan.env_kwargs,'keys':{}}
for key,layouts in requests.items():
 kd=next(k for k in select.key_defs() if k['key']==key);cs=select.candidates_by_layout(tasks,kd)
 entries=[]
 for layout in layouts:
  assert layout not in {s['layout'] for s in plan.scenes[key]}
  entry=select.pick_scene(kd,layout,cs[layout],{})
  assert entry is not None
  entries.append(entry)
 selection['keys'][key]={'kind':'pickplace','task':'PickPlaceCounterToCabinet','task_env':base.extra['env_names'][key],'scenes':entries}
extra=build.build(selection,name='ppcc_topup')
for key in requests:
 plan.instructions[key]+=extra.instructions[key]
 plan.scenes[key]+=extra.scenes[key]
 plan.jitters[key]+=extra.jitters[key]
plan.name='n15_topup_ppcc_20260909'
plan.note='5 new kitchens, 250 new cells; canonical 1800 cells unchanged/skipped. Coffee/apple excluded.'
plan.extra['topup']={'parent_plan_id':base.plan_id,'new_scene_machine':'kanu','excluded_collection':['CoffeeSetupMug','PPCC/apple']}
for key in base.instructions:
 for attr in ['instructions','scenes','jitters']:assert getattr(plan,attr)[key][:3]==getattr(base,attr)[key]
assert plan.env_kwargs==base.env_kwargs and plan.n_cells==2050
out=REPO/'configs/collect/n15_topup_ppcc_20260909';plan.save(out)
(out/'new_scene_selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
(out/'skip_existing_1800.txt').write_text('\n'.join(c.key for c in base.cells())+'\n')
(out/'new_cells.txt').write_text('\n'.join(c.key for c in plan.cells() if c.scene_idx>=3)+'\n')
print('PLAN',plan.plan_id)
for k in requests:
 for i,s in enumerate(plan.scenes[k][3:],3):print(k,i,s)
