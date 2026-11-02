import json,math,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];O=ROOT/'outputs/eval/robocasa/groot_n15/safe_qam_multitask_scratch_20261021'
import sys,time,traceback
VERIFY_RUN_ID=str(time.time_ns())
(O/'verification_history').mkdir(exist_ok=True)
def save_exception(kind,value,tb):
 (O/'verification_history'/f'{VERIFY_RUN_ID}.error.txt').write_text(''.join(traceback.format_exception(kind,value,tb)))
 sys.__excepthook__(kind,value,tb)
sys.excepthook=save_exception
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def wilson(s,n):
 if not n:return None
 z=1.95996398454;p=s/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [c-h,c+h]
M=json.loads((O/'manifest.json').read_text());allrows=lines(O/'progress.jsonl');legacy=[x for x in allrows if x['arm']=='always'];originals=json.loads((O/'original_baseline_audit.json').read_text());rows=[{'cell':x['s'],'n':x['n'],'arm':'base','success':x['meta']['success'],'provenance':'original archive','original_pkl_sha256':x['meta']['pkl_sha256']} for x in originals]+[x for x in allrows if x['arm']=='safe'];M['arms']=['base','safe'];result={'completed':len(rows),'expected':60,'status':'complete' if len(rows)==60 else 'partial','arms':{},'cells':{},'paired':{},'audits':{}}
for arm in M['arms']:
 v=[x for x in rows if x['arm']==arm];s=sum(bool(x['success']) for x in v);result['arms'][arm]={'success':s,'n':len(v),'sr':s/len(v) if v else None,'wilson95':wilson(s,len(v))}
for c in M['cells']:
 sid=c['scene_idx'];result['cells'][str(sid)]={}
 for arm in M['arms']:
  v=[x for x in rows if x['arm']==arm and x['cell']==sid];s=sum(bool(x['success']) for x in v);result['cells'][str(sid)][arm]={'success':s,'n':len(v),'wilson95':wilson(s,len(v))}
lookup={(x['cell'],x['n'],x['arm']):x for x in rows}
for arm in ['safe']:
 pairs=[(bool(b['success']),bool(lookup[(cell,n,arm)]['success'])) for (cell,n,a),b in lookup.items() if a=='base' and (cell,n,arm) in lookup];res=sum(not b and q for b,q in pairs);damage=sum(b and not q for b,q in pairs)
 diff=np.array([int(q)-int(b) for b,q in pairs]);ci=np.quantile(np.random.default_rng(20261021).choice(diff,(10000,len(diff)),replace=True).mean(1),[.025,.975]).tolist() if len(diff) else None
 result['paired'][arm]={'n':len(pairs),'rescue':res,'damage':damage,'net_sr_difference':float(diff.mean()) if len(diff) else None,'paired_bootstrap95_exploratory':ci}
errors=[];traces={};decisions={}
for c in M['cells']:
 sid=c['scene_idx']
 for arm in M['arms']:
  received=[];sent=[];ds=[]
  for p in O.glob(f'safe-qam-s{sid}_{arm}-*.module.received.jsonl'):received+=lines(p)
  for p in O.glob(f'safe-qam-s{sid}_{arm}-*.module.decisions.jsonl'):ds+=lines(p)
  ds.sort(key=lambda x:x['request']);decisions[sid,arm]=ds
  for d in ds:
   expected_gate=(arm=='always' or (arm=='safe' and d['detector']['fired']))
   if d['gate']!=expected_gate:errors.append(f'gate rule mismatch {sid}/{arm}')
   if not d['gate'] and d['qam_delta_max_normalized']!=0:errors.append(f'off delta nonzero {sid}/{arm}')
   if not d['action_finite'] or not d['off_identity_verified']:errors.append(f'finite/identity failure {sid}/{arm}')
  for row in [x for x in rows if x['cell']==sid and x['arm']==arm]:
   if arm=='base' and not any(x['arm']=='base' and x['cell']==sid and x['n']==row['n'] for x in allrows):continue
   ps=list(O.glob(f'safe-qam-collect-s{sid}_{arm}_n{row["n"]}-*.module.task_trace.jsonl'));
   if arm=='base' and len(ps)==0:continue
   assert len(ps)==1,ps
   tr=lines(ps[0]);traces[sid,row['n'],arm]=tr
   if tr[-1]['event']!='complete':errors.append('incomplete trace '+str(ps[0]))
   sent += [x for x in tr if x['event']=='http_send']
  if len(rows)==60 and collections.Counter(x['payload_sha256'] for x in sent)!=collections.Counter(x['payload_sha256'] for x in received):errors.append(f'HTTP hash count mismatch {sid}/{arm}')
  result['audits'][f'{sid}/{arm}']={'sent':len(sent),'received':len(received),'decisions':len(ds),'gated':sum(x['gate'] for x in ds),'latency_ms_p50_p95':np.quantile([x['latency_ms'] for x in ds],[.5,.95]).tolist() if ds else None,'max_qam_delta_normalized':max([x['qam_delta_max_normalized'] for x in ds],default=0),'all_finite':all(x['action_finite'] for x in ds),'off_identity_verified':all(x['off_identity_verified'] for x in ds)}
for (sid,n,arm),tr in traces.items():
 if arm=='base':continue
 base=traces.get((sid,n,'base'))
 if not base:continue
 a=next(x for x in tr if x['event']=='reset');b=next(x for x in base if x['event']=='reset')
 if a['state_sha256']!=b['state_sha256'] or a['task']!=b['task']:errors.append(f'initial state/task mismatch {sid}/{n}/{arm}')
result['baseline_provenance']={'original_records':30,'prior_replay_qa_completed':17,'new_replay_qa_completed':0,'unreplayed_original_initial_full_state_hash':'unavailable; paired by fixed canonical coordinate and seed, scene-level representative exact replay checks'};result['legacy_always']={'excluded_from_primary':True,'completed':len(legacy),'success':sum(bool(x['success']) for x in legacy)};result['errors']=errors;result['claim']='Fresh10task QAM final500k; exploratory OpenDrawer30pair evaluation; no checkpoint or threshold selection from these outcomes; not full RoboCasa benchmark.'
OLD=ROOT/'outputs/eval/robocasa/groot_n15/safe_qam_opendrawer_heldout_20261021'
prior=[x for x in lines(OLD/'progress.jsonl') if x['arm']=='safe'];oldlookup={(x['cell'],x['n']):x for x in prior}
for (sid,n,arm),tr in traces.items():
 if arm!='safe':continue
 ps=list(OLD.glob(f'safe-qam-collect-s{sid}_safe_n{n}-*.module.task_trace.jsonl'));assert len(ps)==1
 previous=lines(ps[0]);a=next(x for x in tr if x['event']=='reset');b=next(x for x in previous if x['event']=='reset')
 if a['state_sha256']!=b['state_sha256'] or a['task']!=b['task']:errors.append(f'prior paired reset mismatch {sid}/{n}')
result['versus_previous_qam']={}
for sid in [None,0,1,2]:
 pairs=[(bool(oldlookup[(x['cell'],x['n'])]['success']),bool(x['success'])) for x in rows if x['arm']=='safe' and (sid is None or x['cell']==sid)]
 diff=np.array([int(q)-int(b) for b,q in pairs]);result['versus_previous_qam'][str(sid)]={'n':len(pairs),'rescue':sum(not b and q for b,q in pairs),'damage':sum(b and not q for b,q in pairs),'prior_success':sum(b for b,q in pairs),'new_success':sum(q for b,q in pairs),'paired_bootstrap95':np.quantile(np.random.default_rng(20261021).choice(diff,(10000,len(diff)),replace=True).mean(1),[.025,.975]).tolist() if len(diff) else None}
for sid in [0,1,2]:
 pairs=[(bool(x['success']),bool(lookup[(sid,x['n'],'safe')]['success'])) for x in rows if x['arm']=='base' and x['cell']==sid and (sid,x['n'],'safe') in lookup]
 result['cells'][str(sid)]['paired']={'n':len(pairs),'rescue':sum(not b and q for b,q in pairs),'damage':sum(b and not q for b,q in pairs)}
result['confound_audit']={'length':'SR per episode; gate fraction duration-sensitive, no detector accuracy claim','task':'OpenDrawer only, report each cell','instruction':'same canonical left request; per-step actual fixture/language assertions','fit_eval':'original demos only for QAM; detector targetj heldout; historical human demo reset seeds unpublished','pooling':'online per-inference feature, no future episode pooling','phase':'no phase-matched detector accuracy claim','causality':'paired intervention outcomes, no MSE-to-SR inference','generalization':'three cells only, exploratory30cases'}
result['claim_strength']='intervention effect on fixed OpenDrawer30cases; no full10task SR claim'
(O/'verification_history'/f'{VERIFY_RUN_ID}.json').write_text(json.dumps(result,indent=2));(O/'evaluation_summary_v3.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if len(rows)==60 and errors:raise SystemExit(1)
