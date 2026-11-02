import json,os,time
from pathlib import Path
import serve_qam
from task_language_guard import canonical_hash
orig=serve_qam.import_serve_module
stream=Path(os.environ['PDK_LOG_FILE']).with_suffix('.received.jsonl').open('x',buffering=1)
def load():
 m=orig();apply0=m._apply_inference_seed
 def apply(payload):
  if 'task' in payload:stream.write(json.dumps({'task':payload['task'],'payload_sha256':canonical_hash(payload),'inference_seed':payload.get('inference_seed'),'time':time.time()})+'\n')
  return apply0(payload)
 m._apply_inference_seed=apply;return m
serve_qam.import_serve_module=load
try:serve_qam.main()
finally:stream.close()
