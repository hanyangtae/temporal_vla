"""Fail-closed audit; never changes environment targets or instructions."""
import hashlib,json
class TaskGuard:
 def __init__(self,expected):self.expected=expected;self.initial=None;self.latest=None;self.steps=0
 def check(self,actual):
  for k in ['language','fixture','layout','style']:
   if actual[k]!=self.expected[k]:raise RuntimeError(f'Task mismatch {k}: {actual[k]!r} != {self.expected[k]!r}')
  if actual['observed_language']!=actual['language']:raise RuntimeError('Observation language differs from environment')
  if actual['language']!=f"Open the {actual['drawer_side']} drawer.":raise RuntimeError('drawer_side differs from language')
  if self.initial is None:self.initial=actual.copy()
  if actual!=self.initial:raise RuntimeError('Task changed within episode')
  self.latest=actual.copy()
 def sent(self,payload):
  if self.latest is None:raise RuntimeError('HTTP action before audited reset')
  if payload['task']!=self.latest['observed_language']:raise RuntimeError('HTTP language differs from latest observation')
def canonical_hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
