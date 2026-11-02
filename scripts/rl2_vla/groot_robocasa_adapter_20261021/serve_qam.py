"""Localhost wrapper for existing LeRobot serve; upstream files are not changed."""
import argparse,sys
from pathlib import Path
from extract_opendrawer_latents import import_serve_module
from composition import ComposedFlowSampler,SPEC
from qam_torch_actor import TorchQAMVelocity

def main():
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument('--qam-arm',choices=['base','gate0','gate1'],required=True)
    ap.add_argument('--qam-actor',type=Path);ap.add_argument('--qam-weight',type=float,default=.5)
    a,rest=ap.parse_known_args()
    if a.qam_arm!='base' and a.qam_actor is None:ap.error('actor required')
    if '--host' in rest and rest[rest.index('--host')+1]!='127.0.0.1':ap.error('localhost only')
    if '--host' not in rest:rest+=['--host','127.0.0.1']
    sv=import_serve_module()
    from lerobot_adapters.groot import GrootPolicyAdapter
    original_load=GrootPolicyAdapter.load
    state={'arm':a.qam_arm,'weight':a.qam_weight,'ready':False,'decision_calls':0,'qam_calls':0,
           'feature_signature':SPEC.signature()}
    def load(self,*args,**kwargs):
        loaded=original_load(self,*args,**kwargs);loaded.policy.requires_grad_(False);loaded.policy.eval()
        if a.qam_arm!='base':
            actor=TorchQAMVelocity(a.qam_actor,device='cpu')
            state['actor']=actor.meta
            def velocity(feature,noisy,t):
                out=actor(feature,noisy,t);state['qam_calls']=actor.calls;return out
            def context():
                # Logical decision counter; fresh feature is always captured inside this call.
                idx=state['decision_calls'];state['decision_calls']+=1
                return {'decision_env_step':idx*5,'gate':float(a.qam_arm=='gate1')}
            head=loaded.policy._groot_model.action_head
            head.get_action=ComposedFlowSampler(SPEC,qam_fn=velocity,qam_weight=a.qam_weight).make_get_action(head,context)
        state['ready']=True
        return loaded
    GrootPolicyAdapter.load=load
    @sv.app.get('/qam_contract')
    def contract():return state
    sys.argv=[str(Path(sv.__file__)),*rest]
    sv.main()
if __name__=='__main__':main()
