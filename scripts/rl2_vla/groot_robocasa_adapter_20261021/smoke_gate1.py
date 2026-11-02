"""Actual frozen GR00T + exported learned QAM diagnostic. No rollout/performance claim."""
import argparse,json,time
from pathlib import Path
import numpy as np
from extract_opendrawer_latents import (import_serve_module,FrozenGroot,checkpoint_dir,REPO_ROOT,
    PROFILE_REL,OpenDrawerPackage,state_payload_map,video_payload_map,decode_video,png_b64)
from baseline_data import write_json_exclusive,sha256_path
import composition as composition_module
from composition import ComposedFlowSampler,SPEC,FirstCallCapture
from qam_torch_actor import TorchQAMVelocity

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--actor',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    import torch
    start=time.time();sv=import_serve_module()
    model=FrozenGroot(sv,profile_path=REPO_ROOT/PROFILE_REL,pretrained=checkpoint_dir(a.data_root),device=a.device)
    model.policy.requires_grad_(False);model.policy.eval()
    pkg=OpenDrawerPackage(a.data_root);d=pkg.read(0);T=len(d['raw_actions'])
    sm=state_payload_map(pkg.modality,model.profile);vm=video_payload_map(pkg.modality)
    payload={}
    for vk,key in vm.items():payload[key]=png_b64(decode_video(pkg.root/pkg.rel(0,vk),T,d['timestamp'],pkg.fps)[0])
    for key,(lo,hi) in sm.items():payload[key]=d['state'][0,lo:hi].tolist()
    payload['task']=pkg.tasks[int(d['task_desc'][0])]
    actor=TorchQAMVelocity(a.actor,device='cpu');features=[];padding_checks=[]
    def callback(feature,noisy,t):
        features.append(feature.copy());return actor(feature,noisy,t)
    def predict():
        sv._apply_inference_seed({'inference_seed':42})
        batch=model.preprocessor(sv._apply_input_remap(sv.parse_payload(payload)))
        with torch.inference_mode():return model.policy.predict_action_chunk(batch).detach().clone()
    head=model.head;original=head.get_action;compose=composition_module.compose_velocity
    def checked_compose(vla,qam,lam):
        out=compose(vla,qam,lam)
        assert torch.equal(out[...,12:],vla[...,12:])
        padding_checks.append(True);return out
    try:
        cap=FirstCallCapture();hook=head.action_decoder.register_forward_pre_hook(cap)
        try:base=predict()
        finally:hook.remove()
        base_feature=SPEC.pool_decoder_input(cap.first).cpu().numpy()
        composition_module.compose_velocity=checked_compose
        zero=ComposedFlowSampler(SPEC,qam_fn=callback,qam_weight=.5).make_get_action(head,lambda:{'decision_env_step':0,'gate':0.})
        head.get_action=zero;gate0=predict()
        assert torch.equal(base,gate0) and actor.calls==0
        assert np.array_equal(base_feature,zero.last_info['feature'])
        one=ComposedFlowSampler(SPEC,qam_fn=callback,qam_weight=.5).make_get_action(head,lambda:{'decision_env_step':0,'gate':1.})
        head.get_action=one;gate1=predict()
        assert actor.calls==4 and one.last_info['qam_steps']==[0,1,2,3]
        assert all(np.array_equal(features[0],x) for x in features)
        assert torch.isfinite(gate1).all()
        repeated=predict();assert torch.equal(gate1,repeated) and actor.calls==8
        assert all(not p.requires_grad for p in model.policy.parameters())
        result={'status':'passed','actor_metadata_sha256':sha256_path(a.actor/'actor.json'),
            'actor_step':actor.meta['source_checkpoint']['step'],'diagnostic_only':actor.meta['diagnostic_only'],
            'gate0_bitwise_equal':True,'gate0_feature_equal':True,'gate0_qam_calls':0,
            'gate1_qam_calls_per_prediction':4,'gate1_first_feature_frozen':True,
            'gate1_repeated_bitwise_equal':True,'gate1_finite':True,'padding_velocity_checks':len(padding_checks),
            'action_shape':list(base.shape),'gate1_vs_base_max_abs':float((gate1-base).abs().max()),
            'frozen_base_parameters':True,'qam_weight':.5,'seconds':time.time()-start,
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'rollout_evaluation_performed':False}
        a.out.mkdir(parents=True,exist_ok=True);write_json_exclusive(a.out/'result.json',result)
        print(json.dumps(result))
    finally:
        head.get_action=original;composition_module.compose_velocity=compose
if __name__=='__main__':main()
