import ast,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rl2_vla/stage2_cluster_reward'))
from export_training_cache import feature,COMMANDS

def test_feature_reduction_matches_upstream():
 import torch
 source=Path(__file__).resolve().parents[1]/'RL2-VLA/third_party/SAFE/failure_prob/data/utils.py'
 nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('parse_and_index_tensor_last','process_tensor_idx_rel')]
 ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
 x=np.arange(3*1*10*5*1024,dtype=np.float32).reshape(3,1,10,5,1024)
 for h in COMMANDS:
  for d in COMMANDS:
   y=torch.from_numpy(x[:,0,:,1:,:])
   for c in (h,d):y=ns['process_tensor_idx_rel'](y,float(c) if c in ('0.0','1.0') else c)
   np.testing.assert_array_equal(feature(x,h,d),y.numpy())
