"""Replay frozen M2 step2100 with original AMP and FP32 descriptor head only."""
import sys,json,types,time
from pathlib import Path
import numpy as np,torch,cv2
OLD=Path('/home/yl/matching/sar_matching/SAR_SP_LG_Train_20261012_1544')
sys.path.insert(0,str(OLD/'code'))
from common import ensure_gpu,readrows,atomic_json,now,sha
from model import SARModel,loss
from data import pair
from checkpoint import restore
OUT=Path(__file__).parent
def fp32_dense(self,image):
 s=self.sp
 with torch.no_grad():
  x=s.relu(s.conv1a(image));x=s.relu(s.conv1b(x));x=s.pool(x);x=s.relu(s.conv2a(x));x=s.relu(s.conv2b(x));x=s.pool(x);x=s.relu(s.conv3a(x));x=s.relu(s.conv3b(x));x=s.pool(x);x=s.relu(s.conv4a(x));x=s.relu(s.conv4b(x))
  base=torch.nn.functional.normalize(self.teacherDb(s.relu(self.teacherDa(x))),dim=1)
 with torch.autocast('cuda',enabled=False):
  learned=torch.nn.functional.normalize(s.convDb(s.relu(s.convDa(x.float()))),dim=1)
 return learned,base
def run(policy):
 cfg=json.loads((OLD/'configs/m2.json').read_text());model=SARModel('m2').cuda()
 if policy=='fp32_descriptor_head':model.dense=types.MethodType(fp32_dense,model)
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
 sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['max_steps'],eta_min=cfg['learning_rate']*.1);scaler=torch.amp.GradScaler('cuda')
 c=restore(OLD/'runs/m2/last.ckpt',model,opt,sched,scaler,cfg);assert c['global_step']==2100
 pools={}
 for r in readrows(OLD/'train_sources.jsonl'):
  category='nearshore' if r['scene_type']=='inshore' else 'background' if r['dataset']=='background' else 'offshore'
  pools.setdefault(category,{}).setdefault(r['training_component'],[]).append(r)
 rows=[];start=time.perf_counter()
 for step in range(2100,2160):
  model.train();opt.zero_grad(set_to_none=True);values=[]
  for micro in range(cfg['batch_accumulation']):
   for attempt in range(20):
    seed=cfg['seed']+step*200+micro*20+attempt;rng=np.random.default_rng(seed);category=str(rng.choice(['nearshore','offshore','background'],p=[.55,.35,.10]));groups=pools[category];group=sorted(groups)[int(rng.integers(len(groups)))];r=groups[group][int(rng.integers(len(groups[group])))];regime=str(rng.choice(['easy','medium','low'],p=[.2,.6,.2]))
    z=pair(r,seed,regime,dynamic_occlusion=bool(r['annotations']) and rng.random()<.15);item=loss(model,z,'cuda',cfg['temperature'],cfg['preserve_weight'])
    if item is not None:break
   else:raise RuntimeError('No positives')
   value,_=item;assert torch.isfinite(value);values.append(float(value.detach()));scaler.scale(value/cfg['batch_accumulation']).backward()
  opt_params=[p for p in model.parameters() if p.requires_grad];bad=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
  scale=scaler.get_scale();scaler.unscale_(opt);grad=torch.nn.utils.clip_grad_norm_(opt_params,1.)
  row=dict(global_step_before_update=step,loss=float(np.mean(values)),scale=scale,finite=bool(torch.isfinite(grad)),bad_gradient_parameters=bad);rows.append(row)
  if not row['finite']:break
  scaler.step(opt);scaler.update();sched.step()
 result=dict(policy=policy,parent_checkpoint_sha256=sha(OLD/'runs/m2/last.ckpt'),at=now(),steps=rows,all_finite=all(x['finite'] for x in rows),elapsed_seconds=time.perf_counter()-start)
 atomic_json(OUT/(policy+'.json'),result);print(policy,json.dumps(result),flush=True)
 del model,opt,sched,scaler;torch.cuda.empty_cache()
 return result
if __name__=='__main__':
 ensure_gpu(1);torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False
 a=run('original_amp');b=run('fp32_descriptor_head')
 assert not a['all_finite'],'Original failure not reproduced; do not claim cause'
 assert b['all_finite'] and len(b['steps'])==60,'Numerical repair insufficient'
 print('REPLAY_COMPARISON_PASSED',flush=True)
