"""Controlled fixed-pair transfer; isolated checkpoints, exact sampler and resume state."""
from mixed_io import *
from joint_objective import scope,objective,load_item
from joint_evaluate import run as evaluate
import argparse,copy,math,signal

CORE=['runtime.py','mixed_io.py','joint_objective.py','joint_sampling.py','joint_train.py','joint_evaluate.py']
def code_identity():return {name:sha(ROOT/'code'/name) for name in CORE}
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(s):random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])
def atomic_checkpoint(path,state):
 tmp=path.with_name(path.name+'.tmp');path.parent.mkdir(parents=True,exist_ok=True)
 with tmp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

class Bank:
 def __init__(self,rows,indices):
  self.rows=rows;self.entries={}
  for i in sorted(set(indices)):
   row=rows[i];q=torch.load(ROOT/'cache/train'/f"{row['pair_id']}_{TAG}.pt",map_location='cpu',weights_only=False)
   assert q['image_hash']==row['image_file_hashes'];assert q['x'].dtype==torch.float32
   self.entries[i]=(q['x'],q['responses'],load_item(row))
 def batch(self,ids):
  xs,rs,ss=zip(*(self.entries[i] for i in ids));return torch.cat(xs).cuda(),torch.cat(rs).cuda(),ss

def terms(m,teacher,original,x,rs,items,amp=True):
 with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
  logits,dense=m.cached_maps(x,rs)
  with torch.no_grad():
   _,td=teacher.cached_maps(x,rs);s=original.sp;tl=s.convPb(s.relu(s.convPa(x)))
 # Cross entropy, probabilities, geometry, interpolation and normalization use FP32.
 losses=[];infos=[]
 with torch.autocast('cuda',enabled=False):
  for j,item in enumerate(items):
   loss,info=objective(logits[2*j:2*j+2].float(),dense[2*j:2*j+2].float(),tl[2*j:2*j+2].float(),td[2*j:2*j+2].float(),item,int(x.shape[-1]*8));losses.append(loss);infos.append(info)
 return torch.stack(losses).mean(),infos

def optimizer(m):
 groups=[]
 for prefixes,lr in [(('structure.',),1e-4),(('sp.convP',),1e-5),(('sp.convD',),1e-5)]:
  ps=[p for n,p in m.named_parameters() if p.requires_grad and n.startswith(prefixes)]
  if ps:groups.append(dict(params=ps,lr=lr))
 return torch.optim.AdamW(groups,weight_decay=1e-4)

def rank(metrics):
 a=metrics['all'];return (a['Success@5px'],a['structure_anchor_recall'] or 0,a['structure_repeatability'] or 0)

def calibration_probes(m):
 result={};rng=np.random.default_rng(42)
 for key,im in [('black',np.zeros((256,256),np.uint8)),('gray',np.full((256,256),128,np.uint8)),('noise',rng.integers(0,256,(256,256),dtype=np.uint8))]:
  with torch.no_grad():result[key]=len(m.extract(torch.tensor(im,device='cuda').float()[None,None]/255)['keypoints'][0])
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['t1','t2','t3'],required=True);p.add_argument('--run',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--plan',type=Path,required=True);p.add_argument('--steps',type=int,default=6000);p.add_argument('--resume',type=Path);p.add_argument('--validation-every',type=int,default=2000);a=p.parse_args()
 identity=setup(a.seed);rows=records('train');plan=[json.loads(s) for s in a.plan.read_text().splitlines()];assert len(plan)==a.steps
 folder=ROOT/'runs'/a.run;folder.mkdir(parents=True,exist_ok=True);runstamp=stamp();status=folder/f'status_{TAG}.json';stop=False
 def stopped(sig,frame):
  nonlocal stop;stop=True
 signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
 m=Model('b').cuda();scope(m,a.kind);teacher=Model('b').cuda().eval().requires_grad_(False);original=Model('m0').cuda().eval().requires_grad_(False)
 opt=optimizer(m);scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:.1+.9*(1+math.cos(math.pi*min(step,a.steps)/a.steps))/2);scaler=torch.amp.GradScaler('cuda',enabled=False)
 config=dict(kind=a.kind,run=a.run,seed=a.seed,steps=a.steps,effective_batch=8,microbatch=2,gradient_accumulation=4,precision='BF16 convolutions; FP32 losses/normalization; no FP16 scaler',sampler=str(a.plan),sampler_sha=sha(a.plan),initialization=str(WEIGHTS['b']),initialization_sha=sha(WEIGHTS['b']),train_manifest_sha=sha(MANIFEST['train']),val_manifest_sha=sha(MANIFEST['val']),code_sha=code_identity(),trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),trainable_names=[n for n,p in m.named_parameters() if p.requires_grad],objective='joint_objective.py fixed coefficients',validation_every=a.validation_every,selection='Among nonzero-step candidates: Natural all-geometric S5 primary; anchor recall then repeatability; qualified only if black/gray=0, noise<=min(initial B, 2*historical descriptor baseline+8), FPR<=baseline B and offshore S5>=baseline B-0.02. Keep diagnostic trained best separately from qualified best; never relabel step0 as newly trained.')
 best=None;qualified=None;startstep=0;history=[]
 frozen={n:v.detach().cpu().clone() for n,v in m.state_dict().items() if n not in config['trainable_names']}
 if a.resume:
  ck=torch.load(a.resume,map_location='cpu',weights_only=False);assert ck['config']['code_sha']==config['code_sha'];assert ck['config']['sampler_sha']==config['sampler_sha'];assert ck['config']['kind']==a.kind
  m.load_state_dict(ck['model_state']);opt.load_state_dict(ck['optimizer_state']);scheduler.load_state_dict(ck['scheduler_state']);scaler.load_state_dict(ck['amp_scaler']);startstep=ck['global_step'];best=ck['best'];qualified=ck['qualified'];history=ck['validation_history'];restore_rng(ck['random_state'])
 else:write(folder/f'config_{runstamp}.json',dict(config,identity=identity))
 baseline=json.loads(sorted(ROOT.glob('summary_baseline_*.json'))[-1].read_text())['b/natural'];baselineprobes=calibration_probes(teacher);referenceprobes=calibration_probes(Model('m2').cuda());noise_limit=min(baselineprobes['noise'],2*referenceprobes['noise']+8)
 def checkpoint(step):
  assert all(torch.equal(m.state_dict()[k].detach().cpu(),v) for k,v in frozen.items())
  return dict(model_state=m.state_dict(),optimizer_state=opt.state_dict(),scheduler_state=scheduler.state_dict(),amp_scaler=scaler.state_dict(),global_step=step,epoch=None,random_state=rng(),best=best,qualified=qualified,validation_history=history,config=config,identity=identity,frozen_parameters_unchanged=True,initialization_sha=config['initialization_sha'])
 print(now(),'loading fixed FP32 train embeddings',flush=True);bank=Bank(rows,[i for s in plan for i in s['active']]);print(now(),'bank ready',len(bank.entries),flush=True)
 metrics=folder/f'metrics_{runstamp}.jsonl';usage=folder/f'sampling_actual_{runstamp}.jsonl';start=time.perf_counter();step=startstep
 try:
  for entry in plan[startstep:]:
   if stop or (folder/'STOP_REQUESTED').exists():break
   m.train();opt.zero_grad(set_to_none=True);infos=[];stepstart=time.perf_counter()
   for j in range(0,8,2):
    ids=entry['active'][j:j+2];x,rs,items=bank.batch(ids);loss,inf=terms(m,teacher,original,x,rs,items);assert torch.isfinite(loss),'nonfinite loss';(loss/4).backward();infos+=inf
   grads=[p.grad for p in m.parameters() if p.requires_grad];assert all(g is not None and torch.isfinite(g).all() for g in grads),'missing/nonfinite trainable gradient'
   gn=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],5.);opt.step();scheduler.step();step=entry['step'];torch.cuda.synchronize()
   record=dict(step=step,updated=now(),loss=float(np.mean([i['loss'] for i in infos])),descriptor=float(np.mean([i['descriptor'] for i in infos])),detector=float(np.mean([i['detector_anchor'] for i in infos])),KL=float(np.mean([i['teacher_KL'] for i in infos])),coordinate=float(np.mean([i['coordinate_equivariance'] for i in infos])),local_R1=float(np.mean([i['local_R1'] for i in infos])),gradient_norm=float(gn),lr=[g['lr'] for g in opt.param_groups],points=sum(i['points'] for i in infos),positive_only_points=sum(i['positive_only'] for i in infos),effective_pairs=8,seconds=time.perf_counter()-stepstart,peak_memory_MiB=torch.cuda.max_memory_allocated()/2**20)
   with metrics.open('a') as f:f.write(json.dumps(record)+'\n')
   with usage.open('a') as f:f.write(json.dumps(dict(step=step,size=entry['size'],consumed=entry['consumed'],active=entry['active'],pair_ids=[rows[i]['pair_id'] for i in entry['active']],supervision=infos))+'\n')
   if step%25==0 or step==1:
    write(status,dict(state='training',**identity,step=step,total_steps=a.steps,record=record,best=best,qualified=qualified,updated=now()));print(now(),a.run,step,a.steps,round(record['loss'],5),round(record['seconds'],3),flush=True)
   if step%500==0:atomic_checkpoint(folder/f'last_{TAG}.ckpt',checkpoint(step))
   if step%a.validation_every==0 or step==a.steps:
    m.eval();write(status,dict(state='validation',**identity,step=step,total_steps=a.steps,updated=now()))
    tag=f'{a.run}_step{step}';rr,summ=evaluate({a.run:m},tag,natural=True,equal=False,save=True);metric=summ[a.run+'/natural'];probes=calibration_probes(m)
    offshore='scene/offshore_vessels_low_texture_sea';ok=bool(probes['black']==0 and probes['gray']==0 and probes['noise']<=noise_limit and metric['all']['no_overlap_FPR']<=baseline['all']['no_overlap_FPR'] and metric[offshore]['Success@5px']>=baseline[offshore]['Success@5px']-.02)
    item=dict(step=step,metrics=metric,probes=probes,qualified=ok,selection_rank=rank(metric));history.append(item)
    improved=best is None or rank(metric)>tuple(best['rank']);qimproved=ok and (qualified is None or rank(metric)>tuple(qualified['rank']))
    if improved:best=dict(step=step,rank=rank(metric),qualified=ok)
    if qimproved:qualified=dict(step=step,rank=rank(metric))
    ck=checkpoint(step);atomic_checkpoint(folder/f'checkpoint_step{step}_{TAG}.ckpt',ck)
    if improved:atomic_checkpoint(folder/f'best_trained_{TAG}.ckpt',ck)
    if qimproved:atomic_checkpoint(folder/f'best_qualified_{TAG}.ckpt',ck)
    atomic_checkpoint(folder/f'last_{TAG}.ckpt',ck);write(folder/f'validation_step{step}_{stamp()}.json',item);print(now(),a.run,'VALIDATION',step,metric['all'],probes,'qualified',ok,flush=True)
    assert code_identity()==config['code_sha'],'Training code changed during run'
  atomic_checkpoint(folder/f'last_{TAG}.ckpt',checkpoint(step));write(status,dict(state='completed' if step==a.steps else 'stopped',**identity,step=step,total_steps=a.steps,best=best,qualified=qualified,elapsed_seconds=time.perf_counter()-start,updated=now(),last=str(folder/f'last_{TAG}.ckpt'),best_checkpoint=str(folder/f'best_trained_{TAG}.ckpt'),frozen_parameters_unchanged=True))
 except BaseException as exc:
  atomic_checkpoint(folder/f'failure_step{step}_{stamp()}.ckpt',checkpoint(step));write(status,dict(state='failed',**identity,step=step,error=repr(exc),updated=now()));raise
if __name__=='__main__':main()
