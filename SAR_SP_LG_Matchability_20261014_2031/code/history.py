"""Deterministic historical sampler replay. Reconstructed coverage is labelled as replay, not original logging."""
from common import *
from data import pair as old_pair
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
def audit(name):
 sources=readrows(OLD/'train_sources.jsonl');pool=defaultdict(lambda:defaultdict(list))
 for r in sources:
  cat='nearshore' if r['scene_type']=='inshore' else ('background' if r['dataset']=='background' else 'offshore');pool[cat][component(r)].append(r)
 spec={'m1':(OLD/'configs/m1.json',2700,'retry','adapter only'),'m2':(FIX/'configs/m2.json',5100,'retry','convDa/convDb only'),'m3':(BASE/'SAR_SP_LG_M3_Detector_20261012_1932/config.json',2250,'direct','convPa/convPb only'),'a':(STRUCT/'config_20261013_2210.json',3000,'direct','structure branch only'),'b':(STRUCT/'config_20261013_2210.json',3000,'direct','structure + convPa/convPb'),'control':(STRUCT/'config_20261013_2210.json',2000,'direct','convDa/convDb only')}
 cfgpath,steps,mode,scope=spec[name];cfg=json.loads(cfgpath.read_text());cnt=Counter();attempted=Counter();failures=0
 for step in range(steps):
  for micro in range(2):
   for attempt in range(20 if mode=='retry' else 1):
    seed=cfg['seed']+(step*200+micro*20+attempt if mode=='retry' else step*20+micro);rng=np.random.default_rng(seed);cat=str(rng.choice(['nearshore','offshore','background'],p=[.55,.35,.1]));cs=sorted(pool[cat]);rs=pool[cat][cs[int(rng.integers(len(cs)))]];r=rs[int(rng.integers(len(rs)))];key=r['dataset']+':'+r['source_id'];attempted[key]+=1
    if mode=='retry':
     regime='easy' if step<600 else ('medium' if step<1800 else str(rng.choice(['easy','medium','low'],p=[.2,.6,.2])));dyn=step>=1800 and bool(r['annotations']) and rng.random()<.15;z=old_pair(r,seed,regime,dynamic_occlusion=dyn)
     if len(z['points0'])<16:failures+=1;continue
    cnt[key]+=1;break
   else:raise RuntimeError((name,step,micro,'no accepted replay'))
  if step%1000==0:print(name,'replay',step,flush=True)
 ckpath=(OLD if name=='m1' else FIX if name=='m2' else BASE/'SAR_SP_LG_M3_Detector_20261012_1932' if name=='m3' else STRUCT)/'runs'/name/'last.ckpt'
 ck=torch.load(ckpath,map_location='cpu',weights_only=False);optimizer_steps=sorted({int(v['step']) for v in ck['optimizer_state']['state'].values() if 'step' in v})
 return dict(model=name,trainable_scope=scope,shared_backbone_trained=False,LightGlue_trained=False,available_source_records=5111,global_step=steps,optimizer_steps=optimizer_steps,accepted_pair_calls=sum(cnt.values()),view_instances=2*sum(cnt.values()),attempted_pair_calls=sum(attempted.values()),retry_rejections=failures,unique_accepted_source_records=len(cnt),unique_attempted_records=len(attempted),source_coverage=cnt,reconstruction='Seed/loop replay of frozen sampler and pair acceptance; not original per-source log; failed pre-restart numerical runs excluded',config_sha=sha(cfgpath),checkpoint_sha=sha(ckpath))
def main():
 torch.set_num_threads(1);cv2.setNumThreads(1)
 with ProcessPoolExecutor(max_workers=2) as ex:rr=list(ex.map(audit,['m1','m2','m3','a','b','control']))
 write_new(output('historical_training_audit'),dict(rows=rr,all_LightGlue_frozen=True,all_shared_backbones_pretrained_only=True));export_csv(output('historical_training_summary',ext='csv'),[{k:v for k,v in r.items() if k!='source_coverage'} for r in rr]);print('HISTORY REPLAY COMPLETE')
if __name__=='__main__':main()
