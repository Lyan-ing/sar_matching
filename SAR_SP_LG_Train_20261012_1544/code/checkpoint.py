import os,torch,random,numpy as np
from pathlib import Path
from common import now
def atomic_save(path,content):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
 with temp.open('wb') as f:torch.save(content,f);f.flush();os.fsync(f.fileno())
 os.replace(temp,path)
def pack(model,optimizer,scheduler,scaler,step,config,identity,best,current):
 return dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict(),amp_scaler=scaler.state_dict(),epoch=step*config['batch_accumulation']/config['train_sources'],global_step=step,random_state=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),config=config,data_identity=identity,variant=model.variant,model_structure='M1:256-64-256 residual alpha0.1; M2:convDa/convDb only; SP detection/shared and LG frozen',best_validation=best,current_validation=current,created_at=now())
def restore(path,model,optimizer,scheduler,scaler,config):
 c=torch.load(path,map_location='cuda',weights_only=False);assert c['variant']==model.variant and c['config']==config,'Frozen architecture/config mismatch'
 model.load_state_dict(c['model_state']);optimizer.load_state_dict(c['optimizer_state']);scheduler.load_state_dict(c['scheduler_state']);scaler.load_state_dict(c['amp_scaler'])
 random.setstate(c['random_state']['python']);np.random.set_state(c['random_state']['numpy']);torch.set_rng_state(c['random_state']['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in c['random_state']['cuda']]);return c
