"""Finite descriptor-only optimization, with complete state and per-loss gradients."""
from keep_data import *
from keep_objective import configure,terms
from keep_evaluate import run as evaluate,frozen_rows
import math,copy,signal,argparse


CORE=['runtime.py','mixed_io.py','keep_data.py','keep_objective.py','keep_train.py','keep_evaluate.py','old_objective.py']
def code_identity():return {n:sha(ROOT/'code'/n) for n in CORE}
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(s):random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda']])
def save(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def optim(model):return torch.optim.AdamW(configure(model),lr=1e-5,weight_decay=1e-4)
def schedule(opt):return torch.optim.lr_scheduler.LambdaLR(opt,lambda s:.1+.9*(1+math.cos(math.pi*min(s,6000)/6000))/2)
def vector(model):return torch.cat([p.grad.detach().float().reshape(-1) for p in model.parameters() if p.requires_grad])
def frozen_check(model,teacher):
    train={n for n,p in model.named_parameters() if p.requires_grad}
    assert train=={'sp.convDa.weight','sp.convDa.bias','sp.convDb.weight','sp.convDb.bias'}
    assert all(torch.equal(v,teacher.state_dict()[n]) for n,v in model.state_dict().items() if n not in train)
    assert all(p.grad is None for p in teacher.parameters())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def update(model,teacher,opt,scheduler,bank,entry,weight,amp=True):
    model.train();opt.zero_grad(set_to_none=True);infos=[];gd=None;start=time.perf_counter()
    for j in range(0,8,2):
        x,rs,items=bank.batch(entry['ids'][j:j+2]);roles=entry['roles'][j:j+2];d,k,inf=terms(model,teacher,x,rs,items,roles,amp)
        loss=(d+weight*k)/2;assert torch.isfinite(loss);loss.backward();infos+=inf
        if j==2:gd=vector(model).clone()
    total=vector(model);assert torch.isfinite(total).all();gk=(total-gd)/weight if weight else torch.zeros_like(total)
    nd=float(torch.linalg.vector_norm(gd));nk=float(torch.linalg.vector_norm(gk));cos=float(torch.dot(gd,gk)/(torch.linalg.vector_norm(gd)*torch.linalg.vector_norm(gk)+1e-12))
    gn=float(torch.nn.utils.clip_grad_norm_(configure_parameters(model),5.));opt.step();scheduler.step();torch.cuda.synchronize()
    ds=[r['desc'] for r in infos if r['structure_pairs']];ks=[r['keep'] for r in infos if r['keep_pairs']]
    return dict(step=entry['step'],desc=float(np.mean(ds)),keep=float(np.mean(ks)),loss=float(np.mean(ds))+weight*float(np.mean(ks)),gradient_desc=nd,gradient_keep=nk,weighted_keep_gradient=weight*nk,gradient_cosine=cos,preclip_gradient=gn,structure_pairs=sum(r['structure_pairs'] for r in infos),keep_pairs=sum(r['keep_pairs'] for r in infos),points=sum(r['points'] for r in infos),keep_positions=sum(r['keep_positions'] for r in infos),contrastive_queries=sum(r['contrastive_queries'] for r in infos),positive_only=sum(r['positive_only'] for r in infos),R1=float(np.mean([r['descriptor_R1'] for r in infos if r['descriptor_R1'] is not None])),lr=opt.param_groups[0]['lr'],seconds=time.perf_counter()-start,supervision=infos)


def configure_parameters(model):return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def drift(model,teacher,bank,indices):
    rows=[]
    for i in indices:
        x,rs,items=bank.batch([i]);_,k,inf=terms(model,teacher,x,rs,items,['keep']);rows.append(dict(index=i,pair_id=bank.rows[i]['pair_id'],scene=bank.rows[i]['source_scene'],cosine_drift=float(k),positions=inf[0]['keep_positions']))
    return rows


def probes(model):
    out={};g=np.random.default_rng(42)
    for size in [256,512,640]:
        for name,a in [('black',np.zeros((size,size),np.uint8)),('gray',np.full((size,size),128,np.uint8)),('noise',g.integers(0,256,(size,size),dtype=np.uint8))]:
            with torch.no_grad():f=model.extract(torch.tensor(a,device='cuda').float()[None,None]/255)
            out[f'{size}/{name}']=dict(count=len(f['keypoints'][0]),coordinates=f['keypoints'][0].cpu().tolist())
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',type=Path);args=parser.parse_args()
    identity=setup(42);rows=records('train');path=ROOT/f'plans/sampling_seed42_{TAG}.jsonl';plan=[json.loads(s) for s in path.read_text().splitlines()];assert len(plan)==6000
    mechanism=json.loads((ROOT/f'MECHANISM_ACCEPTED_{TAG}.json').read_text());assert mechanism['passed'];weight=mechanism['lambda_keep']
    folder=ROOT/'runs/formal';folder.mkdir(parents=True,exist_ok=True)
    if not args.resume:assert not list(folder.glob('status_*.json')),'Refuse to overwrite an existing training run'
    else:
        state=json.loads((folder/f'status_{TAG}.json').read_text());assert state['state']!='completed','Do not resume completed training'
        oldpid=state.get('pid');proc=Path(f'/proc/{oldpid}/stat')
        assert not proc.exists() or proc.read_text().rsplit(')',1)[1].split()[19]!=str(state.get('process_start_ticks')),'Previous training process is still alive'
    model=Model('b').cuda();teacher=Model('b').cuda().eval().requires_grad_(False);opt=optim(model);scheduler=schedule(opt);frozen_check(model,teacher)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in teacher.state_dict().items())
    # Verify historical structure initialization really contains the historical descriptor head.
    original=torch.load(WEIGHTS['m2'],map_location='cpu',weights_only=False)['model_state'];assert all(torch.equal(v.detach().cpu(),original[k]) for k,v in model.state_dict().items() if k.startswith('sp.convD'))
    config=dict(seed=42,steps=6000,effective_batch=8,microbatch=2,accumulation=4,structure_pairs=4,keep_pairs=4,lambda_keep=weight,learning_rate=1e-5,weight_decay=1e-4,clip_norm=5.,scheduler='cosine floor0.1 over6000',precision='BF16 convolutions; FP32 normalized descriptors/loss; scaler disabled',initialization=str(WEIGHTS['b']),initialization_sha=sha(WEIGHTS['b']),teacher_sha=sha(WEIGHTS['b']),lightglue_sha=sha(WEIGHTS['lg']),trainable_parameters=360960,trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],data={k:dict(path=str(p),sha=sha(p)) for k,p in MANIFEST.items()},sampler=str(path),sampler_sha=sha(path),code_sha=code_identity(),descriptor_objective='old reliable CE/positive-only, same q/temperature0.1/negative exclusion8px/local weight; old0.1 anchor preserve removed',keep_objective='mean(1-cos) same-input teacher/student on explicit legal uniform positions; no new match labels',validation_every=2000,selection='nonzero-step Natural geometric S5 primary, then offshore S5, then structure S5; best candidate and qualified candidate separate',qualification='Natural gain>=0.01; offshore>=initial-0.02; structure>=initial; Equal>=initial-0.01; positive wrong accept rate<=initial+0.005; no-overlap FPR<=initial; detector/probes unchanged')
    if not args.resume:write(ROOT/f'FORMAL_PROTOCOL_{TAG}.json',config);write(folder/f'config_{TAG}.json',dict(config,identity=identity))
    write(folder/f'status_{TAG}.json',dict(state='loading_features',step=0,total_steps=6000,**identity,updated=now()))
    audit=inventory();probeids=mechanism['probe_indices'];bank=Bank(rows,[i for e in plan for i in e['ids']]+probeids,teacher)
    base={mode:scorer.summaries([r for r in frozen_rows() if r['method']=='b' and r['protocol']==mode]) for mode in ['natural','equal']};baseprobes=probes(teacher);best=None;qualified=None;history=[];step=0;stop=False
    assert all(v['count']==0 for k,v in baseprobes.items() if k.endswith(('/black','/gray')))
    if not args.resume:write(ROOT/f'DETECTOR_BASELINE_PROBES_{TAG}.json',baseprobes)
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=False);assert ck['config']['code_sha']==config['code_sha'] and ck['config']['sampler_sha']==config['sampler_sha'] and ck['config']['lambda_keep']==weight
        model.load_state_dict(ck['model_state']);opt.load_state_dict(ck['optimizer_state']);scheduler.load_state_dict(ck['scheduler_state']);restore_rng(ck['random_state']);step=ck['global_step'];best=ck['best'];qualified=ck['qualified'];history=ck['validation_history'];frozen_check(model,teacher)
    def stopping(sig,frame):
        nonlocal stop;stop=True
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    def checkpoint():
        frozen_check(model,teacher)
        return dict(model_state=model.state_dict(),optimizer_state=opt.state_dict(),scheduler_state=scheduler.state_dict(),amp_scaler={},global_step=step,random_state=rng(),config=config,identity=identity,best=best,qualified=qualified,validation_history=history,frozen_parameters_unchanged=True)
    status=folder/f'status_{TAG}.json';start=time.perf_counter();write(status,dict(state='training',step=0,total_steps=6000,**identity,updated=now()))
    try:
        logtag=TAG if not args.resume else 'resume_'+stamp()
        for entry in plan[step:]:
            if stop or (folder/'STOP_REQUESTED').exists():break
            metric=update(model,teacher,opt,scheduler,bank,entry,weight);step=entry['step'];assert metric['structure_pairs']==4 and metric['keep_pairs']==4
            metric.update(updated=now(),peak_memory_MiB=torch.cuda.max_memory_allocated()/2**20)
            with (folder/f'metrics_{logtag}.jsonl').open('a') as f:f.write(json.dumps(clean(metric))+'\n')
            usage=dict(step=step,ids=entry['ids'],roles=entry['roles'],pairs=[dict(audit[i],gradient_role=role,positions=inf['keep_positions'],points_used=inf['points']) for i,role,inf in zip(entry['ids'],entry['roles'],metric['supervision'])])
            with (folder/f'sampling_actual_{logtag}.jsonl').open('a') as f:f.write(json.dumps(usage)+'\n')
            if step%25==0 or step==1:write(status,dict(state='training',step=step,total_steps=6000,metric=metric,best=best,**identity,updated=now()));print(now(),'FORMAL',step,round(metric['desc'],5),round(metric['keep'],6),flush=True)
            if step%500==0:save(folder/f'last_{TAG}.ckpt',checkpoint())
            if step%2000==0:
                model.eval();write(status,dict(state='validation',step=step,**identity,updated=now()));rr,summ=evaluate(model,f'formal_step{step}')
                np_,eq=summ['natural'],summ['equal'];pos=[r for r in rr if r['protocol']=='natural' and r['geometry_eligible']];wrong=sum(r['accepted'] and not r['Success@5px'] for r in pos)/len(pos);probe=probes(model);assert probe==baseprobes
                off='scene/offshore_vessels_low_texture_sea';score=(np_['all']['Success@5px'],np_[off]['Success@5px'],np_['all']['StructureSuccess@5px'])
                ok=bool(score[0]>=base['natural']['all']['Success@5px']+.01 and score[1]>=base['natural'][off]['Success@5px']-.02 and score[2]>=base['natural']['all']['StructureSuccess@5px'] and eq['all']['Success@5px']>=base['equal']['all']['Success@5px']-.01 and wrong<=.005 and np_['all']['no_overlap_FPR']<=base['natural']['all']['no_overlap_FPR'])
                item=dict(step=step,metrics=summ,rank=score,qualified=ok,positive_wrong_accept_rate=wrong,probes_identical=True,drift=drift(model,teacher,bank,probeids));history.append(item)
                improved=best is None or score>tuple(best['rank']);qi=ok and (qualified is None or score>tuple(qualified['rank']))
                if improved:best=dict(step=step,rank=score,qualified=ok)
                if qi:qualified=dict(step=step,rank=score)
                ck=checkpoint();save(folder/f'checkpoint_step{step}_{TAG}.ckpt',ck)
                if improved:save(folder/f'best_trained_{TAG}.ckpt',ck)
                if qi:save(folder/f'best_qualified_{TAG}.ckpt',ck)
                save(folder/f'last_{TAG}.ckpt',ck);write(folder/f'validation_step{step}_{stamp()}.json',item);print(now(),'VALIDATION',step,score,'qualified',ok,flush=True)
                assert code_identity()==config['code_sha']
        save(folder/f'last_{TAG}.ckpt',checkpoint());write(status,dict(state='completed' if step==6000 else 'stopped',step=step,total_steps=6000,best=best,qualified=qualified,elapsed_seconds=time.perf_counter()-start,**identity,updated=now(),frozen_parameters_unchanged=True))
    except BaseException as e:
        save(folder/f'failure_step{step}_{stamp()}.ckpt',checkpoint());write(status,dict(state='failed',step=step,error=repr(e),**identity,updated=now()));raise


if __name__=='__main__':main()
