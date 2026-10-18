"""Train-only, bounded mechanism proof; no Val or test-guided weight choice."""
from keep_train import *


def main():
    identity=setup(42);rows=records('train');audit=inventory();plan=[json.loads(s) for s in (ROOT/f'plans/sampling_seed42_{TAG}.jsonl').read_text().splitlines()]
    status=ROOT/f'mechanism_status_{TAG}.json';write(status,dict(stage='loading_frozen_features',identity=identity,updated=now()))
    teacher=Model('b').cuda().requires_grad_(False);bank=Bank(rows,[i for e in plan for i in e['ids']],teacher)
    probeids=[]
    for size in [512,640]:
        for scene in sorted({r['scene'] for r in audit}):
            choices=[r['index'] for r in audit if r['size']==size and r['scene']==scene and r['keep_eligible']]
            probeids+=choices[-4:]
    probeids=sorted(set(probeids));assert probeids
    model=Model('b').cuda();opt=optim(model);scheduler=schedule(opt);initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};overfit=[]
    # Four fixed Train batches warm the descriptor head solely to obtain a nonzero keep gradient.
    for step in range(64):
        entry=plan[step%8];ids=entry['ids'][:4];x,rs,items=bank.batch(ids);opt.zero_grad(set_to_none=True);d,_,_=terms(model,teacher,x,rs,items,['desc']*4);d.backward();torch.nn.utils.clip_grad_norm_(configure_parameters(model),5.);opt.step();scheduler.step();overfit.append(float(d))
    assert np.mean(overfit[-8:])<np.mean(overfit[:8]),'Small fixed-Train descriptor loss did not decrease'
    frozen_check(model,teacher);assert any(not torch.equal(v.detach().cpu(),initial[k]) for k,v in model.state_dict().items() if k.startswith('sp.convD'))
    scales=[]
    for entry in plan[:8]:
        x,rs,items=bank.batch(entry['ids']);d,k,_=terms(model,teacher,x,rs,items,entry['roles']);ps=configure_parameters(model)
        gd=torch.autograd.grad(d,ps,retain_graph=True);gk=torch.autograd.grad(k,ps);nd=float(torch.sqrt(sum(g.float().square().sum() for g in gd)));nk=float(torch.sqrt(sum(g.float().square().sum() for g in gk)))
        assert nd>0 and nk>1e-8 and np.isfinite([nd,nk]).all();scales.append(dict(size=entry['size'],desc=float(d),keep=float(k),gradient_desc=nd,gradient_keep=nk))
    weight=float(f'{.5*np.median([r["gradient_desc"]/r["gradient_keep"] for r in scales]):.6g}');assert 0<weight<1e6
    locks=sorted(ROOT.glob('GRADIENT_SCALE_LOCK_*.json'));recovering=bool(locks)
    if recovering:
        locked=json.loads(locks[0].read_text());assert weight==locked['lambda_keep'],'Do not change the locked coefficient during engineering-check recovery'
        write(ROOT/f'GRADIENT_SCALE_RECHECK_{stamp()}.json',dict(lambda_keep=weight,original_lock=str(locks[0]),same_coefficient=True,reason='Recover final frozen-inference scope assertion; no repeated pilot optimization',scales=scales))
    else:write(ROOT/f'GRADIENT_SCALE_LOCK_{stamp()}.json',dict(lambda_keep=weight,target_weighted_keep_gradient_ratio=.5,warmup_train_steps=64,fixed_batches=8,scales=scales,validation_accessed=False,old_anchor_preserve_removed=True,rule='0.5*median(norm(g_desc)/norm(g_keep)) after64 fixed-Train descriptor-only warmup; discarded calibration model'))
    # Accumulation check uses FP32 to separate the batching identity from AMP rounding.
    a=Model('b').cuda();b=Model('b').cuda();oa=optim(a);ob=optim(b);sa=schedule(oa);sb=schedule(ob);entry=plan[0]
    update(a,teacher,oa,sa,bank,entry,weight,False)
    x,rs,items=bank.batch(entry['ids']);ob.zero_grad(set_to_none=True);d,k,_=terms(b,teacher,x,rs,items,entry['roles'],False);(d+weight*k).backward();torch.nn.utils.clip_grad_norm_(configure_parameters(b),5.);ob.step();sb.step()
    accumulation=max(float((p-q).abs().max()) for p,q in zip(a.parameters(),b.parameters()));assert accumulation<2e-5
    # Exact checkpoint/RNG/optimizer/scheduler restoration on the next update.
    cp=ROOT/'runs/smoke'/f'resume_{TAG}.ckpt';save(cp,dict(model_state=a.state_dict(),optimizer_state=oa.state_dict(),scheduler_state=sa.state_dict(),amp_scaler={},global_step=1,random_state=rng()))
    update(a,teacher,oa,sa,bank,plan[1],weight,False);c=Model('b').cuda();oc=optim(c);sc=schedule(oc);ck=torch.load(cp,map_location='cpu',weights_only=False);c.load_state_dict(ck['model_state']);oc.load_state_dict(ck['optimizer_state']);sc.load_state_dict(ck['scheduler_state']);restore_rng(ck['random_state']);update(c,teacher,oc,sc,bank,plan[1],weight,False)
    resume=max(float((p-q).abs().max()) for p,q in zip(a.parameters(),c.parameters()));assert resume<1e-6
    del a,b,c,oa,ob,oc;torch.cuda.empty_cache();results={};probe0=probes(teacher)
    # Same inputs, update budget and sampling: only lambda differs in this short diagnostic.
    for name,lam in [('no_keep',0.),('with_keep',weight)]:
        m=Model('b').cuda();op=optim(m);sch=schedule(op);folder=ROOT/'runs'/f'pilot_{name}';folder.mkdir(parents=True,exist_ok=True);metricpath=folder/f'metrics_{TAG}.jsonl'
        write(status,dict(stage='pilot_'+name,step=0,total_steps=400,lambda_keep=weight,updated=now()))
        metrics=[]
        if metricpath.exists():
            metrics=[json.loads(s) for s in metricpath.read_text().splitlines()];assert len(metrics)==400 and [r['step'] for r in metrics]==list(range(1,401))
            saved=torch.load(folder/f'last_{TAG}.ckpt',map_location='cpu',weights_only=False);assert saved['global_step']==400 and saved['lambda_keep']==lam and saved['teacher_sha']==sha(WEIGHTS['b']);m.load_state_dict(saved['model_state']);op.load_state_dict(saved['optimizer_state']);sch.load_state_dict(saved['scheduler_state']);print(now(),'REUSED_COMPLETED_PILOT',name,flush=True)
        else:
            for entry in plan[:400]:
                metric=update(m,teacher,op,sch,bank,entry,lam);metrics.append(metric)
                with metricpath.open('a') as f:f.write(json.dumps(clean(metric))+'\n')
                if entry['step']%50==0:write(status,dict(stage='pilot_'+name,step=entry['step'],total_steps=400,metric=metric,updated=now()));print(now(),'PILOT',name,entry['step'],metric['desc'],metric['keep'],flush=True)
        frozen_check(m,teacher);assert probes(m)==probe0
        assert np.mean([r['desc'] for r in metrics[-50:]])<np.mean([r['desc'] for r in metrics[:50]])
        dr=drift(m,teacher,bank,probeids);results[name]=dict(drift=dr,mean_drift=float(np.mean([r['cosine_drift'] for r in dr])),first50_desc=float(np.mean([r['desc'] for r in metrics[:50]])),last50_desc=float(np.mean([r['desc'] for r in metrics[-50:]])),nonzero_keep_gradient_steps=sum(r['gradient_keep']>1e-8 for r in metrics),max_gradient=max(r['preclip_gradient'] for r in metrics))
        if not recovering:save(folder/f'last_{TAG}.ckpt',dict(model_state=m.state_dict(),optimizer_state=op.state_dict(),scheduler_state=sch.state_dict(),global_step=400,lambda_keep=lam,teacher_sha=sha(WEIGHTS['b'])))
        if name=='with_keep':
            x,rs,items=bank.batch(plan[0]['ids'][:1])
            with torch.no_grad():logs,ds=m.cached_maps(x,rs);lt,dt=teacher.cached_maps(x,rs)
            assert torch.equal(logs,lt);assert torch.allclose(torch.linalg.vector_norm(ds,dim=1),torch.ones_like(ds[:,0]),atol=2e-5);assert float(ds.std(dim=(-2,-1)).mean())>.001
        del m,op,sch;torch.cuda.empty_cache()
    ratio=results['with_keep']['mean_drift']/max(results['no_keep']['mean_drift'],1e-12);assert ratio<.95,'Preservation failed to constrain Train-probe drift; inspect before formal training'
    assert results['with_keep']['nonzero_keep_gradient_steps']>=390
    # New actual trained descriptor head with frozen LG is exercised, not just shape checked.
    m=Model('b',ROOT/'runs/pilot_with_keep'/f'last_{TAG}.ckpt').cuda();lg=LightGlue(features='superpoint').cuda().eval().requires_grad_(False)
    ims=images(rows[plan[0]['ids'][0]])
    with torch.no_grad():fs=[m.extract(torch.tensor(im,device='cuda').float()[None,None]/255) for im in ims];pred,e,diag,conf=predict(lg,fs,ims[0].shape)
    assert all(p.grad is None for p in lg.parameters());configure(m);frozen_check(m,teacher)
    report=dict(passed=True,identity=identity,lambda_keep=weight,probe_indices=probeids,trainable_parameters=360960,frozen_parameters_unchanged=True,detector_logits_bitwise_identical=True,probes_identical=True,initial_black_gray_zero=True,teacher_stopped=True,small_pool_overfit=dict(first8=float(np.mean(overfit[:8])),last8=float(np.mean(overfit[-8:]))),gradient_accumulation_parameter_difference=accumulation,resume_parameter_difference=resume,pilots=results,drift_ratio=ratio,short_optimizer_steps=936 if recovering else 868,engineering_assertion_recovery=recovering,completed_pilot_weights_reused=recovering,validation_not_used=True,LightGlue_called=True,LightGlue_frozen=True,loss='L_desc + lambda*L_keep; old .1 anchor regularizer removed',same_source_teacher_preservation_is_not_new_correspondence_GT=True)
    write(ROOT/f'MECHANISM_ACCEPTED_{TAG}.json',report);write(status,dict(stage='accepted',updated=now(),lambda_keep=weight,drift_ratio=ratio));print(now(),'MECHANISM_ACCEPTED',weight,ratio,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as e:write(ROOT/f'mechanism_failure_{stamp()}.json',dict(error=repr(e),updated=now()));raise
