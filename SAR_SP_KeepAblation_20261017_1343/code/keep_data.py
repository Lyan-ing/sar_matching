"""Train-only mask/position preparation; no new images or cross-image labels."""
from mixed_io import *
from collections import defaultdict, deque, Counter
from old_objective import local_coordinates

OLD = BASE/'SAR_SP_JointTransfer_Mixed640512_20261016_2234'
OLDTAG = '20261016_2234'


def uniform_positions(mask, points):
    mask = cv2.erode(mask.astype(np.uint8), np.ones((9,9),np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    for x,y in points:
        cv2.circle(mask,(int(round(x)),int(round(y))),12,0,-1)
    size=mask.shape[0];xy=np.stack(np.meshgrid(np.arange(3.5,size,8),np.arange(3.5,size,8)),axis=-1).reshape(-1,2).astype(np.float32)
    xy=xy[dc.bilinear_at(mask.astype(bool),xy)]
    selected=[]
    for by in range(16):
        for bx in range(16):
            inside=(xy[:,0]>=bx*size/16)&(xy[:,0]<(bx+1)*size/16)&(xy[:,1]>=by*size/16)&(xy[:,1]<(by+1)*size/16)
            q=xy[inside]
            if len(q):
                center=np.array([(bx+.5)*size/16,(by+.5)*size/16]);selected.append(q[np.argmin(((q-center)**2).sum(1))])
    return np.asarray(selected,np.float32).reshape(-1,2)


def prepare():
    rows=records('train');folder=ROOT/'cache/supervision';folder.mkdir(parents=True,exist_ok=True);audit=[]
    for i,row in enumerate(rows):
        path=folder/f"{row['pair_id']}_{TAG}.npz"
        old=OLD/'cache/train_supervision'/f"{row['pair_id']}_{OLDTAG}.npz"
        if old.exists():
            with np.load(old) as f:item={k:f[k].copy() for k in f.files}
        else:
            item=dict(points=np.empty((2,0,2),np.float32),patch_legal=np.empty((2,0,25),bool),raw_count=np.array(0),H_AB=np.array(row['H_AB']),local_weight=np.array(row['local_weight']))
        target = row['source_scene']=='offshore_vessels_low_texture_sea' or not row['global_eligible']
        with np.load(row['gt_path'],allow_pickle=False) as gt:
            for j,L in enumerate('AB'):
                legal=gt['valid_'+L]&~gt['ignore_'+L]&~gt['supervision_ignore_'+L]&~gt['ship_'+L]
                item['keep'+str(j)]=uniform_positions(legal,gt['structure_points_'+L]) if target else np.empty((0,2),np.float32)
                # Verify every selected location respects the original exclusion masks.
                assert dc.bilinear_at(legal,item['keep'+str(j)]).all()
                item['legal_pixels'+str(j)]=np.array(int(legal.sum()) if target else 0)
        np.savez_compressed(path,**item)
        audit.append(dict(index=i,pair_id=row['pair_id'],scene=row['source_scene'],component=row['source_component_id'],source=row['source_path'],size=row['width'],role=row['loss_role'],structure_points=item['points'].shape[1],raw_points=int(item['raw_count']),keep0=len(item['keep0']),keep1=len(item['keep1']),keep_eligible=bool(len(item['keep0'])+len(item['keep1'])),keep_target=target,legal_pixels0=int(item['legal_pixels0']),legal_pixels1=int(item['legal_pixels1'])))
        if i%200==0:print(now(),'PREPARE',i,4800,flush=True)
    traincomps={r['source_component_id'] for r in rows};valcomps={r['source_component_id'] for r in records('val')};assert not traincomps&valcomps
    write(ROOT/f'supervision_inventory_{TAG}.json',audit);csvwrite(ROOT/f'SUPERVISION_INVENTORY_{stamp()}.csv',audit)
    write(ROOT/f'DATA_IDENTITY_{stamp()}.json',dict(manifests={k:dict(path=str(v),sha=sha(v)) for k,v in MANIFEST.items()},train=4800,val=1200,train_components=len(traincomps),val_components=len(valcomps),component_overlap=[],teacher_sha=sha(WEIGHTS['b']),old_descriptor_sha=sha(OLD/'runs/t2/best_trained_20261016_2234.ckpt'),no_data_generated=True,ignore_rule='erode(valid & ~ignore & ~supervision_ignore & ~ship,9px), exclude all original anchors radius12px',same_image_teacher_only=True))
    for scene in sorted({r['scene'] for r in audit}):
        a=[r for r in audit if r['scene']==scene];print(scene,'relations',len(a),'desc',sum(bool(r['structure_points']) for r in a),'keep',sum(r['keep_eligible'] for r in a),flush=True)
    return audit


def inventory():return json.loads((FROZEN_ROOT/f'supervision_inventory_{FROZEN_TAG}.json').read_text())


def ordered(pool,seed,epoch):
    rng=np.random.default_rng(seed+epoch*100003);groups=defaultdict(lambda:defaultdict(list))
    for r in pool:groups[r['scene']][r['component']].append(r['index'])
    queues={}
    for scene,components in sorted(groups.items()):
        keys=sorted(components);rng.shuffle(keys);qq={k:deque(rng.permutation(components[k]).tolist()) for k in keys};q=[]
        while any(qq.values()):
            for k in keys:
                if qq[k]:q.append(qq[k].popleft())
        queues[scene]=deque(q)
    sizes={k:len(v) for k,v in queues.items()};used=Counter();out=[]
    while any(queues.values()):
        k=min((k for k in queues if queues[k]),key=lambda k:((used[k]+.5)/sizes[k],k));out.append(queues[k].popleft());used[k]+=1
    return out


def make_plan(steps=6000,seed=42):
    audit=inventory();pools={(w,role):[r for r in audit if r['size']==w and (r['structure_points']>0 if role=='desc' else r['keep_eligible'])] for w in [512,640] for role in ['desc','keep']}
    assert all(pools.values());epochs=Counter();queues={k:deque(ordered(v,seed+k[0]+(0 if k[1]=='desc' else 1000007),0)) for k,v in pools.items()};plan=[]
    for step in range(1,steps+1):
        size=512 if step%5==0 else 640;roles=['desc']*4+['keep']*4;ids=[]
        for role in roles:
            key=(size,role)
            if not queues[key]:epochs[key]+=1;queues[key]=deque(ordered(pools[key],seed+size+(0 if role=='desc' else 1000007),epochs[key]))
            ids.append(queues[key].popleft())
        plan.append(dict(step=step,size=size,ids=ids,roles=roles))
    p=ROOT/f'plans/sampling_seed{seed}_{TAG}.jsonl';p.parent.mkdir(exist_ok=True)
    with p.open('x') as f:
        for r in plan:f.write(json.dumps(r)+'\n')
    write(ROOT/f'SAMPLING_PROTOCOL_{stamp()}.json',dict(path=str(p),sha=sha(p),seed=seed,steps=steps,effective_batch=8,desc_pairs_per_step=4,keep_pairs_per_step=4,descriptive_updates=steps*4,preservation_updates=steps*4,pool_sizes={str(k):len(v) for k,v in pools.items()},without_replacement_scene_proportional_component_interleaving=True,changed_sampler_vs_previous=True,previous_descriptor_pair_calls=48000,current_descriptor_pair_calls=24000,no_causal_isolation_of_loss_from_sampling=True))
    return p


class Bank:
    def __init__(self,rows,indices,model):
        self.entries={};self.rows=rows
        for n,i in enumerate(sorted(set(indices))):
            row=rows[i];x,rs,_=shared_cache(row,model)
            with np.load(FROZEN_ROOT/'cache/supervision'/f"{row['pair_id']}_{FROZEN_TAG}.npz") as f:item={k:f[k].copy() for k in f.files}
            self.entries[i]=(x.cpu(),rs.cpu(),item)
            if n%200==0:print(now(),'BANK',n,len(set(indices)),flush=True)
    def batch(self,ids):
        xs,rs,items=zip(*(self.entries[i] for i in ids));return torch.cat(xs).cuda(),torch.cat(rs).cuda(),items


if __name__=='__main__':prepare();print(make_plan(),flush=True)
