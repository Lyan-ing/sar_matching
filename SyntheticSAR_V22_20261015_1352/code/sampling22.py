"""Deterministic role/component/source/ROI cycling, fresh online pairs and full replay."""
from data22 import *
from collections import defaultdict,Counter

class BalancedSampler:
    def __init__(self,records,mode='B',seed=202622):
        self.seed=seed;self.mode=mode;self.weights=dict(structural_positive=40,limited_structure=25,no_structural_positive=27,no_overlap=8) if mode=='B' else dict(structural_positive=55,limited_structure=35,no_structural_positive=0,no_overlap=10)
        self.groups={role:defaultdict(lambda:defaultdict(lambda:defaultdict(list))) for role in self.weights}
        for r in records:
            assert r['split']=='train','training sampler cannot contain Val'
            role=r['loss_role']
            if role=='limited_structure' and mode=='M2' and not r['local_descriptor_eligible']:continue
            if role=='no_structural_positive' and sum(r['calibration_pixels'])<64:continue
            self.groups[role][r['known_component']][r['source_path']][r['roi_family_id']].append(r)
        for role,weight in self.weights.items():
            if weight and not self.groups[role]:raise ValueError('No eligible role; no silent rebalance: '+role)
        self.components={role:sorted(g,key=lambda x:stable_seed(seed,role+x)) for role,g in self.groups.items()}
        self.source_lists={(role,c):sorted(g,key=lambda x:stable_seed(seed,role+x)) for role,grp in self.groups.items() for c,g in grp.items()}
    def get(self,step):
        block,pos=divmod(step,100);roles=[r for r,w in self.weights.items() for _ in range(w)]
        order=np.random.default_rng(stable_seed(self.seed,'block'+str(block))).permutation(100);sequence=[roles[i] for i in order];role=sequence[pos]
        within=sum(r==role for r in sequence[:pos]);visit=block*self.weights[role]+within
        comps=self.components[role];comp=comps[visit%len(comps)];cvisit=visit//len(comps)
        srcs=self.source_lists[(role,comp)];src=srcs[cvisit%len(srcs)];svisit=cvisit//len(srcs)
        rois=self.groups[role][comp][src];roi=sorted(rois)[svisit%len(rois)];rvisit=svisit//len(rois);rs=rois[roi];row=rs[rvisit%len(rs)]
        return row,dict(step=step,requested_role=role,component=comp,source_path=src,roi_family_id=roi,relation_id=row['relation_id'],role_visit=visit)

class Online22:
    def __init__(self,config,reference_manifest,log_path,mode='B',seed=202622,arm='V2-R',use_ship_aux=False):
        self.cfg=json.loads(Path(config).read_text());self.references=rows(reference_manifest);self.sampler=BalancedSampler(self.references,mode,seed)
        self.sources={r['image_path']:r for r in sources()};self.seed=seed;self.arm=arm;self.use_ship_aux=use_ship_aux;self.log_path=Path(log_path)
        self.log_path.parent.mkdir(parents=True,exist_ok=True)
        with self.log_path.open('x') as f:f.write(json.dumps(dict(type='identity',created_at=now(),config=str(config),config_sha=sha(config),reference_manifest=str(reference_manifest),reference_sha=sha(reference_manifest),
            code_sha=sha(__file__),arm=arm,mode=mode,seed=seed,use_ship_aux=use_ship_aux))+'\n')
    def get(self,step,epoch=0,visit=0):
        ref,event=self.sampler.get(step);seed=stable_seed(self.seed,f'{step}:{epoch}:{visit}')
        spec=v21.make_spec(step,seed,self.cfg,forced=dict(roi_center=np.asarray(ref['source_window_corners'][0]).mean(0).tolist(),requested_size=ref['width']))
        if event['requested_role']=='no_overlap':spec['profile']='none'
        elif spec['profile']=='none':spec['profile']='affine' # Conditional nonnegative profile draw; negative role sampled separately.
        event.update(type='sample',at=now(),epoch=epoch,visit=visit,source_sha=sha(ref['source_path']),spec=spec,arm=self.arm,use_ship_aux=self.use_ship_aux)
        try:
            z=online_pair(self.sources[ref['source_path']],spec,self.cfg,self.arm,self.use_ship_aux)
            event.update(state='usable',actual_role=z['loss_role'],ground_points=len(z['points0']),ship_points=len(z['ship_points0']),H_AB=z['H_AB'].tolist(),
                appearance=z['appearance'],image_sha=[digest_array(a) for a in z['images']],mask_sha={k:[digest_array(a) for a in z[k]] for k in ['valid','static','ignore','calibration_mask']},
                ground_point_sha=[digest_array(z[k]) for k in ['points0','points1']])
        except Exception as ex:
            event.update(state='rejected',reason=repr(ex));self._log(event);raise
        self._log(event);z['sampling_record']=event;return z
    def _log(self,event):
        with self.log_path.open('a') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

def replay22(event,source,cfg):
    assert sha(source['image_path'])==event['source_sha']
    z=online_pair(source,event['spec'],cfg,event['arm'],event['use_ship_aux'])
    assert [digest_array(a) for a in z['images']]==event['image_sha']
    assert np.array_equal(z['H_AB'],np.asarray(event['H_AB']))
    for key,hashes in event['mask_sha'].items():assert [digest_array(a) for a in z[key]]==hashes
    assert [digest_array(z[k]) for k in ['points0','points1']]==event['ground_point_sha']
    return z
