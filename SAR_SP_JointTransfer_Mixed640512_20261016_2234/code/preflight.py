from mixed_io import *
import subprocess
state=setup();train,val=records('train'),records('val')
a={r['source_component_id'] for r in train};b={r['source_component_id'] for r in val};assert not a&b
assert not {r['source_path'] for r in train}&{r['source_path'] for r in val}
models={k:Model(k).cuda() for k in ['m0','m2','b','m2v22']};ref=models['m0'].state_dict();shared=[k for k in ref if k.startswith('sp.conv') and not k.startswith(('sp.convP','sp.convD'))]
for m in models.values():assert all(torch.equal(m.state_dict()[k],ref[k]) for k in shared)
assert all(torch.equal(v,models['m2'].state_dict()[k]) for k,v in models['b'].state_dict().items() if k.startswith('sp.convD'))
checks=[]
for size in [512,640]:
 row=next(r for r in val if r['width']==size);im=images(row)[0];x=torch.tensor(im,device='cuda').float()[None,None]/255
 with torch.no_grad():
  m=models['b'];actual=m.maps(x);legacy=SARNet.maps(m,x);assert all(torch.equal(v,w) for v,w in zip(actual,legacy))
  f=m.extract(x);assert f['image_size'].tolist()==[[size,size]];k=f['keypoints'][0];assert ((k>=0)&(k<size)).all()
  # Stride-8 official descriptor sampling remains bitwise unchanged.
  assert torch.equal(f['descriptors'],sample_descriptors(k[None],actual[1],8).transpose(1,2))
  lg=LightGlue(features='superpoint').cuda().eval().requires_grad_(False);pred,e,di,c=predict(lg,[f,f],im.shape);assert e.accepted
 checks.append(dict(size=size,keypoints=len(k),identity_error=float(np.linalg.norm(e.candidate_H-np.eye(3))),maps_bitwise_equal=True))
 del lg
report=dict(state=state,checks=checks,counts=dict(train=len(train),val=len(val),train_components=len(a),val_components=len(b)),hashes={k:sha(p) for k,p in WEIGHTS.items()},manifest_hashes={k:sha(p) for k,p in MANIFEST.items()},shared_encoder_identical=True,B_descriptor_identical_historical_M2=True,trainable_parameters={k:sum(p.numel() for n,p in models['b'].named_parameters() if any(n.startswith(q) for q in prefixes)) for k,prefixes in {'t1':['structure.','sp.convP'],'t2':['sp.convD'],'t3':['structure.','sp.convP','sp.convD']}.items()},environment=subprocess.check_output(['conda','list','-n','match','--explicit'],text=True))
write(ROOT/f'preflight_{stamp()}.json',report);print(json.dumps(clean(report),ensure_ascii=False))
