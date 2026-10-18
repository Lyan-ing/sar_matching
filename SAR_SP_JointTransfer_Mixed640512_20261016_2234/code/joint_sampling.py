"""Finite, train-only, without-replacement source/scene interleaving; no new views."""
from mixed_io import *
from collections import defaultdict,deque,Counter

def ordered_bucket(rows,width,seed,epoch):
 rng=np.random.default_rng(seed+width*1009+epoch*100003);groups=defaultdict(lambda:defaultdict(list))
 for i,r in enumerate(rows):
  if r['width']==width:groups[r['source_scene']][r['source_component_id']].append(i)
 scenes={}
 for scene,components in sorted(groups.items()):
  keys=list(sorted(components));rng.shuffle(keys);queues={k:deque(rng.permutation(components[k]).tolist()) for k in keys};out=[]
  while any(queues.values()):
   for k in keys:
    if queues[k]:out.append(queues[k].popleft())
  scenes[scene]=deque(out)
 sizes={k:len(v) for k,v in scenes.items()};used=Counter();out=[]
 while any(scenes.values()):
  key=min((k for k in scenes if scenes[k]),key=lambda k:((used[k]+.5)/sizes[k],k));out.append(scenes[key].popleft());used[key]+=1
 return out

def make_plan(steps=6000,seed=42):
 rows=records('train');epochs={512:0,640:0};queues={w:deque(ordered_bucket(rows,w,seed,0)) for w in epochs};plan=[];coverage=Counter();grad=Counter();scene=Counter();size=Counter();drawn=0
 for step in range(steps):
  w=512 if step%5==4 else 640;calls=[];active=[]
  while len(active)<8:
   if not queues[w]:epochs[w]+=1;queues[w]=deque(ordered_bucket(rows,w,seed,epochs[w]))
   idx=queues[w].popleft();r=rows[idx];calls.append(idx);coverage[idx]+=1;drawn+=1
   if r['structural_anchor_count']:
    active.append(idx);grad[idx]+=1;scene[r['source_scene']]+=1;size[w]+=1
  plan.append(dict(step=step+1,size=w,active=active,consumed=calls))
 assert len(coverage)==4800 and set(grad)=={i for i,r in enumerate(rows) if r['structural_anchor_count']}
 folder=ROOT/'plans';folder.mkdir(exist_ok=True);path=folder/f'sampling_seed{seed}_{stamp()}.jsonl'
 with path.open('x') as f:
  for s in plan:f.write(json.dumps(s,separators=(',',':'))+'\n')
 csvwrite(folder/f'source_coverage_seed{seed}_{stamp()}.csv',[dict(pair_id=r['pair_id'],source=r['source_path'],component=r['source_component_id'],scene=r['source_scene'],size=r['width'],role=r['loss_role'],anchors=r['structural_anchor_count'],consumed=coverage[i],gradient_calls=grad[i]) for i,r in enumerate(rows)])
 write(folder/f'sampling_identity_seed{seed}_{stamp()}.json',dict(seed=seed,steps=steps,effective_batch=8,size_schedule='640,640,640,640,512',path=str(path),sha=sha(path),train_manifest_sha=sha(MANIFEST['train']),consumed_calls=drawn,gradient_calls=sum(grad.values()),distinct_relationships_consumed=len(coverage),distinct_gradient_relationships=len(grad),distinct_source_files_consumed=len({r['source_path'] for r in rows}),distinct_gradient_source_files=len({rows[i]['source_path'] for i in grad}),gradient_source_components=len({rows[i]['source_component_id'] for i in grad}),scene_gradient_calls=dict(scene),size_gradient_calls=dict(size),empty_records_skipped_without_fake_labels=drawn-sum(grad.values()),no_online_view_regeneration=True))
 print(path);return path
if __name__=='__main__':make_plan()
