"""Ordered, resumable CPU workers. Failed/empty calls advance the stream, never hide resampling."""
from support22 import *
from torch.utils.data import Dataset,DataLoader,get_worker_info
from collections import defaultdict
class FixedCycle:
 def __init__(self,rr,seed):
  self.rows=rr;self.groups=defaultdict(lambda:defaultdict(list))
  for i,r in enumerate(rr):self.groups[r['known_component']][r['source_path']].append(i)
  self.comps=sorted(self.groups,key=lambda x:stable_seed(seed,x));self.sources={c:sorted(self.groups[c],key=lambda x:stable_seed(seed,x)) for c in self.comps}
 def get(self,k):
  c=self.comps[k%len(self.comps)];v=k//len(self.comps);s=self.sources[c][v%len(self.sources[c])];rs=self.groups[c][s];i=rs[(v//len(self.sources[c]))%len(rs)];r=self.rows[i]
  return r,dict(requested_role=r['loss_role'],source_path=s,component=c,relation_id=r['relation_id'])
class Stream(Dataset):
 def __init__(self,manifest,mode,seed,folder,start=0,online=True,ship_aux=False,limit=100000):
  self.manifest=str(manifest);self.mode=mode;self.seed=seed;self.folder=Path(folder);self.start=start;self.online=online;self.ship_aux=ship_aux;self.limit=limit;self.sampler=None;self.reader=None
 def __len__(self):return self.limit
 def __getitem__(self,index):
  index+=self.start
  if self.sampler is None:
   rr=rows(self.manifest);self.sampler=BalancedSampler(rr,self.mode,self.seed) if self.mode in ['M2','B'] else FixedCycle(rr,self.seed)
   worker=get_worker_info();wid=worker.id if worker else 0
   if self.online:self.reader=Online22(E['config'],self.manifest,str(self.folder/f'online_w{wid}_from{self.start}_{TAG}.jsonl'),mode=self.mode,seed=self.seed,arm='V2-R',use_ship_aux=self.ship_aux)
  row,trace=self.sampler.get(index);trace.update(call_index=index,reference_role=row['loss_role'],reference_points=row['structural_anchor_count'],origin=row['origin'])
  try:
   z=self.reader.get(index) if self.online else load_pair(row,self.ship_aux)
   z['trace']=trace;return dict(pair=z,trace=trace,error=None)
  except ValueError as ex:return dict(pair=None,trace=trace,error=repr(ex))
def identity_collate(x):return x
def loader(stream,workers=4):return DataLoader(stream,batch_size=None,num_workers=workers,collate_fn=identity_collate,multiprocessing_context='spawn' if workers else None,prefetch_factor=2 if workers else None)
