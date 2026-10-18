"""Replay a few actually consumed online calls; never re-audit the historical corpus."""
from support22 import *
from sampling22 import replay22,sources
def main():
 cfg=json.loads(Path(E['config']).read_text());sr={r['image_path']:r for r in sources()};checks=[]
 for name in ['m2_core','b_core_evidence']:
  run=ROOT/'runs'/name;consumed=set()
  for p in run.glob('sampling_from*.jsonl'):
   for line in p.read_text().splitlines():
    try:consumed.add(json.loads(line)['call_index'])
    except json.JSONDecodeError:pass
  selected={}
  for p in sorted(run.glob('online_w*.jsonl')):
   for line in p.read_text().splitlines():
    try:event=json.loads(line)
    except json.JSONDecodeError:continue
    if event.get('type')!='sample' or event.get('state')!='usable' or event['step'] not in consumed:continue
    role=event['actual_role'];key='few_points' if 0<event['ground_points']<4 else role
    if key not in selected:selected[key]=event
    if len(selected)>=5:break
   if len(selected)>=5:break
  for key,event in selected.items():
   z=replay22(event,sr[event['source_path']],cfg);p,q=z['points0'],z['points1'];error=float(np.linalg.norm(transform_points(z['H_AB'],p)-q,axis=1).max()) if len(p) else None
   assert error is None or error<.001
   assert all(not (c&u).any() for c,u in zip(z['calibration_mask'],z['supervision_ignore']))
   assert all(not (m&s).any() for m,s in zip(z['static'],z['ship_masks']))
   checks.append(dict(run=name,call_index=event['step'],actual_role=z['loss_role'],size=z['images'][0].shape[0],points=len(p),max_GT_coordinate_error=error,pixel_masks_H_and_points_exactly_replayed=True))
 t=stamp();write(ROOT/f'consumed_online_replay_{t}.json',dict(at=now(),passed=True,checks=checks));print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
