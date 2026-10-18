"""Supplement: ORB original 1024 cap, same images/geometric gate, no GPU inference."""
from finalize import ORBEvaluator
from evaluation import *

def main():
 cv2.setNumThreads(1);torch.set_num_threads(2);plan=json.loads(output('plan_select').read_text())+json.loads(output('plan_holdout').read_text());ev=ORBEvaluator.__new__(ORBEvaluator);ev.plan=plan;ev.common_budgets={};ev.budgets={}
 rr=ev.evaluate(None,'orb_native1024','orb_native1024',save=True)
 for r in rr:
  r['protocol']='original ORB max1024, no M2-derived cap; same pixels/full gate'
  for k in list(r):
   if k.startswith(('descriptor_','NN_','positive_cosine','negative_cosine')):r[k]=None
 event('orb_native',dict(summary=summarize(rr),rows=rr));export_csv(ROOT/f'orb_native_per_pair_{stamp()}.csv',rr);print(json.dumps(summarize(rr),indent=2))
if __name__=='__main__':main()
