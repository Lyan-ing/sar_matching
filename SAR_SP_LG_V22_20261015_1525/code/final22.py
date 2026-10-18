from support22 import *
from evaluate22 import Evaluation,summarize,probes
import argparse
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--group',choices=['main','attribution','ship'],required=True);p.add_argument('--version',choices=['v1','v21','v22'],required=True);a=p.parse_args();process=gpu(a.gpu)
 if a.group=='main':models={'M0':Model('m0').cuda(),'M2':Model('m2').cuda(),'B':Model('b').cuda(),'M2_V22':Model('m2',ROOT/'runs/m2_core/best.ckpt').cuda(),'B_V22':Model('b',ROOT/'runs/b_core/best.ckpt').cuda(),'ORB':None}
 elif a.group=='attribution':models={'M2':Model('m2').cuda(),**{k:Model('m2',ROOT/f'runs/{k}/best.ckpt').cuda() for k in ['attr_old','attr_local','attr_roi']}}
 else:models={'B_V22':Model('b',ROOT/'runs/b_core/best.ckpt').cuda(),**{k:Model('b',ROOT/f'runs/{k}/best.ckpt').cuda() for k in ['ship_off','ship_on']}}
 for name in ['m2_core','b_core'] if a.group=='main' else ['attr_old','attr_local','attr_roi'] if a.group=='attribution' else ['ship_off','ship_on']:assert out('selection_locked',folder=ROOT/'runs'/name).exists(),name
 records=rows(E['frozen_validation']['V1_standard']['val_parent_core'] if a.version=='v1' else E['frozen_validation']['V21_all']['V2-R/val/core'] if a.version=='v21' else E['recommended_val_all'])
 ev=Evaluation();tag=a.group+'_'+a.version;rr=ev.run(models,records,a.version,tag,save_predictions=True)
 result=dict(at=now(),process=process,group=a.group,version=a.version,rows=rr,summaries={mode:{m:summarize([r for r in rr if r['method']==m and r['protocol']==mode]) for m in models} for mode in ['natural','equal']},probes={m:probes(v) for m,v in models.items()},weights={m:sha(ROOT/f'runs/{m}/best.ckpt') for m in []})
 write(out(tag),result);csvwrite(out(tag,ext='csv'),rr);print(json.dumps(result['summaries'],ensure_ascii=False),flush=True)
if __name__=='__main__':main()
