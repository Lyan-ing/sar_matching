"""Fixed step6000 B diagnostic, never used to replace the validation-selected model."""
from support22 import *
from evaluate22 import Evaluation,summarize,probes
import argparse
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args();process=gpu(a.gpu);lock=json.loads(out('selection_locked',folder=ROOT/'runs/b_core_evidence').read_text());assert lock['actual_steps']==6000
 models={'M2':Model('m2').cuda(),'B':Model('b').cuda(),'B_step6000':Model('b',ROOT/'runs/b_core_evidence/last.ckpt').cuda()};ev=Evaluation();rr=ev.run(models,rows(E['recommended_val_all']),'v22','terminal_b_v22',save_predictions=True)
 result=dict(at=now(),process=process,role='Predeclared terminal diagnostic, not checkpoint selection',steps=6000,rows=rr,summaries={mode:{m:summarize([r for r in rr if r['method']==m and r['protocol']==mode]) for m in models} for mode in ['natural','equal']},probes={m:probes(v) for m,v in models.items()},weights=dict(B_step6000=sha(ROOT/'runs/b_core_evidence/last.ckpt')))
 write(out('terminal_b_v22'),result);csvwrite(out('terminal_b_v22',ext='csv'),rr);print(json.dumps(result['summaries'],ensure_ascii=False))
if __name__=='__main__':main()
