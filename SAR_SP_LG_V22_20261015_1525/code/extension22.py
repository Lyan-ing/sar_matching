"""SSDD extension reporting, always separate from the core source-isolated denominator."""
from support22 import *
from evaluate22 import Evaluation,summarize,probes
import argparse
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--version',choices=['v1','v21','v22'],required=True);a=p.parse_args();process=gpu(a.gpu)
 for name in ['m2_core','b_core_evidence']:assert out('selection_locked',folder=ROOT/'runs'/name).exists()
 manifest=E['frozen_validation']['V1_standard']['val_ssdd_extension'] if a.version=='v1' else E['frozen_validation']['V21_all']['V2-R/val/ssdd'] if a.version=='v21' else E['manifests']['V2-R/val/ssdd/all'];rr=rows(manifest);tag='main_'+a.version+'_ssdd';assert not out(tag).exists()
 models={'M0':Model('m0').cuda(),'M2':Model('m2').cuda(),'B':Model('b').cuda(),'M2_V22':Model('m2',ROOT/'runs/m2_core/best.ckpt').cuda(),'B_V22':Model('b',ROOT/'runs/b_core_evidence/best.ckpt').cuda(),'ORB':None};ev=Evaluation();result=ev.run(models,rr,a.version,tag,save_predictions=True)
 data=dict(at=now(),version=a.version,pool='ssdd_extension_unproven_raw_acquisition_independence',process=process,manifest=manifest,manifest_sha=sha(manifest),rows=result,summaries={mode:{m:summarize([r for r in result if r['method']==m and r['protocol']==mode]) for m in models} for mode in ['natural','equal']},probes={m:probes(v) for m,v in models.items()})
 write(out(tag),data);csvwrite(out(tag,ext='csv'),result);print(json.dumps(data['summaries'],ensure_ascii=False))
if __name__=='__main__':main()
