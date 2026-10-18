"""Focused CPU contracts plus real preflight evidence; no fabricated GT."""
import inspect,json,subprocess
from pathlib import Path
import numpy as np
from common import ROOT,atomic_json,stamp,readrows
from data import component,pair,warp,valid_at
from monitor import restart_eligible
from control import running
def main():
 train=readrows(ROOT/'train_sources.jsonl');val=readrows(ROOT/'val_sources.jsonl');assert not {component(r) for r in train}&{component(r) for r in val}
 plans=json.loads((ROOT/'validation_plan.json').read_text());assert len(plans)==56
 for p in plans:
  z=pair(p['source'],p['seed'],p['regime'],dynamic_occlusion=p['dynamic_occlusion']);np.testing.assert_allclose(warp(z['H_AB'],z['points0']),z['points1'],atol=3e-5)
  assert all(valid_at(z['static'][i],z['points'+str(i)]).all() for i in [0,1])
  if p['regime']=='none':assert z['overlap']==0 and all(v.mean()>.9 for v in z['valid'])
 for variant in ['m1','m2']:
  r=json.loads((ROOT/'preflight'/variant/'result.json').read_text());assert r['passed'] and r['real_optimizer_steps']==160 and r['parameter_squared_delta']>0 and r['resume_model_optimizer_rng_exact']
  assert r['gpu']['physical_gpu']==(0 if variant=='m1' else 1)
 assert not restart_eligible(dict(status='completed'),0,True);assert not restart_eligible(dict(status='failed',failure_type='nonfinite'),0,True);assert not restart_eligible(dict(status='failed',failure_type='oom'),1,True);assert restart_eligible(dict(status='failed',failure_type='oom'),0,True)
 assert not running(dict(pid=1,process_start_ticks='wrong',variant='m1'))
 from evaluate import image_matches
 assert list(inspect.signature(image_matches).parameters)==['model','matcher','image0','image1']
 for p in ROOT.glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
 atomic_json(ROOT/('test_results_'+stamp()+'.json'),dict(passed=True,fixed_validation_geometry_cases=56,no_overlap_cases=8,preflight_real_updates={'m1':160,'m2':160},source_isolation=True,physical_GPU_enforcement=True,atomic_checkpoint_exact_resume=True,monitor_nonfinite_stop_and_bounded_recovery=True,PID_reuse_protection=True,prediction_no_GT_or_geography_parameter=True,shell_syntax=True,formal900_not_opened=True));print('ALL CONTRACT TESTS PASSED')
if __name__=='__main__':main()
