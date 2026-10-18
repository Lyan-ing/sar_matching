"""A rejected ORB must execute SIFT; an accepted ORB must not execute it."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

spec=importlib.util.spec_from_file_location('phase26',Path(__file__).parents[1]/'scripts/phase26_benchmark.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_actual_fallback_branch():
    for accepted in [True,False]:
        called=[]
        def orb():called.append('orb');return {'estimate':SimpleNamespace(accepted=accepted),'variant':'orb'}
        def sift():called.append('sift');return {'estimate':SimpleNamespace(accepted=True),'variant':'sift'}
        chosen,branches,fallback=runner.fallback_pipeline(orb,sift)
        assert called==(['orb'] if accepted else ['orb','sift'])
        assert chosen['variant']==('orb' if accepted else 'sift')
        assert len(branches)==(1 if accepted else 2) and fallback is (not accepted)
