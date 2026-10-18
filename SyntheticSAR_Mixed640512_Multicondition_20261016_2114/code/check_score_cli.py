"""GT-oracle CLI unit test only; never an algorithm prediction or performance claim."""
from common import *
import subprocess

def main():
    release=json.loads(latest('meta/RELEASE_METADATA_*.json').read_text());manifest=release['manifests']['val'];rr=rows(manifest);t=stamp();folder=ROOT/f'scorer_cli_test_{t}';folder.mkdir(exist_ok=False);pred=[]
    for r in rr:
        with np.load(r['gt_path']) as z:
            no=bool(z['no_overlap']);pred.append(dict(pair_id=r['pair_id'],accepted=not no,H_AB=None if no else z['H_AB'].tolist(),
                matches_A=z['structure_points_A'].tolist(),matches_B=z['structure_points_B'].tolist(),
                keypoints_A=z['structure_points_A'].tolist(),keypoints_B=z['structure_points_B'].tolist(),
                test_only_GT_oracle_not_algorithm=True))
    path=folder/f'ORACLE_TEST_ONLY_PREDICTIONS_{t}.jsonl';jsonl(path,pred)
    cmd=[sys.executable,'-B',str(ROOT/'code/score.py'),manifest,str(path),'--output-dir',str(folder)]
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=240);assert result.returncode==0,result.stderr
    metrics=json.loads(result.stdout.strip().splitlines()[-1]);assert metrics['count']==1200 and metrics['no_overlap_count']==60 and metrics['Success@3px']==1. and metrics['no_overlap_FPR']==0.
    save(ROOT/f'meta/SCORER_CLI_CHECK_{t}.json',dict(created=now(),passed=True,command=cmd,metrics=metrics,GT_oracle_unit_test_only=True,
        output_folder=str(folder),not_algorithm_results=True));print(json.dumps(dict(passed=True,rows=1200,oracle_only=True)))
if __name__=='__main__':main()
