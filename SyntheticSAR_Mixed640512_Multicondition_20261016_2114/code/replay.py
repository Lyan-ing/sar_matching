"""Regenerate selected arrays in memory and compare with frozen disk data."""
from generator import *
import argparse
def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',nargs='+');a=p.parse_args();rr=[r for p in a.manifest for r in rows(p)];cfgpath=latest('meta/GENERATION_CONFIG_*.json');cfg=json.loads(cfgpath.read_text());ds={d['anchor']:d for d in rows(cfg['source_domains'])}
    chosen=[];seen=set()
    for r in rr:
        keys=[('size-profile',r['width'],r['enhancement_profile']),('size-overlap',r['width'],r['overlap_bin']),('angle',r['angle_bin']),('scene',r['source_scene'])]
        if any(k not in seen for k in keys):chosen.append(r);seen.update(keys)
    outcomes=[]
    for r in chosen:
        a=generate(ds[r['source_path']],r['sampling_spec'],cfg);g=a['g']
        assert [ahash(x) for x in a['images']]==r['output_pixel_hashes']
        with np.load(r['gt_path']) as z:
            for i,L in enumerate('AB'):
                assert np.array_equal(g['T'][i],z['T_'+L]);assert np.array_equal(g['valid'][i],z['valid_'+L]);assert np.array_equal(a['ignore'][i],z['ignore_'+L])
                assert np.array_equal(a['points'][i],z['structure_points_'+L]);assert np.array_equal(a['checks'][i],z['check_points_'+L])
        outcomes.append(dict(pair_id=r['pair_id'],size=r['width'],profile=r['enhancement_profile'],replayed=True))
    save(ROOT/f'meta/REPLAY_CHECK_{stamp()}.json',dict(created=now(),passed=True,cases=outcomes,exact_images_matrices_masks_and_points=True,model_inference_or_training=False))
    print(json.dumps(dict(passed=True,replayed=len(outcomes))))
if __name__=='__main__':main()
