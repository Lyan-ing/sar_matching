"""Read-only evidence from executed historical manifests, masks and source code."""
from common import *
from common22 import old_rows
from concurrent.futures import ProcessPoolExecutor

def old_overlap(r):
    v=[read_gray(p)>0 for p in r['valid_masks']];H=np.asarray(r['H_AB']);over=overlap_mask(*(x.astype('uint8')*255 for x in v),H)
    fr=[ratio(o,a) for o,a in zip(over,v)]
    return dict(sample_id=r['sample_id'],split=r['split'],size=r['width'],actual_overlap=min(fr),overlap_bin=overlap_bin(min(fr)),
        manifest_difference=max(abs(a-b) for a,b in zip(fr,r['overlap_fractions'])),
        angle_actual_deg=angle_of(H,r['width']),angle_manifest_difference=abs(angle_of(H,r['width'])-r['image_rotation_2D_deg']))

def main():
    old=old_rows('V2-R');t=stamp()
    with ProcessPoolExecutor(max_workers=4) as ex:rr=list(ex.map(old_overlap,old,chunksize=8))
    jsonl(ROOT/f'meta/HISTORICAL_OVERLAP_RECOMPUTED_{t}.jsonl',rr)
    summary=dict(created=now(),historical_V21_R_pairs=len(rr),actual_overlap_from_saved_masks_and_H=True,
        overlaps=dict(Counter(r['overlap_bin'] for r in rr)),sizes=dict(Counter(r['size'] for r in rr)),
        max_overlap_field_discrepancy=max(r['manifest_difference'] for r in rr),max_angle_field_discrepancy=max(r['angle_manifest_difference'] for r in rr),
        source_native_upsampling_in_V21=False,old_generator_sizes_requested=CFG21['geometry']['requested_sizes'],
        executed_old_appearance=CFG21['appearance'],
        old_V21_structure_bug='insufficient global support caused saved points/static masks to be cleared; V2.2 graded labels preserve local evidence',
        new_sampler_reuses_exact_old_view_matrix_transform_points_warp_overlap_mask_and_ratio=True,
        reused_radiometry='sarreg.synthesis._apply_augment, noise/blur disabled for P1; tested exact pixels under same seed',
        adapted_appearance='V2.1 fine/coarse residual, standardized gamma/correlated fields, smooth clutter, separable small blur and soft local scatter split into optional operators; lighter settings locked',
        added_ops='unit-mean lognormal multiplicative perturbation uses existing sarreg.synthesis._apply_augment formula in float composition',
        scene_labels='existing official coarse/AI source-context labels retained, no per-crop physical semantics fabricated',
        old_raw_crop_identity_not_fully_known=True,historical_data_modified=False)
    save(ROOT/f'meta/HISTORICAL_CODE_AUDIT_{t}.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
