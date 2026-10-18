"""Materialize the declared supervision schema without inventing real positives."""
import sys
sys.dont_write_bytecode=True
from geo_tools import ROOT,lines,jsonl,save
import json

def main():
    regions={r['sample_id']:r for r in lines(ROOT/'regions.jsonl')};rows=[]
    for old in lines(ROOT/'training_examples/index.jsonl'):
        parents=[regions[s] for s in old.get('samples',[old.get('sample_id')])];layer=old['layer']
        row=dict(layer=layer,real_acquisitions=[v['acquisition_id'] for r in parents for v in (r['views'][:1] if layer in ['single_SAR_known_2D_transform','geographic_no_overlap_negative'] else r['views'])],
            source_component=sorted({r['component'] for r in parents}),site=sorted({r['site_name'] for r in parents}),split_role='diagnostic_development_seen',
            supervision_kind=old.get('supervision',old.get('label','geographic_no_overlap_only' if layer=='geographic_no_overlap_negative' else 'weak_search_only')),
            uncertainty=dict(physical_correspondence=None,independent_checks=0),transform_or_search_region=dict(manifest=old.get('path'),geographic_search_halfwindow_native_px=64 if layer=='geographic_weak_search' else None),
            parent_native_window=[dict(sample_id=r['sample_id'],windows=[v['original_native_window_xywh'] for v in r['views']],region_manifest=str(ROOT/'regions'/r['sample_id']/'metadata.json')) for r in parents],
            license=[v.get('license') for r in parents for v in r['acquisition_metadata']['views']],allow_training_now=False,independent_new_acquisition_count=0,original_demonstration_row=old)
        if layer=='single_SAR_known_2D_transform':row['uncertainty']['supervision']='analytic 2D synthetic geometry, not real acquisition correspondence'
        if layer=='difficult_unknown':row.update(positive_label=None,negative_label=None)
        if layer=='geographic_no_overlap_negative':row.update(geographic_intersection_area_m2=0,physical_scatterer_negative_claim=False)
        rows.append(row)
    schema=json.loads((ROOT/'training_examples/schema.json').read_text())
    assert all(set(schema['required_fields'])<=set(r) for r in rows)
    jsonl(ROOT/'training_examples/normalized_index.jsonl',rows)
    print('NORMALIZED',len(rows),'schema-conformant diagnostic examples; no real positives')
if __name__=='__main__':main()
