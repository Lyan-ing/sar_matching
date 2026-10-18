"""Backward-compatible evidence policy: unknown uncertainty stays null/ineligible."""
import importlib.util,math
from pathlib import Path
SOURCE=Path(__file__).resolve().parent.parent/'RealSAR_GeometryGT_20261011_0220/gt_quality/quality.py'
spec=importlib.util.spec_from_file_location('benchmark_legacy_quality',SOURCE)
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
POLICY=legacy.POLICY
coord=legacy.coord
def validate_points(points,task):
    ids=[];a=[];b=[]
    for p in points:
        pid=p.get('point_id')
        if not isinstance(pid,str) or not pid or len(pid)>80:raise ValueError('invalid point ID')
        if p.get('sample_id')!=task['sample_id']:raise ValueError('point/sample ID mismatch')
        if p.get('role') not in ['control','check']:raise ValueError('role must be control or check')
        ids.append(pid);a.append(coord(p.get('xy0'),(task['height0'],task['width0'])));b.append(coord(p.get('xy1'),(task['height1'],task['width1'])))
        for key in ['sigma0_px','sigma1_px']:
            v=p.get(key)
            if v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0):raise ValueError('uncertainty must be null or explicit nonnegative number')
    if len(set(ids))!=len(ids):raise ValueError('duplicate point IDs')
    for pts in [a,b]:
        for i in range(len(pts)):
            if any(legacy.np.linalg.norm(pts[i]-pts[j])<.5 for j in range(i)):raise ValueError('duplicate coordinate within 0.5px in either view')
    return legacy.np.asarray(a).reshape(-1,2),legacy.np.asarray(b).reshape(-1,2)
legacy.validate_points=validate_points
legacy_quality=legacy.quality
def quality(doc,task):
    q=legacy_quality(doc,task)
    for p,e in zip(doc.get('points',[]),q['point_evidence']):
        if p.get('sigma0_px') is None or p.get('sigma1_px') is None:
            # The legacy reviewer calculation already excludes unknown values.
            # Never impute a number or relax any eligibility/support threshold.
            assert not e['eligible']
            e['reasons'].append('annotation_uncertainty_not_explicit')
    return q
legacy.quality=quality
quantitative=legacy.quantitative
