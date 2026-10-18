"""Small SAR structure baseline; frozen geometry and native pixel-center mapping."""
import sys
sys.dont_write_bytecode=True
import json, hashlib, importlib.util, time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.path.insert(0,str(BASE/'MaritimeSARRegistrationPhase2'))
from sarbench.matchers import Prediction
from sarbench.estimate import estimate,perspective
from sarbench.reliability import spatial,geometry_diagnostics,gate
spec=importlib.util.spec_from_file_location('frozen_geo',BASE/'RealSAR_AutoLabels_20261011_1149/geo.py')
geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
PixelMap,load_pair,sample_valid,regular_grid=geo.PixelMap,geo.load_pair,geo.sample_valid,geo.regular_grid
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(clean(x),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def selected():return lines(ROOT/'samples.jsonl')

def amplitude(raw,valid,units):
    a=np.asarray(raw,np.float64);v=np.asarray(valid,bool)&np.isfinite(a)
    if a.shape!=v.shape or a.ndim!=2:raise ValueError('invalid native data shape')
    if units not in ['linear_amplitude','linear_power']:raise ValueError('unknown radiometry')
    if np.any(a[v]<0):raise ValueError('negative valid linear SAR')
    b=np.zeros_like(a);b[v]=np.sqrt(a[v]) if units=='linear_power' else a[v]
    return b,v

def descriptors(raw,valid,units):
    a,v=amplitude(raw,valid,units)
    pos=a[v&(a>0)];eps=float(np.median(pos))*1e-6 if len(pos) else 1e-6
    left=np.zeros((7,7),np.float64);left[:,:3]=1/21
    right=np.fliplr(left);top=left.T;bottom=np.flipud(top)
    means=[cv2.filter2D(a,-1,k,borderType=cv2.BORDER_REFLECT_101) for k in [left,right,top,bottom]]
    gx=np.log((means[1]+eps)/(means[0]+eps));gy=np.log((means[3]+eps)/(means[2]+eps))
    mag=np.hypot(gx,gy);ori=np.mod(np.arctan2(gy,gx),2*np.pi)*8/(2*np.pi)
    # Full technical-valid support: no imputed NoData, and valid zeros remain data.
    support=cv2.erode(v.astype(np.uint8),np.ones((7,7),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    points=[];desc=[];failure={'grid_total':256,'window_outside':0,'invalid_support':0,'zero_structure':0}
    for p in regular_grid(a.shape,16):
        cx,cy=p;leftx=int(np.floor(cx-23.5));topy=int(np.floor(cy-23.5))
        if leftx<0 or topy<0 or leftx+48>a.shape[1] or topy+48>a.shape[0]:failure['window_outside']+=1;continue
        sl=np.s_[topy:topy+48,leftx:leftx+48]
        if not support[sl].all():failure['invalid_support']+=1;continue
        m=mag[sl];o=ori[sl];hist=[]
        for y in range(4):
            for x in range(4):
                om=o[y*12:(y+1)*12,x*12:(x+1)*12].ravel();mm=m[y*12:(y+1)*12,x*12:(x+1)*12].ravel()
                lo=np.floor(om).astype(int)%8;f=om-np.floor(om)
                hist.extend(np.bincount(lo,weights=mm*(1-f),minlength=8)+np.bincount((lo+1)%8,weights=mm*f,minlength=8))
        d=np.asarray(hist,float);norm=np.linalg.norm(d)
        if not np.isfinite(norm) or norm<=1e-12:failure['zero_structure']+=1;continue
        d/=norm;d=np.minimum(d,.2);d/=max(np.linalg.norm(d),1e-12)
        points.append(p);desc.append(d)
    return np.asarray(points,np.float32).reshape(-1,2),np.asarray(desc,np.float32).reshape(-1,128),dict(failure,descriptors=len(points),epsilon=eps,units=units)

def geographical_window(p0,p1,mapper,radius=64):
    q=mapper(p0);return np.isfinite(q).all(axis=1)[:,None] & (np.abs(q[:,None,:]-p1[None,:,:])<=radius).all(axis=2)

def mutual_match(p0,d0,p1,d1,allowed=None):
    n,m=len(p0),len(p1);allowed=np.ones((n,m),bool) if allowed is None else np.asarray(allowed,bool)
    if allowed.shape!=(n,m):raise ValueError('wrong candidate window shape')
    stats={'no_descriptors':not(n and m),'forward_no_candidates':int(np.sum(allowed.sum(1)==0)),
           'backward_no_candidates':int(np.sum(allowed.sum(0)==0)),
           'forward_less_than_two':int(np.sum(allowed.sum(1)<2)),'backward_less_than_two':int(np.sum(allowed.sum(0)<2))}
    if not n or not m:return np.empty((0,2)),np.empty((0,2)),np.empty(0),dict(stats,forward_ratio_pass=0,backward_ratio_pass=0,mutual_pass=0,cycle_failures=0)
    dist=np.sqrt(np.maximum(0,np.sum((d0[:,None,:]-d1[None,:,:])**2,axis=2)))
    dist=np.where(allowed,dist,np.inf)
    def nearest(z):
        ids=np.argsort(z,axis=1,kind='stable')
        if z.shape[1]<2:return np.zeros(z.shape[0],int),np.zeros(z.shape[0],bool),np.full(z.shape[0],np.inf)
        first,second=z[np.arange(len(z)),ids[:,0]],z[np.arange(len(z)),ids[:,1]]
        ok=np.isfinite(second)&(first<.8*second)
        ratios=np.divide(first,second,out=np.full(len(z),np.inf),where=np.isfinite(second)&(second>0))
        return ids[:,0],ok,ratios
    fi,fg,fr=nearest(dist);bi,bg,br=nearest(dist.T)
    ids=np.flatnonzero(fg);mutual=ids[bg[fi[ids]]&(bi[fi[ids]]==ids)];cycle=np.linalg.norm(p0[bi[fi[mutual]]]-p0[mutual],axis=1)
    kept=mutual[cycle<=3]
    stats.update(forward_ratio_pass=int(fg.sum()),backward_ratio_pass=int(bg.sum()),mutual_pass=len(mutual),
                 mutual_failures=int(len(ids)-len(mutual)),cycle_failures=int(np.sum(cycle>3)),cycle_max_px=float(cycle.max()) if len(cycle) else None,
                 cycle_note='Mutual grid-node roundtrip, identically zero for exact mutual IDs; not independent geometric evidence')
    return p0[kept],p1[fi[kept]],1-np.maximum(fr[kept],br[fi[kept]]),stats

def roma_native_coordinates(normalized,shape=(512,512)):
    # Official helper returns pixel-corner coordinates W/2*(u+1); repository
    # grid_sample uses align_corners=False, so our native pixel centers need -.5.
    n=np.asarray(normalized,float);h,w=shape
    return (n+1)*np.array([w,h])/2-.5

def fit(points0,points1,shape0=(512,512),shape1=(512,512)):
    pred=Prediction('shared',np.asarray(points0,np.float32),np.asarray(points1,np.float32))
    e=estimate(pred,shape0,model='affine',seed=42);diag=geometry_diagnostics(e,pred,shape0,shape1,42) if e.candidate_H is not None else {}
    g=gate(e,diag,'full')
    return g,diag

def diagnostics(row,method,protocol,p0,p1,confidence,valid,mapper,timing,extra=None):
    start=time.perf_counter();p0=np.asarray(p0,float).reshape(-1,2);p1=np.asarray(p1,float).reshape(-1,2)
    if len(p0)!=len(p1):raise ValueError('unequal points')
    good=sample_valid(valid[0],p0)&sample_valid(valid[1],p1)
    e,d=fit(p0,p1);inliers=e.inlier_mask if e.inlier_mask is not None else np.zeros(len(p0),bool)
    maperr=np.linalg.norm(p1-mapper(p0),axis=1) if len(p0) else np.empty(0)
    support={f'{kind}{i}':spatial(p,(512,512)) for kind,pts in [('matches',[p0,p1]),('inliers',[p0[inliers&good],p1[inliers&good]])] for i,p in enumerate(pts)}
    tiles=[];masks=[]
    for gy in range(4):
        for gx in range(4):
            ids=np.flatnonzero((p0[:,0]>=gx*128)&(p0[:,0]<(gx+1)*128)&(p0[:,1]>=gy*128)&(p0[:,1]<(gy+1)*128))
            # Use exactly the global native full-gate rules; do not normalize a
            # tiny tile to enlarge its spatial support or relax any thresholds.
            le,ld=fit(p0[ids],p1[ids]);lm=le.inlier_mask if le.inlier_mask is not None else np.zeros(len(ids),bool)
            tiles.append(dict(gx=gx,gy=gy,n_matches=len(ids),n_inliers=le.n_inliers,inlier_ratio=le.inlier_ratio,
                output_available=le.candidate_H is not None,accepted=le.accepted,reason=le.reason,candidate_H=le.candidate_H,
                diagnostics=ld,match_indices=ids,inlier_indices=ids[lm],scope='source128tile_only; full-gate thresholds in native512 frame'))
            masks.append(ids[lm])
    r={k:row[k] for k in ['sample_id','sensor','component','look_pair_category','site_name']}
    geo_med=float(np.median(maperr[np.isfinite(maperr)])) if np.isfinite(maperr).any() else None
    chosen=maperr[inliers&good];chosen=chosen[np.isfinite(chosen)]
    r.update(method=method,protocol=protocol,n_matches=len(p0),n_native_valid_matches=int(good.sum()),n_inliers=e.n_inliers,
        n_native_valid_inliers=int((good&inliers).sum()),inlier_ratio=e.inlier_ratio,accepted=e.accepted,reason=e.reason,
        output_available=e.candidate_H is not None,candidate_H=e.candidate_H,fit_inlier_rmse_NOT_GT=e.inlier_rmse,
        geometry=d,spatial=support,geo_all_median_px=geo_med,
        geo_all_p90_px=float(np.quantile(maperr[np.isfinite(maperr)],.9)) if np.isfinite(maperr).any() else None,
        geo_inlier_median_px=float(np.median(chosen)) if len(chosen) else None,
        geo_inlier_p90_px=float(np.quantile(chosen,.9)) if len(chosen) else None,
        local_output_tiles=sum(t['output_available'] for t in tiles),local_accepted_tiles=sum(t['accepted'] for t in tiles),
        global_scope='engineering_candidate_NOT_correctness',GT_evidence='C=0; no independent accuracy',
        decision='accepted_unassessable' if e.accepted else ('rejected' if e.candidate_H is not None else 'no_output'),
        timing=timing,diagnostic_ms=1000*(time.perf_counter()-start),extra=extra or {})
    return clean(r),clean(tiles),dict(points0=p0,points1_native=p1,confidence=np.asarray(confidence),inlier_mask=inliers,
        native_valid_mask=good,geographic_moving=mapper(p0),geo_difference_px=maperr)
