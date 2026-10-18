#!/usr/bin/env python3
"""Build source-safe *candidate* SAR registration crops from distinct real acquisitions.

Requires already georeferenced, real SAR products. Coordinates obtained by map reprojection
are COARSE preprocessing, never independent registration truth. No synthetic-view replacement.
Works with local GeoTIFF or public HTTPS Cloud Optimized GeoTIFF (range requests via GDAL).
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict, Counter
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import reproject, transform as transform_coords

REQUIRED = ('pair_id', 'site_id', 'split', 'sensor', 'reference_acquisition_id',
            'moving_acquisition_id', 'reference_path', 'moving_path', 'reference_utc',
            'moving_utc', 'input_units')
SPLITS = {'train', 'dev', 'val', 'test', 'confirmation'}
UNITS = {'linear_power', 'linear_amplitude', 'db'}
REMOTE = ('https://', 'http://')


def safe_name(s):
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,119}', s):
        raise ValueError('Unsafe ID (use ASCII letters, numbers, underscore, hyphen): '+repr(s))
    return s


def is_remote(s):
    return s.startswith(REMOTE)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def source_fingerprint(path):
    """Do not pretend URL string hash represents remote COG bytes."""
    if is_remote(path):
        return {'url': path, 'content_sha256': None, 'hash_status': 'remote_bytes_not_downloaded'}
    return {'path': path, 'content_sha256': sha256_file(path), 'hash_status': 'local_full_file_sha256'}


def read_manifest(path):
    src = Path(path).resolve()
    with src.open(newline='', encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        if not set(REQUIRED).issubset(rd.fieldnames or []):
            raise ValueError('Missing required columns: '+','.join(sorted(set(REQUIRED)-set(rd.fieldnames or []))))
        rows = list(rd)
    if not rows:
        raise ValueError('Empty source manifest')
    pair_ids = set()
    site_splits, acquisition_splits, path_splits = defaultdict(set), defaultdict(set), defaultdict(set)
    for r in rows:
        for field in REQUIRED:
            if not str(r.get(field) or '').strip():
                raise ValueError(f'Empty {field} for pair {r.get("pair_id")}')
            r[field] = str(r[field]).strip()
        safe_name(r['pair_id'])
        safe_name(r['site_id'])
        if r['pair_id'] in pair_ids:
            raise ValueError('Duplicate pair_id: '+r['pair_id'])
        pair_ids.add(r['pair_id'])
        if r['split'] not in SPLITS:
            raise ValueError('Invalid split: '+r['split'])
        if r['input_units'] not in UNITS:
            raise ValueError('Unsupported input_units: '+r['input_units'])
        if r['reference_acquisition_id'] == r['moving_acquisition_id']:
            raise ValueError('Same acquisition cannot form real pair: '+r['pair_id'])
        site_splits[r['site_id']].add(r['split'])
        for side in ('reference', 'moving'):
            acquisition_splits[(r['sensor'], r[side+'_acquisition_id'])].add(r['split'])
            raw = r[side+'_path']
            if is_remote(raw):
                if urlsplit(raw).scheme != 'https':
                    raise ValueError('Only https COG URLs are allowed: '+raw)
                p = raw
            else:
                pp = Path(raw)
                if not pp.is_absolute():
                    pp = (src.parent/pp).resolve()
                if not pp.is_file():
                    raise FileNotFoundError(pp)
                p = str(pp)
            r[side+'_path'] = p
            path_splits[p].add(r['split'])
    for label, d in (('site_id', site_splits), ('acquisition', acquisition_splits), ('source_path',path_splits)):
        errors = {str(k): sorted(v) for k,v in d.items() if len(v)>1}
        if errors:
            raise ValueError('Cross-split source leakage in '+label+': '+repr(list(errors.items())[:8]))
    return rows


def to_db(a, units):
    with np.errstate(divide='ignore', invalid='ignore'):
        if units == 'linear_power':
            return 10*np.log10(np.maximum(a, 1e-12))
        if units == 'linear_amplitude':
            return 20*np.log10(np.maximum(a, 1e-12))
        if units == 'db':
            return a
    raise ValueError(units)


def display(a, valid, lo, hi, units):
    scaled = np.clip((to_db(a,units)-lo)/(hi-lo),0,1)
    return np.where(valid,np.rint(scaled*255),0).astype(np.uint8)


def read_common_window(reference, moving, win, units='linear_power',allow_valid_zero=False):
    """Reference native grid + one actual warp of moving acquisition into the reference map grid."""
    if not reference.crs or not moving.crs:
        raise ValueError('Both products must have real georeferencing')
    ref_masked = reference.read(1, window=win, masked=True)
    a = ref_masked.astype('float32').filled(np.nan)
    va = (~np.ma.getmaskarray(ref_masked)) & np.isfinite(a)
    if units != 'db':
        va &= a>=0 if allow_valid_zero else a>0
    b = np.full(a.shape,-9999,dtype='float32')
    t = reference.window_transform(win)
    # Bound remote reads to the destination footprint, including rotated native GEC grids.
    edge=np.linspace(0,win.width,9);hh=float(win.height)
    pixels=[(v,0) for v in edge]+[(v,hh) for v in edge]+[(0,v*hh/win.width) for v in edge]+[(win.width,v*hh/win.width) for v in edge]
    coords=[t*p for p in pixels]
    xx,yy=transform_coords(reference.crs,moving.crs,[p[0] for p in coords],[p[1] for p in coords])
    native=[(~moving.transform)*(x,y) for x,y in zip(xx,yy)]
    left=max(0,int(np.floor(min(p[0] for p in native)))-2)
    top=max(0,int(np.floor(min(p[1] for p in native)))-2)
    right=min(moving.width,int(np.ceil(max(p[0] for p in native)))+2)
    bottom=min(moving.height,int(np.ceil(max(p[1] for p in native)))+2)
    if right>left and bottom>top:
        mw=Window(left,top,right-left,bottom-top)
        source=moving.read(1,window=mw,masked=True).astype('float32').filled(-9999)
        reproject(source=source, destination=b,
                  src_transform=moving.window_transform(mw), src_crs=moving.crs, src_nodata=-9999,
                  dst_transform=t, dst_crs=reference.crs,dst_nodata=-9999,
                  resampling=Resampling.bilinear,init_dest_nodata=True,XSCALE=1,YSCALE=1)
    vb = np.isfinite(b)&(b!=-9999)
    if moving.nodata is not None:
        vb &= b!=moving.nodata
    if units != 'db':
        vb &= b>=0 if allow_valid_zero else b>0
    return a,b,va,vb,t


def read_moving_native_window(reference, moving, ref_win, units,allow_valid_zero=False):
    """Read a separate native moving-raster crop at the *approximate* reference map center.

    Does not warp/align source samples; preserves actual moving pixels and its own map grid.
    Pair center uses geolocation only for image discovery, never as pixel-exact ground truth.
    """
    half=ref_win.width/2
    x,y=reference.xy(ref_win.row_off+half,ref_win.col_off+half)
    xx,yy=transform_coords(reference.crs,moving.crs,[x],[y])
    movrow,movcol=moving.index(xx[0],yy[0])
    w=int(ref_win.width);h=int(ref_win.height)
    left=int(movcol-w//2);top=int(movrow-h//2)
    if left<0 or top<0 or left+w>moving.width or top+h>moving.height:
        raise ValueError('Reference center near moving source boundary; native crop unavailable')
    win=Window(left,top,w,h)
    m=moving.read(1,window=win,masked=True)
    raw=m.astype('float32').filled(np.nan)
    valid=(~np.ma.getmaskarray(m)) & np.isfinite(raw)
    if units!='db':valid &= raw>=0 if allow_valid_zero else raw>0
    return raw,valid,moving.window_transform(win),[left,top]


def save_tiff(path,a,valid,crs,transform):
    profile={'driver':'GTiff','height':a.shape[0],'width':a.shape[1],
        'count':1,'dtype':'float32','crs':crs,'transform':transform,
        'nodata':-9999,'compress':'deflate','tiled':True,
        'blockxsize':256,'blockysize':256}
    with rasterio.open(path,'w',**profile) as d:
        d.write(np.where(valid,a,-9999).astype('float32'),1)


def spatially_disjoint(ax,ay,bx,by,size,gap=16):
    return abs(ax-bx)>=size+gap or abs(ay-by)>=size+gap


def small_patches(root,a,b,va,vb,rng,size,positive_count,negative_count,units,lo,hi):
    h,w = a.shape
    if size>=min(h,w) and (positive_count or negative_count):
        raise ValueError('Patch size must be smaller than crop size')
    out=[]
    for kind,count in (('positive_candidate',positive_count),('negative_candidate',negative_count)):
        n=0
        for _ in range(max(100,count*150)):
            if n>=count:break
            x=int(rng.integers(0,w-size+1));y=int(rng.integers(0,h-size+1))
            if kind=='positive_candidate':xx,yy=x,y
            else:
                xx=int(rng.integers(0,w-size+1)); yy=int(rng.integers(0,h-size+1))
                if not spatially_disjoint(x,y,xx,yy,size):continue
            if va[y:y+size,x:x+size].mean()<.99 or vb[yy:yy+size,xx:xx+size].mean()<.99:
                continue
            dst=root/f'{kind}_{n:03d}'
            dst.mkdir(parents=True)
            Image.fromarray(display(a[y:y+size,x:x+size],va[y:y+size,x:x+size],lo,hi,units)).save(dst/'a.png')
            Image.fromarray(display(b[yy:yy+size,xx:xx+size],vb[yy:yy+size,xx:xx+size],lo,hi,units)).save(dst/'b.png')
            out.append({'type':kind,'a_xy':[x,y],'b_xy':[xx,yy],
                        'status':'PROVISIONAL_NOT_INDEPENDENT_GT'})
            n+=1
        if n<count:
            raise ValueError(f'Insufficient nonoverlapping valid areas for {kind}: {n}/{count}')
    return out


def build(rows,opts):
    out=Path(opts.output).resolve()
    if out.exists():
        raise FileExistsError('Will not overwrite existing output: '+str(out))
    out.mkdir(parents=True)
    rng=np.random.default_rng(opts.seed)
    candidates=[]; failures=[]; coverage=[]
    sources={}
    for row in rows:
        for side in ('reference','moving'):
            src=row[side+'_path']
            if src not in sources:
                sources[src]=source_fingerprint(src)
        print(f"[{row['split']}] {row['pair_id']}",flush=True)
        pairdir=out/row['split']/row['pair_id']
        pairdir.mkdir(parents=True)
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                              GDAL_HTTP_MAX_RETRY='3',GDAL_HTTP_RETRY_DELAY='2',
                              GDAL_HTTP_TIMEOUT='40',
                              AWS_NO_SIGN_REQUEST='YES'):
                with rasterio.open(row['reference_path']) as ref, rasterio.open(row['moving_path']) as mov:
                    if ref.count!=1 or mov.count!=1:
                        raise ValueError('Expected single-band intensity SAR GeoTIFF; select band before import')
                    if not ref.crs or not mov.crs:
                        raise ValueError('Georeferenced products required')
                    if min(ref.width,ref.height)<opts.chip_size:
                        raise ValueError('Source native raster is smaller than requested chip size')
                    if row.get('product_level') and row.get('reference_product_level') and row.get('moving_product_level'):
                        if row['reference_product_level']!=row['moving_product_level']:
                            raise ValueError('Source product levels differ: pre-process separately and document')
                    accepted=0; sampled=set(); accepted_windows=[]
                    roi=json.loads(row.get('reference_roi_xyxy') or 'null') or [0,0,ref.width,ref.height]
                    xmin,ymin,xmax,ymax=map(int,roi)
                    if xmax-xmin<opts.chip_size or ymax-ymin<opts.chip_size:raise ValueError('ROI smaller than native chip')
                    for trial in range(opts.max_attempts):
                        if accepted>=opts.chips_per_pair:break
                        x=int(rng.integers(xmin,xmax-opts.chip_size+1))
                        y=int(rng.integers(ymin,ymax-opts.chip_size+1))
                        cell=(x//max(1,opts.chip_size//2),y//max(1,opts.chip_size//2))
                        if cell in sampled:continue
                        sampled.add(cell)
                        if any(not spatially_disjoint(x,y,xx,yy,opts.chip_size,0) for xx,yy in accepted_windows):
                            continue
                        win=Window(x,y,opts.chip_size,opts.chip_size)
                        allow_zero=getattr(opts,'allow_valid_zero',False)
                        a,b,va,vb,t=read_common_window(ref,mov,win,row['input_units'],allow_zero)
                        try:
                            b_native,vb_native,t_native,mov_origin=read_moving_native_window(ref,mov,win,row['input_units'],allow_zero)
                        except ValueError:
                            continue
                        # Combined valid mask is meaningful ONLY for the coarsely map-aligned pair.
                        joint=va&vb
                        coverage.append({'pair_id':row['pair_id'],'candidate_xy':[x,y],
                                         'joint_valid_fraction':float(joint.mean())})
                        if float(joint.mean())<opts.min_valid or float(vb_native.mean())<opts.min_valid:continue
                        ta=joint&(a>0) if row['input_units']!='db' else joint
                        tb=joint&(b>0) if row['input_units']!='db' else joint
                        if min(ta.sum(),tb.sum())<256:continue
                        sa=float(np.std(to_db(a[ta],row['input_units'])))
                        sb=float(np.std(to_db(b[tb],row['input_units'])))
                        if min(sa,sb)<opts.min_texture_db:continue
                        lo=float(row.get('display_db_min') or opts.db_min)
                        hi=float(row.get('display_db_max') or opts.db_max)
                        if row.get('display_mode')=='joint_native_percentiles':
                            da=va&(a>0) if row['input_units']!='db' else va
                            db=vb_native&(b_native>0) if row['input_units']!='db' else vb_native
                            values=np.concatenate([to_db(a[da],row['input_units']),to_db(b_native[db],row['input_units'])])
                            lo,hi=map(float,np.percentile(values,[2,98]))
                        if not np.isfinite([lo,hi]).all() or hi<=lo:raise ValueError('Invalid display bounds')
                        chipid=f'chip_{accepted:03d}'
                        chipdir=pairdir/chipid
                        chipdir.mkdir()
                        save_tiff(chipdir/'reference_float.tif',a,va,ref.crs,t)
                        save_tiff(chipdir/'moving_float.tif',b,vb,ref.crs,t)
                        save_tiff(chipdir/'moving_native_float.tif',b_native,vb_native,mov.crs,t_native)
                        Image.fromarray(display(a,va,lo,hi,row['input_units'])).save(chipdir/'reference.png')
                        Image.fromarray(display(b,vb,lo,hi,row['input_units'])).save(chipdir/'moving.png')
                        Image.fromarray(display(b_native,vb_native,lo,hi,row['input_units'])).save(chipdir/'moving_native.png')
                        Image.fromarray(va.astype('uint8')*255).save(chipdir/'reference_valid.png')
                        Image.fromarray(joint.astype('uint8')*255).save(chipdir/'common_valid.png')
                        Image.fromarray(vb_native.astype('uint8')*255).save(chipdir/'moving_native_valid.png')
                        patches=small_patches(chipdir/'patches',a,b,va,vb,rng,opts.patch_size,
                            opts.positive_per_chip,opts.negative_per_chip,row['input_units'],lo,hi)
                        entry={'pair_id':row['pair_id'],'chip_id':chipid,'split':row['split'],
                          'site_id':row['site_id'],'sensor':row['sensor'],'input_units':row['input_units'],
                          'reference_acquisition_id':row['reference_acquisition_id'],
                          'moving_acquisition_id':row['moving_acquisition_id'],
                          'reference_utc':row['reference_utc'],'moving_utc':row['moving_utc'],
                          'reference_path':row['reference_path'],'moving_path':row['moving_path'],
                          'source_refs':{'reference':sources[row['reference_path']],
                                         'moving':sources[row['moving_path']]},
                          'reference_window_xy':[x,y], 'moving_native_window_xy':mov_origin,
                          'size_px':[opts.chip_size,opts.chip_size],
                          'crs':str(ref.crs),'map_transform_first6':list(t)[:6],
                          'native_moving_crs':str(mov.crs),'native_moving_transform_first6':list(t_native)[:6],
                          'reference_pixel_spacing_grid_units':[float(np.hypot(t.a,t.d)),float(np.hypot(t.b,t.e))],
                          'moving_native_pixel_spacing_grid_units':[float(np.hypot(t_native.a,t_native.d)),float(np.hypot(t_native.b,t_native.e))],
                          'reference_crs_units':ref.crs.linear_units if ref.crs.is_projected else 'degree',
                          'moving_native_crs_units':mov.crs.linear_units if mov.crs.is_projected else 'degree',
                          'common_valid_fraction':float(joint.mean()),
                          'moving_native_valid_fraction':float(vb_native.mean()),
                          'allow_declared_valid_zero':allow_zero,
                          'zero_policy':'Retain nonnegative samples only where provider mask declares valid; declared nodata stays excluded' if allow_zero else 'Conservatively exclude nonpositive amplitude/power, including quantized zero',
                          'db_std':[sa,sb],
                          'display_db_range':[lo,hi],
                          'coarse_resampling':{'kernel':'bilinear','XSCALE':1,'YSCALE':1,'source_window_halo_px':2,'note':'Explicit interpolation footprint avoids GDAL chunk-dependent scale estimation; native imagery unchanged'},
                          'source_attributes':{k:v for k,v in row.items() if k not in REQUIRED},
                          'patches':patches,
                          'GT_STATUS':'COARSE_GEO_ONLY__INDEPENDENT_GCP_REQUIRED',
                          'geocoding_note':'reference and moving_native preserve each raster sampled pixels; moving_float is warped to reference grid for coarse prealignment; neither is independent GT'}
                        (chipdir/'metadata.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2),encoding='utf-8')
                        candidates.append(entry)
                        accepted_windows.append((x,y))
                        accepted+=1
                    if accepted<opts.chips_per_pair:
                        failures.append({'pair_id':row['pair_id'],'reason':'not_enough_valid_textured_nonredundant_crops',
                            'requested':opts.chips_per_pair,'completed':accepted})
        except Exception as exc:
            failures.append({'pair_id':row['pair_id'],'reason':type(exc).__name__+': '+str(exc)})
            if opts.fail_fast:
                raise
    with (out/'candidates.jsonl').open('w',encoding='utf-8') as f:
        for obj in candidates:f.write(json.dumps(obj,ensure_ascii=False)+'\n')
    summary={'dataset_type':'REAL_DISTINCT_ACQUISITION_COARSE_GEO_CANDIDATES',
      'candidate_chip_pairs':len(candidates),'requested_source_pairs':len(rows),
      'chips_per_pair_target':opts.chips_per_pair,'patches_generated':sum(len(c['patches']) for c in candidates),
      'by_sensor':dict(Counter(x['sensor'] for x in candidates)),
      'by_split':dict(Counter(x['split'] for x in candidates)),
      'unique_sources':len(sources), 'unique_sites':len(set(x['site_id'] for x in candidates)),
      'seed':opts.seed,'size_px':opts.chip_size,
      'status':'CANDIDATES_NOT_INDEPENDENT_GT','candidate_windows_evaluated':len(coverage),
      'failures':failures,'requirements':'Review original acquisitions, coverage, preprocessing, source separation, independently verified GCPs'}
    (out/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'window_coverage_audit.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in coverage),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--chip-size',type=int,default=512)
    p.add_argument('--patch-size',type=int,default=128)
    p.add_argument('--chips-per-pair',type=int,default=4)
    p.add_argument('--positive-per-chip',type=int,default=0)
    p.add_argument('--negative-per-chip',type=int,default=0)
    p.add_argument('--min-valid',type=float,default=.98)
    p.add_argument('--min-texture-db',type=float,default=.5)
    p.add_argument('--max-attempts',type=int,default=150)
    p.add_argument('--db-min',type=float,default=-25.)
    p.add_argument('--db-max',type=float,default=0.)
    p.add_argument('--seed',type=int,default=260512)
    p.add_argument('--fail-fast',action='store_true')
    p.add_argument('--allow-valid-zero',action='store_true',help='Preserve quantized zero amplitude where provider mask declares it valid; declared nodata is still excluded')
    opt=p.parse_args(argv)
    if opt.chip_size<128 or opt.patch_size<=0 or opt.patch_size>=opt.chip_size:
        p.error('chip size >=128 and 0 < patch size < chip size required')
    if opt.db_min>=opt.db_max or not 0<opt.min_valid<=1:
        p.error('Bad display or valid threshold')
    if opt.chips_per_pair<1 or opt.max_attempts<opt.chips_per_pair:
        p.error('Insufficient attempts')
    build(read_manifest(opt.manifest),opt)

if __name__=='__main__':main()
