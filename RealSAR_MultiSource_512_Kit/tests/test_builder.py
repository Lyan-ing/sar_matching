import csv,sys,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_real_pairs import read_manifest,read_common_window,read_moving_native_window,spatially_disjoint,build,to_db,source_fingerprint

def test_quantized_zero_respects_provider_nodata_mask(tmp_path):
    data=np.ones((512,512),dtype='uint8');data[20:100,20:100]=0
    for nodata in [None,0]:
        path=tmp_path/f'zero_{nodata}.tif'
        with rasterio.open(path,'w',driver='GTiff',height=512,width=512,count=1,dtype='uint8',crs='EPSG:32614',transform=from_origin(0,512,1,1),nodata=nodata) as d:d.write(data,1)
        with rasterio.open(path) as d:
            a,b,va,vb,t=read_common_window(d,d,Window(0,0,512,512),'linear_amplitude',True)
            raw,vnative,_,_=read_moving_native_window(d,d,Window(0,0,512,512),'linear_amplitude',True)
            assert bool(va[30,30])==(nodata is None)
            assert bool(vb[30,30])==(nodata is None)
            assert bool(vnative[30,30])==(nodata is None)
            assert raw[30,30]==0 if nodata is None else np.isnan(raw[30,30])
            assert va[200,200] and vb[200,200] and vnative[200,200]

COLUMNS=['pair_id','site_id','split','sensor','reference_acquisition_id',
'moving_acquisition_id','reference_path','moving_path','reference_utc','moving_utc','input_units']

def make_raster(path,scale=1):
    img=np.random.default_rng(4).uniform(.003,.2,(1200,1200)).astype('float32')*scale
    with rasterio.open(path,'w',driver='GTiff',height=1200,width=1200,count=1,
            dtype='float32',crs='EPSG:32614',transform=from_origin(0,24000,20,20),nodata=0) as ds: ds.write(img,1)

def make_manifest(tmp_path):
    a=tmp_path/'a.tif';b=tmp_path/'b.tif';make_raster(a);make_raster(b,1.1)
    m=tmp_path/'source.csv'
    row=dict(zip(COLUMNS,['p1','site1','train','Sentinel-1','acqA','acqB',str(a),str(b),
                          '2020-05-09Z','2020-05-21Z','linear_power']))
    with m.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=COLUMNS);w.writeheader();w.writerow(row)
    return m,a,b

def test_source_manifest_split_audit(tmp_path):
    m,a,b=make_manifest(tmp_path)
    assert len(read_manifest(m))==1
    with m.open('a',newline='') as f:
        csv.writer(f).writerow(['p2','site1','val','Sentinel-1','acqC','acqD',str(a),str(b),'a','b','linear_power'])
    with pytest.raises(ValueError,match='leakage'):read_manifest(m)

def test_same_acquisition_denied(tmp_path):
    m,a,b=make_manifest(tmp_path)
    s=m.read_text().replace('acqB','acqA');m.write_text(s)
    with pytest.raises(ValueError,match='Same acquisition'):read_manifest(m)

def test_reprojection_real_pair(tmp_path):
    m,a,b=make_manifest(tmp_path)
    with rasterio.open(a) as r,rasterio.open(b) as t:
        x,y,va,vb,aff=read_common_window(r,t,Window(50,100,512,512))
        assert x.shape==(512,512) and va.all() and vb.all()
        assert np.allclose(y,x*1.1,rtol=1e-5)

def test_512_crops_and_no_gt(tmp_path):
    m,a,b=make_manifest(tmp_path)
    opts=SimpleNamespace(output=str(tmp_path/'result'),seed=42,chip_size=512,patch_size=128,
        chips_per_pair=2,positive_per_chip=1,negative_per_chip=1,min_valid=.98,
        min_texture_db=.5,db_min=-25,db_max=0,max_attempts=50,fail_fast=True)
    summary=build(read_manifest(m),opts)
    assert summary['candidate_chip_pairs']==2
    assert summary['patches_generated']==4
    one=json.loads((tmp_path/'result'/'train'/'p1'/'chip_000'/'metadata.json').read_text())
    assert one['GT_STATUS']=='COARSE_GEO_ONLY__INDEPENDENT_GCP_REQUIRED'
    assert one['size_px']==[512,512]
    chips=[json.loads(x.read_text()) for x in (tmp_path/'result').rglob('metadata.json')]
    x,y=chips[0]['reference_window_xy'];xx,yy=chips[1]['reference_window_xy']
    assert spatially_disjoint(x,y,xx,yy,512,0)
    assert one['reference_acquisition_id']!=''+one['moving_acquisition_id']
    with pytest.raises(FileExistsError):build(read_manifest(m),opts)

def test_remote_url_not_pretend_full_sha():
    assert source_fingerprint('https://example.org/cloud.tif')['content_sha256'] is None

def test_128_patch_negative_spatial_separation():
    assert spatially_disjoint(0,0,200,5,128)
    assert not spatially_disjoint(0,0,100,5,128)

def test_released_window_replay_is_byte_identical(tmp_path):
    from replay_released_windows import replay
    from build_real_pairs import sha256_file
    m,a,b=make_manifest(tmp_path)
    opts=SimpleNamespace(output=str(tmp_path/'original'),seed=42,chip_size=512,patch_size=128,chips_per_pair=1,positive_per_chip=0,negative_per_chip=0,min_valid=.98,min_texture_db=.5,db_min=-25,db_max=0,max_attempts=30,fail_fast=True)
    build(read_manifest(m),opts);folder=tmp_path/'original/train/p1/chip_000'
    lock=tmp_path/'windows.jsonl';lock.write_text(json.dumps({'metadata_relative_path':str((folder/'metadata.json').relative_to(tmp_path)),'expected_file_sha256':{p.name:sha256_file(p) for p in folder.iterdir() if p.is_file()}})+'\n')
    assert replay(lock,tmp_path/'reproduced')==1
    assert json.loads((tmp_path/'reproduced/SUMMARY.json').read_text())['all_file_sha256_equal_originals']

def test_amplitude_db():
    assert np.allclose(to_db(np.array([10.],dtype='float32'),'linear_amplitude'),20.)

def test_integer_gec_nodata_preserves_native_pixels(tmp_path):
    image=np.full((800,800),1000,dtype='uint16');image[145,145]=0
    paths=[tmp_path/'uint_a.tif',tmp_path/'uint_b.tif']
    for path in paths:
        with rasterio.open(path,'w',driver='GTiff',height=800,width=800,count=1,dtype='uint16',
             crs='EPSG:32614',transform=from_origin(600000,4800000,.4,.4),nodata=0) as ds:ds.write(image,1)
    with rasterio.open(paths[0]) as a,rasterio.open(paths[1]) as b:
        ref,mov,valid,_,_=read_common_window(a,b,Window(100,100,512,512),'linear_amplitude')
        raw,native_valid,_,origin=read_moving_native_window(a,b,Window(100,100,512,512),'linear_amplitude')
        assert origin==[100,100]
        assert np.isnan(ref[45,45]) and not valid[45,45]
        assert np.array_equal(ref,raw,equal_nan=True) and np.array_equal(valid,native_valid)

def test_bounded_warp_matches_full_raster_for_rotated_grid(tmp_path):
    from rasterio.warp import reproject
    from rasterio.enums import Resampling
    from affine import Affine
    paths=[tmp_path/'north.tif',tmp_path/'rotated.tif']
    data=np.random.default_rng(5).uniform(1,100,(1200,1200)).astype('float32');data[:70]=0
    transforms=[from_origin(600000,4800000,.4,.4),from_origin(600000,4800000,.4,.4)*Affine.rotation(17)]
    for path,t in zip(paths,transforms):
        with rasterio.open(path,'w',driver='GTiff',height=1200,width=1200,count=1,dtype='float32',crs='EPSG:32614',transform=t,nodata=0) as ds:ds.write(data,1)
    with rasterio.open(paths[0]) as a,rasterio.open(paths[1]) as b:
        _,bounded,_,_,t=read_common_window(a,b,Window(120,120,512,512),'linear_amplitude')
        full=np.full((512,512),-9999,dtype='float32')
        reproject(source=rasterio.band(b,1),destination=full,src_transform=b.transform,src_crs=b.crs,src_nodata=0,
                  dst_transform=t,dst_crs=a.crs,dst_nodata=-9999,resampling=Resampling.bilinear,XSCALE=1,YSCALE=1)
        assert np.allclose(full,bounded,atol=1e-4)
