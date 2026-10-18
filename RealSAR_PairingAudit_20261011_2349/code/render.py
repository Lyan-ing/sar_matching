"""Native judgement panels and explicitly assisted geocoding panels, no points."""
import sys
sys.dont_write_bytecode = True
import json, shutil, html
from pathlib import Path
import numpy as np
import cv2, rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from PIL import Image, ImageDraw
import audit as a

def polygons(record, path):
    fig,ax=plt.subplots(figsize=(7,7),layout='constrained')
    pp=[np.array(p) for p in record['metric_polygons']];origin=pp[0].mean(axis=0)
    for p,color,style,name in zip(pp,['#0072B2','#D55E00'],['-','--'],['reference footprint','moving footprint']):
        ax.add_patch(Polygon(p-origin,fill=False,edgecolor=color,linestyle=style,lw=2,label=name))
    inter=np.array(record['intersection_metric_polygon'])
    if len(inter):ax.add_patch(Polygon(inter-origin,facecolor='#999999',alpha=.3,hatch='///',label='geographic intersection'))
    for i,p in enumerate(pp):ax.text(*(p[0]-origin),f' {i}: pixel edge (0,0)',fontsize=9)
    ax.autoscale();ax.set_aspect('equal');ax.legend(loc='best',fontsize=9)
    ax.set_xlabel(f"Easting offset (m) | origin {origin[0]:.2f}")
    ax.set_ylabel(f"Northing offset (m) | origin {origin[1]:.2f}")
    fractions=record['polygon_overlap_fractions']
    ax.set_title(f"{record['sample_id']}\n{record['metric_geometry'][0]['local_metric_crs']} | polygon area R/M {fractions[0]:.1%}/{fractions[1]:.1%}\nPROVIDER GEOCODING, NOT SAR CORRESPONDENCE GT",fontsize=10)
    fig.savefig(path,dpi=180);plt.close(fig)

def context(row, record, folder):
    if 'context' not in record['checks'][0]:
        a.dump(folder/'context_status.json',{'available':False,'reason':'no locally retained >512 native source for this crop; no downloads attempted; no synthetic padding','needed':'larger native acquisition window with source GeoTransform'})
        return
    arrays=[];valid=[];infos=[]
    for i in range(2):
        c=record['checks'][i]['context']
        with rasterio.open(c['path']) as ds:
            arrays.append(ds.read(1));valid.append((ds.read_masks(1)>0)&np.isfinite(arrays[-1]));infos.append({'crs':str(ds.crs),'transform':list(ds.transform)[:6],'width':ds.width,'height':ds.height})
    joint,ind,notes=a.display_pair(arrays,valid,[row['reference_input_units'],row['moving_input_units']])
    a.dump(folder/'context_display.json',notes)
    ims=[]
    for i,im in enumerate(ind):
        Image.fromarray(im).save(folder/f'context_native{i}.png')
        rgb=Image.fromarray(im).convert('RGB');d=ImageDraw.Draw(rgb)
        x,y=record['checks'][i]['context']['offset_xy'];d.rectangle((x,y,x+512,y+512),outline=(0,180,255),width=3)
        # Opposite chip polygon on same native context, GEOGRAPHIC assistance.
        own=a.Affine(*infos[i]['transform']);other=a.Affine(*record['checks'][1-i]['native_info']['transform'])
        mapping=a.geo.PixelMap(other,record['checks'][1-i]['native_info']['crs'],own,infos[i]['crs'])
        p=mapping(np.array([[-.5,-.5],[511.5,-.5],[511.5,511.5],[-.5,511.5]]))+.5
        pts=[tuple(x) for x in p];d.line(pts+[pts[0]],fill=(255,160,0),width=3)
        ims.append(np.array(rgb))
    a.panel(ims,['reference: blue own chip / orange opposite footprint','moving: blue own chip / orange opposite footprint'],row['sample_id']+' | TRUE native 1024 contexts; GEOGRAPHIC assistance; NOT GT',folder/'context_footprints.png',1024)
    # Large actual common scene after geography-only remapping; never an input to independent annotation.
    mapper=a.geo.PixelMap(a.Affine(*infos[0]['transform']),infos[0]['crs'],a.Affine(*infos[1]['transform']),infos[1]['crs'])
    q=mapper(a.geo.pixel_grid(arrays[0].shape)).reshape(*arrays[0].shape,2).astype(np.float32)
    warp=cv2.remap(ind[1],q[...,0],q[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    m=valid[0]&a.geo.sample_valid(valid[1],q.reshape(-1,2)).reshape(valid[0].shape)
    warp[~m]=0
    a.panel([ind[0],warp],['native reference 1024','moving 1024 -> reference grid (bilinear)'],row['sample_id']+' | GEOGRAPHIC-assisted context compare; not independent GT',folder/'context_geo_compare.png',1024)
    a.dump(folder/'context_status.json',{'available':True,'real_native_shapes':[list(x.shape) for x in arrays],'central512_checks':[x['context'] for x in record['checks']],'static_physical_correspondences':'pending_human_confirmation'})

def render(row, record, folder, region=None):
    folder.mkdir(parents=True,exist_ok=False)
    record,pack=a.inspect(row,a.read_sources(),contexts)
    arrays,valid,images,infos,mapper,masks,joint,ind=pack
    a.dump(folder/'audit.json',record)
    sid=record['sample_id'];unit=[row['reference_input_units'],row['moving_input_units']]
    labels=[f"R {record['metric_geometry'][0]['coverage_edge_lengths_m'][0]:.1f} m edge",f"M {record['metric_geometry'][1]['coverage_edge_lengths_m'][0]:.1f} m edge"]
    for i,prefix in enumerate(['reference','moving']):
        shutil.copyfile(row[prefix+'_native_png'],folder/f'original{i}.png')
        Image.fromarray(joint[i]).save(folder/f'robust_joint{i}.png')
        Image.fromarray(ind[i]).save(folder/f'robust_independent{i}.png')
        Image.fromarray(valid[i].astype(np.uint8)*255).save(folder/f'valid{i}.png')
        Image.fromarray(masks[i].astype(np.uint8)*255).save(folder/f'geographic_common{i}.png')
    a.panel(images,labels,sid+' | ORIGINAL frozen native PNG | no GT',folder/'original_pair.png')
    a.panel(joint,labels,sid+' | fixed 1-99% JOINT DN dB display | no GT',folder/'robust_joint_pair.png')
    a.panel(ind,labels,sid+' | fixed 1-99% INDEPENDENT DN dB contrast | no GT',folder/'robust_independent_pair.png')
    common=[]
    for im,m in zip(ind,masks):
        rgb=np.repeat(im[...,None],3,axis=2);rgb[~m]=(rgb[~m]*.2).astype(np.uint8);common.append(rgb)
    a.panel(common,labels,sid+' | GEOGRAPHIC overlap highlighted; not physical visibility',folder/'common_coverage.png')
    q=mapper(a.geo.pixel_grid(valid[0].shape)).reshape(512,512,2).astype(np.float32)
    warp=cv2.remap(ind[1],q[...,0],q[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);warp[~masks[0]]=0
    a.panel([ind[0],warp],['reference native','moving -> reference via provider GeoTransform'],sid+' | GEOGRAPHIC-assisted compare; bilinear; NOT independent GT',folder/'geo_compare.png')
    checker=ind[0].copy();yy,xx=np.indices(checker.shape);switch=((xx//64+yy//64)%2)==1;checker[switch]=warp[switch]
    a.panel([checker,np.rint(.5*ind[0]+.5*warp).astype(np.uint8)],['64px geographic checkerboard','50/50 geographic blend'],sid+' | ASSISTED overlay; discrepancy not a GT error metric',folder/'geo_overlay.png')
    polygons(record,folder/'footprints.png')
    # Suggested regions are rough static-structure search ROIs, NOT point pairs.
    if region is None:region=[128,128,384,384]
    x0,y0,x1,y1=region
    center=np.array([[(x0+x1)/2,(y0+y1)/2]])
    qc=mapper(center)[0];sz=min(x1-x0,y1-y0)
    mcenter=np.clip(np.rint(qc),sz//2,512-sz//2).astype(int)
    regions=[region,[int(mcenter[0]-sz//2),int(mcenter[1]-sz//2),int(mcenter[0]+sz//2),int(mcenter[1]+sz//2)]]
    zooms=[]
    for i,(im,roi) in enumerate(zip(ind,regions)):
        x0,y0,x1,y1=roi;cut=im[y0:y1,x0:x1];Image.fromarray(cut).save(folder/f'local_native{i}.png');zooms.append(cut)
    a.panel(zooms,[f'R ROI {regions[0]}',f'M geography-suggested ROI {regions[1]}'],sid+' | local native crops; nearest enlargement only; search ROI NOT GT',folder/'local_zoom.png',768)
    a.dump(folder/'search_regions.json',{'rough_reference_roi_xyxy':regions[0],'moving_roi_xyxy':regions[1],'moving_selection':'geographic center only, bounded native crop; no point correspondences','static_feature':'pending human confirmation','used_in_annotation_ui':False})
    context(row,record,folder)
    return record

if __name__=='__main__':
    a.require_unsealed()
    (a.ROOT/'pairs').mkdir(exist_ok=True)
    contexts={r['sample_id']:r for r in json.loads((a.CONTEXT/'contexts_resolved.json').read_text())['records'] if r['available']}
    rows={r['pair_id']+'__'+r['chip_id']:r for r in [json.loads(l) for l in (a.DATA/'dataset_index.jsonl').read_text().splitlines()]}
    selected=json.loads((a.ROOT/'selection.json').read_text())['candidates']
    ids=list(dict.fromkeys(a.FIVE+[r['sample_id'] for r in selected]))
    for sid in ids:
        item=next((r for r in selected if r['sample_id']==sid),{})
        record=render(dict(rows[sid],sample_id=sid),None,a.ROOT/'pairs'/sid,item.get('reference_structure_roi_xyxy'))
        print('RENDERED',sid,'local_context','context' in record['checks'][0],flush=True)
