"""Build every georeference pair and native-only geographic assisted view."""
import argparse,json,csv,sys,time,subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np,cv2
from PIL import Image,ImageDraw
from common import ROOT,SOURCE,MANUAL,source_rows,save,sha,configure
from geo import *

def main(out):
    out=Path(out).resolve();cfg=configure(out);assert not (out/'georeference_correspondences/SUMMARY.json').exists()
    save(out/'preregistration.json',{'created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'config_sha256':sha(out/'config.json'),'source_index_sha256':sha(SOURCE/'dataset_index.jsonl'),'old_manual_receipt_sha256':sha(MANUAL/'ENGINEERING_ACCEPTANCE.json'),'source_delivery_receipt_sha256':sha(SOURCE/'DELIVERY_COMPLETE.json'),'C_level':0,'config_fixed_before_geographic_and_feature_results':True})
    rows=source_rows();results=[];cv2.setNumThreads(1)
    for i,row in enumerate(rows):
        folder=out/'georeference_correspondences'/row['sample_id'];folder.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
        try:
            arrays,valid,images,infos,mapper=load_pair(row);masks,mapped,counts=overlap(valid,mapper);grids=[];correspondences=[];check=[]
            for side,fn in enumerate([mapper,mapper.reverse()]):
                p=regular_grid(valid[side].shape,cfg['grid_size']);q=fn(p);keep=sample_valid(valid[side],p)&sample_valid(valid[1-side],q);p,q=p[keep],q[keep];check.append(roundtrip(fn,p));grids.append((p,q))
                if len(p)==0 and masks[side].any():
                    # Validate numerics on available real overlap, without inventing grid points.
                    dense=pixel_grid(valid[side].shape)[masks[side].ravel()];idx=np.linspace(0,len(dense)-1,min(256,len(dense))).astype(int);check[-1]=roundtrip(fn,dense[idx])
                correspondences.append([{'point_id':f'G{side}_{j:04d}','xy0':a.tolist() if side==0 else b.tolist(),'xy1':b.tolist() if side==0 else a.tolist(),'label_type':'georeference_correspondences','sampled_from_view':side} for j,(a,b) in enumerate(zip(p,q))])
            grade=bool(row['reference_acquisition_id']!=row['moving_acquisition_id'] and all(c['joint_pixels']>0 for c in counts) and all(c['finite'] and c['max_px'] is not None and c['max_px']<=cfg['roundtrip_tolerance_px'] for c in check))
            metric=[metric_geometry(info) for info in infos];iq=[image_quality(a,v,im) for a,v,im in zip(arrays,valid,images)]
            record={'sample_id':row['sample_id'],'A_qualified':grade,'label_type':'georeference_correspondences','independent_GT':False,'infos':infos,'footprints':[footprint(info) for info in infos],'metric_geometry':metric,'scale_ratio_0_over_1':[metric[0]['spacing_m_x']/metric[1]['spacing_m_x'],metric[0]['spacing_m_y']/metric[1]['spacing_m_y']],'overlap':counts,'roundtrip':check,'grid_points_by_view':[len(x) for x in correspondences],'image_quality':iq,'map_affine_0_to_1':mapper.affine_matrix(),'crs_equal':mapper.crsA==mapper.crsB,'failure_reason':None if grade else 'no_common_valid_coverage_or_roundtrip_failed','dynamic_exclusion_verified':False,'product_geolocation_accuracy_known':False}
            save(folder/'geographic.json',record);save(folder/'correspondences.json',{'sample_id':row['sample_id'],'label_type':'georeference_correspondences','independent_GT':False,'points':correspondences[0],'reverse_sampled_points':correspondences[1]})
            with (folder/'correspondences.csv').open('x',newline='') as f:
                writer=csv.writer(f);writer.writerow(['point_id','x0','y0','x1','y1','label_type','sampled_from_view']);writer.writerows([p['point_id'],*p['xy0'],*p['xy1'],p['label_type'],p['sampled_from_view']] for batch in correspondences for p in batch)
            for side,m in enumerate(masks):Image.fromarray(m.astype(np.uint8)*255).save(folder/f'geographic_valid_overlap{side}.png')
            # Only existing 512 native moving pixels contribute; coarse views from
            # the old full scene may contain points outside this native crop.
            q=mapped[0].reshape(*valid[0].shape,2);warped=cv2.remap(images[1],q[:,:,0].astype(np.float32),q[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0);warped[~masks[0]]=0;Image.fromarray(warped).save(folder/'moving_native_geo_assisted.png')
            canvas=Image.new('RGB',(1536,1080),'white');d=ImageDraw.Draw(canvas);d.text((4,3),f"{row['sample_id']} {row['sensor']} {row['site_name']} | GEOGRAPHIC REFERENCES ONLY, no independent GT",fill='black');d.text((4,20),f"look={row['look_pair_category']} overlap0={counts[0]['fraction_of_all_pixels']:.3f} overlap1={counts[1]['fraction_of_all_pixels']:.3f}",fill='black')
            canvas.paste(Image.fromarray(images[0]).convert('RGB'),(0,50));canvas.paste(Image.fromarray(images[1]).convert('RGB'),(512,50));canvas.paste(Image.fromarray(np.stack([images[0],warped,warped],-1)),(1024,50))
            for j,p in enumerate(correspondences[0][::max(1,len(correspondences[0])//24)]):d.line((*p['xy0'][:1],p['xy0'][1]+50,p['xy1'][0]+512,p['xy1'][1]+50),fill=(255,180,20),width=1)
            for side in [0,1]:
                im=np.stack([images[side]]*3,-1);im[masks[side],1]=255;canvas.paste(Image.fromarray(im),(side*512,565))
            residual=cv2.absdiff(images[0],warped);residual[~masks[0]]=0;canvas.paste(Image.fromarray(residual).convert('RGB'),(1024,565));d.text((1028,575),'display difference, NOT registration error',fill='yellow');canvas.save(folder/'preview.png')
            record['runtime_ms']=1000*(time.perf_counter()-start);save(folder/'runtime.json',{'runtime_ms':record['runtime_ms']});results.append(record)
        except Exception as e:
            record={'sample_id':row['sample_id'],'A_qualified':False,'label_type':'georeference_correspondences','independent_GT':False,'failure_reason':repr(e)};save(folder/'FAILURE.json',record);results.append(record)
        if (i+1)%20==0:print('Geographic pairs',i+1,'/',len(rows),flush=True)
    save(out/'georeference_correspondences/SUMMARY.json',{'attempted':191,'A_qualified':sum(r['A_qualified'] for r in results),'records':results,'C_qualified':0});print('A',sum(r['A_qualified'] for r in results),'/',len(results),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default=str(ROOT));a=p.parse_args();main(a.out)
