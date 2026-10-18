"""Record image-content triage; five task priorities; auxiliary candidates only."""
import sys,json,hashlib,time
from pathlib import Path
from collections import Counter
import numpy as np,cv2,rasterio
from rasterio.warp import transform
from PIL import Image
from prepare import ROOT,sha,write

# Visual triage from native contact sheets, no claimed human correspondences.
CONTENT={
 'Umbra_0014':('mine_terraces','difficult','矿区坡面/台阶可见，但方向变化与非平面地形显著'),
 'Umbra_0019':('mine_shadow','difficult','一幅台阶结构明显，另一幅强暗区；不可把阴影当静态点'),
 'Sentinel1_0008':('mountain_ridges','ready_for_human','较大尺度山脊/谷地结构，20m像素定位有不确定性'),
 'Sentinel1_0009':('mountain_ridges_water','ready_for_human','谷地、水体和坡面结构；避免非平面亮点'),
 'Capella_0001':('volcanic_ridges','ready_for_human','火山地形线性结构清晰；只能由人工确认稳定地物'),
 'Capella_0002':('volcanic_ridges','difficult','近似观测方向，但散斑及山地使点级识别困难'),
 'Umbra_0030':('shore_slope','difficult','暗水域/坡面交界可见；水位和阴影变化有风险'),
 'Umbra_0022':('water_weak_shore','possibly_not_evaluable','低地理重叠、图0几乎均匀，缺少明显共同结构'),
 'Umbra_0036':('field_boundaries','difficult','田界在两图可见但亮暗、方向变化；不假定田内亮点静态'),
 'Umbra_0041':('field_round_features','difficult','低重叠、田界与圆形地物在图0较清晰，需独立辨认'),
 'Umbra_0038':('field_linear_boundaries','difficult','明显亮暗/方向变化，线性边界候选需人工核验'),
 'Capella_0003':('urban_roads_buildings','ready_for_human','建筑群、道路线性结构可见，警惕屋顶散射中心移动'),
 'Capella_0004':('urban_roads_buildings','ready_for_human','建筑与道路及暗区，空间重叠中等'),
 'Umbra_0047':('shore_water','difficult','主要水面/岸线；低重叠，不是普遍城市建筑裁片'),
 'UmbraZero_0042':('shore_sparse_buildings','difficult','水域/岸线及少量建筑，两图可见范围不同'),
 'Umbra_0056':('port_water_shadow','possibly_not_evaluable','图0岸边/亮结构，图1大面积暗/散斑，可能无可靠共同静态点'),
 'Umbra_0061':('marina_piers','ready_for_human','码头固定线形结构可見，船舶动态，勿将船作为GT'),
 'Umbra_0059':('port_linear_structures','difficult','少量线状结构，但一幅很暗，静态结构需辨认'),
 'Umbra_0068':('water_sparse_streak','possibly_not_evaluable','主要散斑/水域，少量条带不能直接视作静态地物'),
 'Umbra_0071':('water_shadow_clutter','possibly_not_evaluable','散斑/暗区，未在总览中识别可靠共同静态结构'),
 'S1SF_EXTENSION_000':('urban_coast_port','ready_for_human','城市、岸线、港区可见；粗分辨率仅支持相应尺度点位'),
 'Sentinel1_0011':('farmland_lakes','ready_for_human','田块边界和水体可见，观测方向尚未知，不填造角度'),
 'Capella_0007':('water_speckle','possibly_not_evaluable','站点为城市海岸，但本裁片主要水面散斑'),
 'Capella_0005':('water_speckle','possibly_not_evaluable','站点为城市海岸，但本裁片主要水面散斑')}
DEMO=['S1SF_EXTENSION_000','Capella_0003','Umbra_0061','Umbra_0014','Umbra_0041']

def geographic_points(task):
    with rasterio.open(task['reference_native_geotiff']) as a,rasterio.open(task['moving_native_geotiff']) as b:
        x,y=np.meshgrid(np.linspace(16,a.width-17,9),np.linspace(16,a.height-17,9));p=np.c_[x.ravel(),y.ravel()]
        m=np.array([a.transform*(u+.5,v+.5) for u,v in p]);xx,yy=transform(a.crs,b.crs,m[:,0].tolist(),m[:,1].tolist())
        q=np.array([(~b.transform)*(u,v) for u,v in zip(xx,yy)])-.5
        return [{'candidate_id':'G'+str(i),'xy0':p[i].tolist(),'xy1':q[i].tolist(),'source':'provider_geocoding_only_not_GT'} for i in range(len(p)) if np.all(q[i]>=0) and q[i,0]<=b.width-1 and q[i,1]<=b.height-1]

def main():
    freeze=json.loads((ROOT/'selection/selection_freeze.json').read_text());assert freeze['sha256']==sha(ROOT/'selection/samples.jsonl')
    project=ROOT.parent/'MaritimeSARRegistrationPhase2';sys.path.insert(0,str(project));from sarbench.matchers import LocalFeatureMatcher
    selection=json.loads((project/'output/phase25_20261010_0000/development/selection.json').read_text())
    assert sha(project/'sarbench/matchers.py')==selection['inference_code_sha256']['sarbench/matchers.py']
    matcher=LocalFeatureMatcher('sift',max_features=2048,detector_options={'contrastThreshold':.01});cv2.setNumThreads(1)
    tasks=[];records=[]
    for r in [json.loads(s) for s in (ROOT/'selection/samples.jsonl').read_text().splitlines()]:
        c,ready,note=CONTENT[r['pair_id']];r.update(visual_scene=c,annotation_readiness=ready,visual_note=note,visual_review_source='assistant visual content triage only; not independent GT annotation',demo_priority=DEMO.index(r['pair_id'])+1 if r['pair_id'] in DEMO else None)
        start=time.perf_counter();candidates=[];error=None
        try:
            pred=matcher.predict(np.array(Image.open(r['image0'])),np.array(Image.open(r['image1'])))
            candidates=[{'candidate_id':'S'+str(i),'xy0':a.tolist(),'xy1':b.tolist(),'source':'frozen_SIFT_mutual_ratio08_no_manual_confirmation'} for i,(a,b) in enumerate(zip(pred.points0,pred.points1))]
        except Exception as e:error=repr(e)
        geo=geographic_points(r);path=ROOT/'candidate_matches'/f"{r['sample_id']}.json"
        write(path,{'sample_id':r['sample_id'],'GT':False,'geographic':geo,'sift_candidates':candidates,'failure':error,'scope':'assistance only; never auto-filled into annotations'})
        r.update(candidate_count=len(candidates)+len(geo),candidate_sha256=sha(path));tasks.append(r);records.append({'sample_id':r['sample_id'],'sift_candidates':len(candidates),'geographic_candidates':len(geo),'failure':error,'candidate_generation_ms':(time.perf_counter()-start)*1000})
    tasks.sort(key=lambda r:(r['demo_priority'] is None,r['demo_priority'] or 99,r['sample_id']))
    (ROOT/'annotation_tasks/tasks.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in tasks));write(ROOT/'annotation_tasks/demo5.json',[r['sample_id'] for r in tasks[:5]])
    write(ROOT/'annotation_tasks/task_freeze.json',{'task_sha256':sha(ROOT/'annotation_tasks/tasks.jsonl'),'selection_sha256':freeze['sha256'],'candidate_hashes':{r['sample_id']:r['candidate_sha256'] for r in tasks},'demo5':DEMO,'human_GT_points':0,'readiness':dict(Counter(r['annotation_readiness'] for r in tasks)),'final_human_exclusions':0})
    write(ROOT/'candidate_matches/SUMMARY.json',{'records':records,'candidate_only':True,'human_confirmed_points':0,'sift_options':{'contrastThreshold':.01,'max_features':2048,'mutual_ratio':.8},'geographic_transform':'pixel centers +0.5 -> source CRS -> target CRS -> inverse affine -0.5'})
    write(ROOT/'selection/visual_triage.json',[{k:r[k] for k in ['sample_id','visual_scene','annotation_readiness','visual_note','visual_review_source']} for r in tasks])
    print(Counter(r['annotation_readiness'] for r in tasks));print('Five first tasks',[r['sample_id'] for r in tasks[:5]])
if __name__=='__main__':main()
