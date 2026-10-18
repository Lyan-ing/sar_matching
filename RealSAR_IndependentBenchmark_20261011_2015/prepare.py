"""Copy five frozen tasks and UI; inspect displays without creating correspondences."""
import sys
sys.dont_write_bytecode = True
import json, hashlib, shutil, importlib.metadata
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'RealSAR_RepresentationContext_20261011_1903'
MANUAL = ROOT.parent / 'RealSAR_GeometryGT_20261011_0220'
AUTO = ROOT.parent / 'RealSAR_AutoLabels_20261011_1149'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, v):
    with p.open('x') as f: json.dump(v, f, ensure_ascii=False, indent=2); f.write('\n')

def main():
    adapter = ROOT / 'annotation_adapter'
    adapter.mkdir(exist_ok=False)
    for folder in ['annotation_tasks', 'candidate_matches']:
        shutil.copytree(OLD / 'annotation_adapter' / folder, adapter / folder)
    for f in ['index.html','app.js','coordinates.js']:
        shutil.copyfile(OLD / 'annotation_adapter' / f, adapter / f)
    for folder in ['annotations','gt_quality']: (adapter / folder).mkdir()
    tasks = [json.loads(x) for x in (adapter/'annotation_tasks/tasks.jsonl').read_text().splitlines()]
    assert len(tasks)==5
    priority = ['Capella_0003__chip_001','Umbra_0014__chip_002','Umbra_0061__chip_001','S1SF_EXTENSION_000__chip_002']
    regions = {'Capella_0003__chip_001': [[96,96],[224,192],[112,352]],
               'Umbra_0014__chip_002': [[64,64],[208,208],[352,352]],
               'Umbra_0061__chip_001': [[32,96],[192,208],[352,352]],
               'S1SF_EXTENSION_000__chip_002': [[400,224],[400,320],[400,32]],
               'Umbra_0071__chip_000': [[48,48],[208,208],[352,352]]}
    receipts = [OLD/'ENGINEERING_ACCEPTANCE.json', AUTO/'ENGINEERING_ACCEPTANCE.json']
    inputs = [p for p in receipts if p.exists()]
    inputs += [OLD/'annotation_adapter'/f for f in ['app.js','coordinates.js','index.html','annotation_tasks/tasks.jsonl','annotation_tasks/task_freeze.json']]
    inputs += [MANUAL/'annotation_app'/f for f in ['server.py','store.py']]
    inputs += [MANUAL/'gt_quality/quality.py',AUTO/'annotation_adapter/serve.py',AUTO/'common.py']
    displays = []
    for t in tasks:
        inputs.append(OLD/'annotation_adapter/candidate_matches'/f"{t['sample_id']}.json")
        tiles = Image.new('RGB',(864,652),'#18202a');d=ImageDraw.Draw(tiles)
        d.text((4,3),t['sample_id']+' | independent VIEWPORTS, NOT correspondences',fill='white')
        stats=[]
        for side in [0,1]:
            p=Path(t['image'+str(side)]);inputs.append(p)
            for field in ['reference_native_geotiff','moving_native_geotiff','provider_metadata_reference','provider_metadata_moving']:
                if t.get(field):inputs.append(Path(t[field]))
            im=Image.open(p);assert im.size==(512,512) and im.mode=='L'
            stats.append(dict(side=side,shape=[512,512],mode=im.mode,sha256=sha(p),range=im.getextrema()))
            for j,(x,y) in enumerate(regions[t['sample_id']]):
                patch=im.crop((x,y,x+96,y+96)).resize((288,288),Image.Resampling.NEAREST)
                tiles.paste(patch.convert('RGB'),(j*288,38+side*310))
                d.text((j*288+3,20+side*310),f'view {side} ROI ({x},{y}) 96x96; 3x display',fill='white')
        out=ROOT/'previews'/f"{t['sample_id']}_local_inspection.png"
        assert not out.exists();tiles.save(out)
        displays.append(dict(sample_id=t['sample_id'],priority=t['sample_id'] in priority,
                             look_pair_category=t['look_pair_category'],views=stats,
                             independent_viewports_xywh=[[*p,96,96] for p in regions[t['sample_id']]],
                             ROIs_are_correspondences=False,static_landmarks_verified_by_human=False))
    save(ROOT/'display_checks.json',displays)
    save(ROOT/'priority.json',dict(priority_order=priority,retained_extra='Umbra_0071__chip_000',
         task_manifest_unchanged=True,new_human_points=0,new_C=0,
         no_new_unseen_confirmation_set=True,all_previously_examined=True))
    save(ROOT/'input_sha256.json',{str(p.resolve()):sha(p) for p in sorted(set(inputs))})
    save(ROOT/'environment.json',dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),python=sys.version,
          executable=sys.executable,packages={p:importlib.metadata.version(p) for p in ['numpy','Pillow','opencv-python-headless','torch','rasterio','pytest']},
          dependencies_installed_this_round=[]))
    print(json.dumps(dict(tasks=5,native_images_checked=10,priority_pairs=4,points_created=0,C_created=0,protected_inputs=len(set(inputs))),indent=2))
if __name__=='__main__':main()
