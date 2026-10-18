"""Isolated dataset imports: never import this module in pure-image inference."""
from runtime import *
import importlib.util
DATA=BASE/'SyntheticSAR_Mixed640512_Multicondition_20261016_2114'
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=sys.modules.get('common');dc=module('mixed_dataset_common',DATA/'code/common.py');sys.modules['common']=dc
reader=module('mixed_dataset_reader',DATA/'code/reader.py');scorer=module('mixed_dataset_scorer',DATA/'code/score.py')
if old is None:sys.modules.pop('common',None)
else:sys.modules['common']=old
ENTRY=DATA/'DATASET_ENTRYPOINTS_MIXED_20261016_2155.json';ENTRY_DATA=json.loads(ENTRY.read_text())
MANIFEST={k:DATA/f'manifests/{k}_CANONICAL_20261016_2147.jsonl' for k in ['train','val']}
def records(split):return [json.loads(s) for s in MANIFEST[split].read_text().splitlines() if s]
def images(row):
 a=[cv2.imread(row[k],cv2.IMREAD_GRAYSCALE) for k in ['image_A','image_B']];assert all(im is not None and im.shape==(row['height'],row['width']) for im in a);return a
@torch.no_grad()
def shared_cache(row,model,ims=None):
 path=ROOT/'cache'/row['split']/f"{row['pair_id']}_{TAG}.pt"
 oldpath=BASE/'SAR_SP_JointTransfer_Mixed640512_20261016_2234/cache'/row['split']/f"{row['pair_id']}_20261016_2234.pt"
 if oldpath.exists():path=oldpath
 if path.exists():
  q=torch.load(path,map_location='cpu',weights_only=False);assert q['image_hash']==row['image_file_hashes'];return q['x'].cuda(),q['responses'].cuda(),0.
 ims=images(row) if ims is None else ims;t=torch.as_tensor(np.stack(ims),device='cuda',dtype=torch.float32)[:,None]/255
 torch.cuda.synchronize();start=time.perf_counter();x=model.shared(t);rs=F.avg_pool2d(model.structure.responses(t),8,8);torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 torch.save(dict(x=x.cpu(),responses=rs.cpu(),image_hash=row['image_file_hashes'],fixed_encoder_sha='official_sp_verified',input_shape=list(t.shape)),tmp);os.replace(tmp,path)
 return x,rs,ms
