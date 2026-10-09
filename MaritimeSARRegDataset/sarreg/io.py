from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np

EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}

def read_gray(path: str | Path) -> np.ndarray:
    path = Path(path)
    arr = np.fromfile(str(path), dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError(f"Cannot decode image: {path}")
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY if im.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    if im.dtype != np.uint8:
        arrf = im.astype(np.float32)
        lo, hi = np.percentile(arrf, [0.3, 99.7])
        im = np.clip((arrf-lo)*255/max(hi-lo,1e-6),0,255).astype('uint8')
    return im

def write_png(path: str | Path, im: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode('.png', im)
    if not ok:
        raise IOError(f"Cannot encode image {path}")
    buf.tofile(str(path))

def load_json(path):
    with Path(path).open('r', encoding='utf8') as f:
        return json.load(f)

def save_json(path, obj):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf8') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False)

def jsonl_read(path):
    with Path(path).open('r',encoding='utf8') as f:
        return [json.loads(x) for x in f if x.strip()]

def jsonl_write(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',encoding='utf8') as f:
        for x in rows:
            f.write(json.dumps(x,ensure_ascii=False)+'\n')

def image_paths(root):
    return sorted(p for p in Path(root).rglob('*') if p.is_file() and p.suffix.lower() in EXTS)
