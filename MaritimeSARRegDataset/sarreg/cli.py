from __future__ import annotations
import argparse
import json
from pathlib import Path
from .indexer import prepare_index
from .synthesis import generate
from .qa import validate_generated, visualize_generated
from .toy import make_toy
from .io import jsonl_read,jsonl_write


def main():
    p=argparse.ArgumentParser(prog='sarreg',description='Maritime SAR image registration proxy dataset builder (manual data acquisition)')
    sp=p.add_subparsers(dest='command',required=True)
    a=sp.add_parser('prepare',help='Index already-extracted HRSID, SSDD, or generic images')
    a.add_argument('--dataset',choices=['hrsid','ssdd','background','generic'],required=True)
    a.add_argument('--root',required=True,help='dataset extracted directory')
    a.add_argument('--out',default='data/index/index.jsonl')
    a.add_argument('--sea-mask-root',default=None,help='SL-SSDD mask root with train/ and test/ folders (for SSDD)')
    a.add_argument('--allow-partial-sea-masks',action='store_true',help='Allow SSDD rows without matching SL-SSDD masks (default: fail)')
    a.add_argument('--seed',type=int,default=42)
    a.add_argument('--val-fraction',type=float,default=.12)
    a.add_argument('--test-fraction',type=float,default=.18)
    a.add_argument('--split-policy',choices=['auto','official','group'],default='auto',
                   help='auto: HRSID/background grouped by source, SSDD official; group: parent-scene-safe; official: retain publisher split')
    b=sp.add_parser('generate',help='Offline generate image-only synthetic registration dataset')
    b.add_argument('--index',required=True)
    b.add_argument('--config',default='configs/default.yaml')
    b.add_argument('--out',default='output/synthetic')
    b.add_argument('--split',choices=['train','val','test','all'],default='train')
    b.add_argument('--count',type=int,default=100)
    b.add_argument('--seed',type=int,default=42)
    b.add_argument('--overwrite',action='store_true')
    c=sp.add_parser('validate',help='Numerically validate matrices, overlap masks, and matches')
    c.add_argument('--root',required=True)
    c.add_argument('--max-pairs',type=int,default=None)
    d=sp.add_parser('visualize',help='Export side-by-side registration and alignment previews')
    d.add_argument('--root',required=True)
    d.add_argument('--out',default='output/previews')
    d.add_argument('--count',type=int,default=8)
    d.add_argument('--seed',type=int,default=42)
    m=sp.add_parser('merge-index',help='Combine prepared dataset indexes for mixed training')
    m.add_argument('--indexes',nargs='+',required=True)
    m.add_argument('--out',required=True)
    e=sp.add_parser('demo',help='Make FAKE imagery only for smoke testing code, not for research results')
    e.add_argument('--out',default='data/demo')
    e.add_argument('--count',type=int,default=24)
    e.add_argument('--seed',type=int,default=17)
    args=p.parse_args()
    if args.command=='prepare':
        result=prepare_index(args.dataset,args.root,args.out,args.sea_mask_root,args.seed,args.val_fraction,args.test_fraction,args.split_policy,args.allow_partial_sea_masks)
    elif args.command=='generate':
        result=generate(args.index,args.config,args.out,args.split,args.count,args.seed,args.overwrite)
    elif args.command=='validate':
        result=validate_generated(args.root,args.max_pairs)
    elif args.command=='visualize':
        result={'previews':visualize_generated(args.root,args.out,args.count,args.seed)}
    elif args.command=='merge-index':
        rows=[item for path in args.indexes for item in jsonl_read(path)]
        keys=set(); merged=[]
        for item in rows:
            key=str(item['image_path'])
            if key not in keys:
                keys.add(key);merged.append(item)
        jsonl_write(args.out,merged)
        result={'merged':len(merged),'output':args.out}
    else:
        result=make_toy(args.out,args.count,args.seed)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if args.command=='generate' and result['generated']<args.count:
        raise SystemExit(2)
    if args.command=='validate' and result['failed']:
        raise SystemExit(3)

if __name__=='__main__':main()
