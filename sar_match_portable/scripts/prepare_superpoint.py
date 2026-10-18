#!/usr/bin/env python3
"""Prepare the fixed external SuperPoint source locally, only after license acceptance."""
import argparse
import hashlib
import os
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
COMMIT='eb42fee2d71449efb0aa5c10549752b5d75384d8'
URL=f'https://raw.githubusercontent.com/cvg/LightGlue/{COMMIT}/lightglue/superpoint.py'


def prepare(data, expected):
    digest=hashlib.sha256(data).hexdigest()
    if digest!=expected:raise ValueError(f'Upstream source SHA256 mismatch: {digest}')
    source=data.decode('utf-8')
    old='        self.load_state_dict(torch.hub.load_state_dict_from_url(url))'
    if source.count(old)!=1:raise ValueError('Unexpected upstream constructor; refusing to patch.')
    return source.replace(old,'        # Local-only preparation: strict SHA-verified model loading is handled by the wrapper.\n        # No constructor download or fallback weights.').encode('utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--accept-superpoint-license',action='store_true',help='You have read and can comply with the upstream SuperPoint research license.')
    p.add_argument('--source-file',type=Path,help='Offline copy of the exact upstream file, legally obtained by the user.')
    a=p.parse_args()
    if not a.accept_superpoint_license:p.error('Read licenses/SuperPoint_RESEARCH_LICENSE.txt first. Explicit acceptance is required.')
    data=a.source_file.read_bytes() if a.source_file else urllib.request.urlopen(URL,timeout=30).read()
    # Exact original source identity, not the private experiment copy.
    source=prepare(data,UPSTREAM_SHA256)
    target=ROOT/'sar_match/_vendor/superpoint.py'
    if target.exists() and target.read_bytes()!=source:raise FileExistsError('Existing local SuperPoint differs; it was not overwritten.')
    if not target.exists():
        tmp=target.with_suffix('.py.partial');tmp.write_bytes(source);os.replace(tmp,target)
    print('Prepared local-only restricted source:',target)
    print('No model weights downloaded. Place authorized final/LightGlue weights as documented.')


UPSTREAM_SHA256='23e8a26137b44f37e9a319313822c5b38d9c55796e756202eb46d64f1979d2b0'
if __name__=='__main__':main()
