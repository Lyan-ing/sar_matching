"""Output-only native-pixel overlays; no point modification or acceptance logic."""
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


def draw_matches(a, b, arrays, record, path, title='', max_lines=180):
    h, w = a.shape; header = 108
    canvas = Image.new('RGB', (2*w, h+header+34), '#f5f8fa')
    canvas.paste(Image.fromarray(a).convert('RGB'), (0, header))
    canvas.paste(Image.fromarray(b).convert('RGB'), (w, header))
    draw = ImageDraw.Draw(canvas)
    try: font = ImageFont.truetype('DejaVuSans.ttf', 13)
    except OSError: font = ImageFont.load_default()
    lines = [title, f"A: {record.get('image_A','reference')}  |  B: {record.get('image_B','moving')}",
             f"keypoints {record['n_keypoints_A']}/{record['n_keypoints_B']}  matches {record['n_matches']}  inliers {record['n_inliers']}  accepted={record['accepted']}",
             f"{record['reason']} | H: A -> B | No independent correspondence GT"]
    for i, line in enumerate(lines):
        # Fit names, not image coordinates. Full names are retained in CSV/JSON.
        if len(line)>2*w//7: line=line[:2*w//7-3]+'...'
        draw.text((10, 5+i*25), line, fill='#163745', font=font)
    for offset, key in [(0, 'keypoints_A'), (w, 'keypoints_B')]:
        for x,y in arrays[key]:
            draw.ellipse((float(x+offset-1), float(y+header-1), float(x+offset+1), float(y+header+1)), fill='#709cb7')
    n=len(arrays['matches_A']); mask=arrays['inliers']
    # Show both accepted and rejected candidates, with deterministic bounded clutter.
    chosen=[]
    for wanted in (False,True):
        ids=np.flatnonzero(mask==wanted)
        if len(ids)>max_lines//2: ids=ids[np.linspace(0,len(ids)-1,max_lines//2,dtype=int)]
        chosen.extend(ids.tolist())
    for i in chosen:
        x,y=map(float,arrays['matches_A'][i]); u,v=map(float,arrays['matches_B'][i]); u+=w; y+=header; v+=header
        color='#31c7ad' if mask[i] else '#e6964e'
        draw.line((x,y,u,v),fill=color,width=1)
        for px,py in [(x,y),(u,v)]: draw.ellipse((px-2,py-2,px+2,py+2),outline=color,width=1)
    for offset,key in [(0,'matches_A'),(w,'matches_B')]:
        pts=arrays[key][mask]
        if len(pts)>=3:
            hull=cv2.convexHull(pts.astype(np.float32)).reshape(-1,2)
            vertices=[(float(x+offset),float(y+header)) for x,y in hull]
            draw.line(vertices+[vertices[0]],fill='#74d7c4',width=1)
    draw.text((10,h+header+8),f"Teal: RANSAC support, not proven correct. Orange: other candidates. Lines shown {len(chosen)}/{n}.",fill='#163745',font=font)
    canvas.save(path)
