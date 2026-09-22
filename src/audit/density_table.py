#!/usr/bin/env python3
"""Density table vs OPEN-morph audit truth only (fair to the by-design crescent skips)."""
import json, glob
import numpy as np, cv2
from pathlib import Path

truth = json.load(open('audit/audit_truth.json'))
SECS = ['section_62','section_81','section_85','section_90','section_92',
        'section_94','section_103','section_104']

def classify(gray, x, y, r):
    """open if interior < 0.75 * median(gray) — same rule as score_models.py"""
    H, W = gray.shape
    yy, xx = np.ogrid[:H, :W]
    m = (xx - x)**2 + (yy - y)**2 <= (r * 0.7)**2   # interior = inner 70%
    if not m.any(): return 'crescent'
    return 'open' if gray[m].mean() < 0.75 * np.median(gray) else 'crescent'

print("section       open d/f  all d/f  v5f d/f  ratio_open   hits_open")
for sec in SECS:
    jpgs = sorted(glob.glob(f'data/clarius_sessions/{sec}/frames_jpg/*.jpg'))
    j = json.load(open(f'data/clarius_sessions/{sec}/sam_detections_v5f.json'))
    v5f = j['detections']; nfr = j.get('n_frames', len(jpgs))
    byframe = {}
    for d in v5f: byframe.setdefault(d['frame_index'], []).append(d)
    keys = [k for k in truth if k.startswith(sec + '__')]
    n_open = n_all = n_fr = hit = pos = 0
    for k in sorted(keys):
        e = truth[k]
        if e.get('unusable'): continue
        fi = int(k.split('__')[1].split('.')[0])
        # find the jpg for this frame
        jp = f'data/clarius_sessions/{sec}/frames_jpg/{fi:05d}.jpg'
        img = cv2.imread(jp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f'  !! {sec} f{fi}: no jpg at {jp}'); continue
        n_fr += 1
        opens = 0
        for (x, y, r) in e['vessels']:
            n_all += 1
            if classify(img, x, y, r) == 'open':
                opens += 1
        n_open += opens
        if opens > 0:
            pos += 1
            if byframe.get(fi): hit += 1
    v_dens = len(v5f) / nfr if nfr else float('nan')
    od = n_open / n_fr if n_fr else float('nan')
    ad = n_all / n_fr if n_fr else float('nan')
    ratio = v_dens / od if od else float('nan')
    print("%-13s %8.2f %8.2f %8.2f %10.1f%%   %d/%d" % (sec, od, ad, v_dens, ratio*100, hit, pos))
