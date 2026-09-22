#!/usr/bin/env python3
"""Detection density per section: v5f detections per frame vs audited truth.

Counts OPEN-morph truth only by default, which is the fair comparison because
the crescent (collapsed) lumens are skipped by design. --all-morphs restores
the older behaviour of counting every audited vessel.
"""
import argparse
import json, glob
import sys
import numpy as np, cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from us3d.paths import AUDIT_TRUTH, SESSIONS

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("sections", nargs="*", help="sections to survey (default: the audited eight)")
ap.add_argument("--all-morphs", action="store_true",
                help="count every audited vessel, not just the open ones")
args = ap.parse_args()

truth = json.loads(AUDIT_TRUTH.read_text())
SECS = args.sections or ['section_62','section_81','section_85','section_90','section_92',
        'section_94','section_103','section_104']

def classify(gray, x, y, r):
    """open if interior < 0.75 * median(gray) — same rule as score_models.py"""
    H, W = gray.shape
    yy, xx = np.ogrid[:H, :W]
    m = (xx - x)**2 + (yy - y)**2 <= (r * 0.7)**2   # interior = inner 70%
    if not m.any(): return 'crescent'
    if args.all_morphs: return 'open'
    return 'open' if gray[m].mean() < 0.75 * np.median(gray) else 'crescent'

print("section       open d/f  all d/f  v5f d/f  ratio_open   hits_open")
for sec in SECS:
    jpgs = sorted(glob.glob(f'{SESSIONS}/{sec}/frames_jpg/*.jpg'))
    j = json.load(open(f'{SESSIONS}/{sec}/sam_detections_v5f.json'))
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
        jp = f'{SESSIONS}/{sec}/frames_jpg/{fi:05d}.jpg'
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
