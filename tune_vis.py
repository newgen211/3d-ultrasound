#!/usr/bin/env python3
"""What does the patch detector see on audit-truth OPEN frames vs COLLAPSED frames?"""
import json, glob
import numpy as np
from pathlib import Path
truth = json.load(open('audit/audit_truth.json'))
def signals(img):
    H, W = img.shape
    band = img[int(0.15*H):int(0.60*H), :]
    ph, pw = max(8, band.shape[0]//12), max(8, band.shape[1]//12)
    g = band[:(band.shape[0]//ph)*ph, :(band.shape[1]//pw)*pw]
    g = g.reshape(g.shape[0]//ph, ph, g.shape[1]//pw, pw)
    means, vars_ = g.mean(axis=(1,3)), g.var(axis=(1,3))
    med = np.median(img)
    return means, vars_, med
for sec in ['section_85','section_62']:
    p = Path(f'data/clarius_sessions/{sec}')
    rep_p = p/'compression_report.json'
    report = json.loads(rep_p.read_text())['frames'] if rep_p.exists() else None
    raws = sorted(glob.glob(str(p/'raw_*.json')))
    keys = sorted(k for k in truth if k.startswith(sec+'__'))
    print(f'== {sec}')
    for k in keys[:8]:
        fi = int(k.split('__')[1].split('.')[0])
        if fi >= len(raws): continue
        meta = json.load(open(raws[fi]))['frame']
        img = np.fromfile(raws[fi].replace('.json','.bin'),dtype=np.uint8
                          ).reshape(meta['samples'],meta['lines'])
        means, vars_, med = signals(img)
        nves = len(truth[k]['vessels'])
        st = report[fi]['state'] if (report and fi<len(report)) else '?'
        dark55 = (means < 0.55*med).sum(); dark75 = (means < 0.75*med).sum()
        smooth = (vars_ < np.percentile(vars_,25))
        both55 = ((means<0.55*med)&smooth).sum(); both75=((means<0.75*med)&smooth).sum()
        print(f'  f{fi:4d} truth={nves} state={st:<9} med={med:5.1f} '
              f'dark@.55={dark55:2d} dark@.75={dark75:2d} dark&smooth@.55={both55:2d} @.75={both75:2d}')
