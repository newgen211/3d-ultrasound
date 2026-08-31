import json
truth = json.load(open('audit/audit_truth.json'))
SECS = ['section_62','section_81','section_85','section_88','section_90',
        'section_92','section_94','section_103','section_104','section_112','section_113']
hdr = "section       audit d/f  v5f d/f   ratio   hits(truth>0 & v5f>0)"
print(hdr)
for sec in SECS:
    try:
        j = json.load(open('data/clarius_sessions/%s/sam_detections_v5f.json' % sec))
        v5f = j['detections']; nfr = j.get('n_frames', 0)
    except Exception:
        print("%-13s  (no v5f)" % sec); continue
    byframe = {}
    for d in v5f:
        byframe.setdefault(d['frame_index'], []).append(d)
    keys = [k for k in truth if k.startswith(sec + '__')]
    v_dens = len(v5f) / nfr if nfr else float('nan')
    if not keys:
        print("%-13s %9s %8.2f %7s   (not audited)" % (sec, '-', v_dens, '-')); continue
    tfr = [(int(k.split('__')[1].split('.')[0]), truth[k]) for k in keys]
    usable = [(fi, e) for fi, e in tfr if not e.get('unusable')]
    t_dens = sum(len(e['vessels']) for _, e in usable) / len(usable) if usable else float('nan')
    pos = [(fi, e) for fi, e in usable if e['vessels']]
    hit = sum(1 for fi, _ in pos if byframe.get(fi))
    ratio = v_dens / t_dens if t_dens else float('nan')
    print("%-13s %9.2f %8.2f %6.1f%%   %d/%d" % (sec, t_dens, v_dens, ratio * 100, hit, len(pos)))
