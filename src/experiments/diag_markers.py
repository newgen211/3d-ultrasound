#!/usr/bin/env python3
import json, glob, itertools
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from us3d.paths import POSE_LOGS, SESSIONS
ROOT = SESSIONS; CAM = POSE_LOGS / "probe_pose_log_smooth.jsonl"
def span(sec):
    ts=[int(json.load(open(f)).get("host_timestamp_ns") or 0) for f in glob.glob(str(ROOT/sec/"*.json"))]
    ts=[t for t in ts if t]; return min(ts), max(ts)
A_lo=min(span(s)[0] for s in["section_106","section_108","section_109"])
A_hi=max(span(s)[1] for s in["section_106","section_108","section_109"])
B_lo,B_hi=span("section_111")
bk={}
for line in open(CAM):
    try:r=json.loads(line)
    except:continue
    i,t=r.get("id"),r.get("t_ns")
    if t is None:continue
    e="A" if A_lo<=t<=A_hi else("B" if B_lo<=t<=B_hi else None)
    if e:bk.setdefault((e,i),[]).append(r["coords"][:3])
ids=sorted({i for(e,i)in bk if (("A",i)in bk and("B",i)in bk and len(bk[("A",i)])>20 and len(bk[("B",i)])>20)})
print("usable ids:",ids)
med={(e,i):np.median(np.asarray(bk[(e,i)]),0) for e in"AB" for i in ids}
def kabsch(A,B):
    ca,cb=A.mean(0),B.mean(0);U,S,Vt=np.linalg.svd((B-cb).T@(A-ca))
    d=np.sign(np.linalg.det(Vt.T@U.T));R=Vt.T@np.diag([1]*(A.shape[1]-1)+[d])@U.T
    t=ca-R@cb;res=np.linalg.norm((R@B.T).T+t-A,axis=1)
    return R,t,res
for combo in itertools.combinations(ids,3) if len(ids)>3 else [tuple(ids)]:
    A=np.array([med[("A",i)] for i in combo]);B=np.array([med[("B",i)] for i in combo])
    R,t,res=kabsch(A,B)
    ang=np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
    print(f"ids {combo}: rot {ang:5.2f}deg  resid {np.round(res,2)}  RMS {np.sqrt((res**2).mean()):.2f}")
if len(ids)>3:
    A=np.array([med[("A",i)] for i in ids]);B=np.array([med[("B",i)] for i in ids])
    R,t,res=kabsch(A,B)
    print(f"all {ids}: resid {np.round(res,2)}  RMS {np.sqrt((res**2).mean()):.2f}")
