# models/

Detector checkpoints. Most are **not tracked** (`.gitignore` covers `models/`
and `*.pt`); three predate that rule and are grandfathered in.

This matters for reproducibility: `paper/ref_score_bench_stdout.txt` scores
four checkpoints, and three of them are untracked. A fresh clone cannot
regenerate that table. If those numbers need to survive, the weights have to
be archived somewhere outside git.

| file | size | tracked | sha256 (first 16) | used by |
|---|---|---|---|---|
| `bench_m.pt` | 44.0 MB | no | `cc89c2709bc31b9a` | src/audit/score_bench.py |
| `bench_s.pt` | 20.3 MB | no | `632d1310a666d340` | src/audit/score_bench.py |
| `best.pt` | 5.4 MB | yes | `6c60d873ac87cfb2` | -- superseded by best_regated.pt |
| `best_regated.pt` | 5.4 MB | yes | `9551809e5e810cd9` | src/audit/score_bench.py, src/audit/score_classical.py, src/supervise/supervisor_v3.py (default), src/viz/diag_lock.py |
| `gold_n.pt` | 5.4 MB | no | `70c03602066e3779` | src/supervise/gauge.py (default), src/viz/diag_left.py (default) |
| `modelA_best.pt` | 5.4 MB | no | `ea1f47cfe63c6379` | -- Model A line, superseded |
| `modelA_v5f.pt` | 5.4 MB | no | `8ebdbd1c61d6ad6c` | -- Model A line, superseded |
| `modelA_v5f_last.pt` | 5.4 MB | no | `cf5d382d21c309ef` | -- Model A line, superseded |
| `modelA_v5fp.pt` | 5.4 MB | no | `3943fae25346da30` | -- Model A line, superseded |
| `yolo26n.pt` | 5.5 MB | yes | `9b09cc8bf347f0fc` | -- pretrained starting point for training |

Full digests:

```
cc89c2709bc31b9a5c44a7b9d19cdeea5101fb5cb3f31d1cf4a2eb4c396e3993  bench_m.pt
632d1310a666d34022848d1f9d1242093ea117e65c1635b79ec8da515db1e6e1  bench_s.pt
6c60d873ac87cfb2212d2e6130c0c586294a026217a4246e5b735321893ee313  best.pt
9551809e5e810cd9f788d00e81d145106585a82b64caafddb004d18b78ee1a09  best_regated.pt
70c03602066e37794aee7e3a32c3ed603df9cfd614d96bcaa7f9b45e0711a2d2  gold_n.pt
ea1f47cfe63c6379cc7771009dfbd422461f19b077a9640cfe70ee87fe0137cf  modelA_best.pt
8ebdbd1c61d6ad6c1b46ac01cedf87a069d9e60126a8c14b9397781952ef8256  modelA_v5f.pt
cf5d382d21c309ef690b91dad33aa9fcec325678282666b13e1c29c5b452fd14  modelA_v5f_last.pt
3943fae25346da30174b3ca8415cd1604abd69d826a80befdc47c907fb18d030  modelA_v5fp.pt
9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef  yolo26n.pt
```
