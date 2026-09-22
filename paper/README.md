# paper/

Frozen artifacts from the 31 August 2026 paper freeze.

## Scoring references — these are the regression oracle

`ref_score_bench_stdout.txt` and `ref_score_classical_stdout.txt` are the exact
stdout of the two scorers at freeze time. Any refactor must still reproduce
them byte for byte:

```bash
/opt/anaconda3/envs/clarius/bin/python src/audit/score_bench.py     | diff paper/ref_score_bench_stdout.txt -
/opt/anaconda3/envs/clarius/bin/python src/audit/score_classical.py | diff paper/ref_score_classical_stdout.txt -
```

**The environment is part of the result.** These reproduce in `clarius`
(Python 3.10, ultralytics 8.4.115). Filter out the three "Ultralytics settings
updated" warning lines before diffing, and keep blank lines: the classical
reference has four.

Three of the four checkpoints `score_bench` needs are untracked, so this table
cannot be regenerated from a fresh clone. See `models/README.md`.

## Figure provenance

Only two of the fourteen figures have a generator in this repo.

| figure | generator |
|---|---|
| `e5_matched_region.png` | `src/viz/make_e5_figs.py` |
| `e5_flight143_timeline.png` | `src/viz/make_e5_figs.py` |
| `blindspot_after.png` | no generator: a hand-renamed copy of a `src/viz/diag_left.py` output — byte-identical to outputs/figures/diag_left_section_106.png |
| `kappa_90.png`, `kappa_92.png`, `kappa_94.png` | no generator: frames grabbed by hand from the `src/viz/viz_kappa_video.py` videos, which write .mp4, not .png |
| `timeline_133/134/136/137/138/139/141/143.png` | **no generator anywhere in the repo** |

### The E5 figures no longer reproduce, and did not before this refactor

Regenerating `e5_*.png` today gives different files from the committed ones
(45195 / 67215 bytes against the committed 40497 / 60929). The cause predates
any refactoring: the gauge replay logs for flights 133, 134, 141 and 143 were
regenerated on 3 September, after the 31 August freeze, so the generator now
reads different inputs.

The generator is self-consistent: two runs against the same logs agree byte for
byte, which is how the refactor was checked. But the committed PNGs correspond
to logs that no longer exist. Use `--out` to regenerate somewhere harmless:

```bash
python src/viz/make_e5_figs.py --out /tmp/figs
```
