#!/usr/bin/env python3
"""Classical-detector duty cycle: how often the vessel is visible at all.

Prints, per section, the fraction of frames with at least one candidate, the
same split into deciles so a dropout shows up as a run of low numbers, and the
longest consecutive miss.

    python3 src/audit/census_duty.py section_104 section_113
    python3 src/audit/census_duty.py 106 108 109 111
    python3 src/audit/census_duty.py            # the hand-held set

Absorbs the former census_duty / census_duty2 / census_hand trio, which were
one script with three hard-coded section lists.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from us3d.frames import load_frame
from us3d.sections import find_section, frame_scale, raw_pairs, read_meta
from us3d.tube import candidates

DEFAULT_SECTIONS = ["section_104", "section_113", "section_115", "section_118"]


def duty(section):
    vis = []
    for jp, bp in raw_pairs(section):
        meta = read_meta(jp)
        axial, lateral = frame_scale(meta)
        img = load_frame(bp, meta)
        vis.append(1 if candidates(img, axial, lateral) else 0)
    return vis


def main(argv):
    for arg in argv or DEFAULT_SECTIONS:
        section = find_section(arg)
        vis = duty(section)
        n = len(vis)
        if not n:
            print("%s: no frames" % section.name)
            continue
        d = max(n // 10, 1)
        deciles = " ".join(
            "%3.0f" % (100 * sum(vis[i * d:(i + 1) * d]) / d) for i in range(10)
        )
        gap = longest = 0
        for v in vis:
            gap = 0 if v else gap + 1
            longest = max(longest, gap)
        print("%s: %d/%d = %.0f%%  deciles [%s]  longest_miss=%d"
              % (section.name, sum(vis), n, 100 * sum(vis) / n, deciles, longest))


if __name__ == "__main__":
    main(sys.argv[1:])
