# One-button sweep. `make sweep` flies; `make post SEC=N` runs the file-keyed
# post chain (each stage only if its inputs are newer than its artifact).
SHELL := /bin/bash
.SHELLFLAGS := -eo pipefail -c

PY_RS   ?= /opt/anaconda3/envs/realsense/bin/python   # anchor, solve, tracker, trim, smooth, merge
PY_CL   ?= /opt/anaconda3/envs/clarius/bin/python     # reconstruct, vessels (matplotlib, numpy 1.x)
HANDEYE ?= calib/handeye.json
LOGS    := data/pose_logs
SEC     ?= $(shell ls data/clarius_sessions 2>/dev/null | sed -n 's/^section_\([0-9]*\)$$/\1/p' | sort -n | tail -1 | awk '{print $$1+1}')
SESS    := data/clarius_sessions/section_$(SEC)

.PHONY: sweep post trim smooth merge reconstruct vessels clean-stamps pi-sync

sweep:
	$(PY_RS) fly.py $(SEC)

post: vessels

# trim and merge leave no artifact of their own, so they keep stamp files.
trim: $(SESS)/.trimmed
$(SESS)/.trimmed: $(SESS)/exec.jsonl $(SESS)/exec_meta.json $(LOGS)/sec$(SEC)_cam.jsonl
	$(PY_RS) src/pose/trim_section.py section_$(SEC) $(SESS)/exec.jsonl $(SESS)/exec_meta.json $(LOGS)/sec$(SEC)_cam.jsonl | tee $(SESS)/trim.log
	$(PY_RS) src/pose/trim_section.py section_$(SEC) $(SESS)/exec.jsonl $(SESS)/exec_meta.json $(LOGS)/sec$(SEC)_cam.jsonl --apply
	touch $@

smooth: $(LOGS)/sec$(SEC)_cam_smooth.jsonl
$(LOGS)/sec$(SEC)_cam_smooth.jsonl: $(LOGS)/sec$(SEC)_cam.jsonl
	$(PY_RS) src/pose/smooth_cam_log.py $< $@

merge: $(SESS)/.merged
$(SESS)/.merged: $(SESS)/.trimmed $(LOGS)/sec$(SEC)_cam_smooth.jsonl
	$(PY_RS) src/pose/merge_poses_cam.py section_$(SEC) $(LOGS)/sec$(SEC)_cam_smooth.jsonl
	touch $@

reconstruct: $(SESS)/volume_handeye.nii.gz
$(SESS)/volume_handeye.nii.gz: $(SESS)/.merged $(HANDEYE)
	$(PY_CL) src/reconstruct/reconstruct_handeye.py section_$(SEC)

vessels: $(SESS)/recovered_tubes_3d.png
$(SESS)/recovered_tubes_3d.png: $(SESS)/volume_handeye.nii.gz
	$(PY_CL) src/reconstruct/view_planning_3d.py section_$(SEC) --handeye $(HANDEYE) --eps 3.5 --n-targets 2 --split-lateral --present --no-show | tee $(SESS)/vessels.log
	MPLBACKEND=Agg $(PY_CL) src/reconstruct/render_tubes_3d.py section_$(SEC) --handeye $(HANDEYE)

clean-stamps:
	rm -f $(SESS)/.trimmed $(SESS)/.merged

# The Pi runs copies of the capture core out of the ultrasound-cobot repo, so
# they deploy by git. src/capture/ is the source; this copies and shows the diff.
pi-sync:
	cp src/capture/cast_capture.py src/capture/cast_headless.py ultrasound-cobot/
	cd ultrasound-cobot && git status --short cast_capture.py cast_headless.py
