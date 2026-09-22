# docs/ — Project Reference

The single source of truth for the Robotic 3D Ultrasound project. Start with the
root [`../README.md`](../README.md) for the code/data map and how to run things;
come here for the *why*, the plan, and the status.

| Document | What it is |
| --- | --- |
| [phase-1.md](phase-1.md) | The immediate near-term focus: arm scans a phantom, end to end. Read this first if you just want to know "what now." |
| [master-plan.md](master-plan.md) | The authoritative plan: thesis, architecture, current status (PART B), the full D1–D10 work plan, pitfalls, risk register, timeframes, and critical path. Includes the 2026-06-12 changelog (open-loop pivot). |
| [project-reference.md](project-reference.md) | The original phased reference: hardware/software stack, pipeline architecture, calibration & validation tables, standing questions, references, glossary. |

The master plan ends with an **August–September 2026 addendum** covering the
E-series experiments (E1 repeatability, E5 contact supervision), which is the
work the current code is about. Read it before trusting the PART B status
table above it.

**Where they conflict on architecture, the master-plan's 2026-06-12 changelog is
authoritative** (the project moved from real-time servoing to open-loop survey →
detect → plan → execute). **On status, the E-series addendum at the end of the
master plan is the current word.**
