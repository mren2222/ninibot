# Fine-align preparation trajectory

This public directory contains one compact **processed** teach/replay example.
The raw operator recording, per-sample CSV, previews, and other sessions remain
in the private working repository.

```text
sessions/
  2026-06-15_8cm_base_raw/
    processed_template.json
    processing_metrics.json
    processed_joint_dynamics.png
    raw_vs_processed_comparison.png
    README.md
```

The default Ninibot parameters point at this processed template.

To process a new local recording, first collect it into a new session directory
and then run:

```bash
ros2 run ninibot process_prep_trajectory \
  data/trajectories/fine_align_prep/sessions/YOUR_SESSION/raw_recording.csv \
  --trim-start-sec 12.0 \
  --trim-end-sec 1.0 \
  --path-smoothing spline \
  --joint4-mode horizontal_offset \
  --spline-smoothing 0.0003 \
  --flat-weight 20.0 \
  --no-latest
```

Generated recordings are machine/workcell-specific and are ignored by the
public repository unless deliberately curated.

