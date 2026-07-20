# Processed 8 cm-base trajectory example

This is a curated processed example from a fine-align preparation teaching
session. It is included to demonstrate the smoothing, retargeting, and
diagnostic pipeline—not as a generally safe trajectory for another robot.

Included files:

- `processed_template.json` — retargetable processed trajectory used by the
  current project defaults.
- `processing_metrics.json` — processing settings and velocity, acceleration,
  jerk, and endpoint-adjustment diagnostics.
- `raw_vs_processed_comparison.png` — shape comparison from the private raw
  recording to the processed result.
- `processed_joint_dynamics.png` — detailed processed position, velocity,
  acceleration, and jerk plot.

The source recording and generated per-sample tables are intentionally omitted
from the public portfolio repository. Recalibrate and teach a new trajectory
for a different arm, mount, or workcell.

