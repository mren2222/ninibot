# Learning and taught-trajectory experiments

Ninibot keeps learned components separate from the production demo. The main
manipulation flow uses explicit state machines and rule-based fine alignment;
the learning path is an experimental baseline for data collection, offline
evaluation, and RViz visualization.

Raw episodes and trained model weights are intentionally excluded from this
public portfolio repository.

## 1. Structured episode collection

`episode_logger` records synchronized structured signals:

```text
real_joint_states
target_joint_states
cup_servo_status
cup_servo_debug
move_cup_status
task_manager_status
action deltas
metadata
```

The default output directory is:

```text
data/episodes/
```

Logging is disabled by default. Enable it explicitly with
`start_episode_logger:=True`.

Keep collection batches in separate dataset directories so controller changes,
failed experiments, and newer demonstrations remain traceable:

```text
data/datasets/dataset1/
data/datasets/dataset2/
data/datasets/dataset3/
```

To archive the current default episode directory:

```bash
cd ~/ninibot
ARCHIVE=data/archive/pre_dataset2_$(date +%Y%m%d_%H%M%S)
mkdir -p "$ARCHIVE"
mv data/episodes "$ARCHIVE/episodes"
mkdir -p data/episodes
```

This command moves the existing episode directory; inspect the paths before
running it.

A supervised 50-episode fine-alignment collection can write directly to
`dataset2`:

```bash
source install/setup.bash

ros2 launch ninibot collect_fine_align_episodes.launch.py \
  dry_run:=False \
  max_episodes:=50 \
  start_episode_logger:=True \
  save_images:=False \
  episode_root:=data/datasets/dataset2
```

Use `dry_run:=False` only for a supervised, calibrated real-arm collection.
Use `dry_run:=True` first to validate the logging and orchestration pipeline.
Demonstrations use the rule-based fine-alignment controller.

## 2. Train a structured imitation policy

The baseline predicts control deltas from structured state rather than raw
images:

```bash
python3 tools/ml/train_structured_policy.py \
  --episode-root data/episodes \
  --output-dir data/models/structured_policy
```

Train only on the `FINE_ALIGN` state:

```bash
python3 tools/ml/train_structured_policy.py \
  --episode-root data/episodes \
  --cup-status FINE_ALIGN \
  --output-dir data/models/structured_policy_fine_align
```

The compact cylindrical policy uses these inputs:

```text
error_x
bbox_error
theta_deg
servo_radius
servo_z
```

`error_x` and `bbox_error` come from vision. `theta_deg`, `servo_radius`, and
`servo_z` describe the current cylindrical pose. The model predicts only:

```text
action_delta_theta
action_delta_radius
```

At runtime, `fine_align_policy_gate` adds those deltas to the current
cylindrical state and runs cylindrical IK to derive joints 1–4. The learned
model therefore does not predict joint angles directly.

Recommended command for the `dataset2` experiment:

```bash
python3 tools/ml/train_structured_policy.py \
  --episode-root data/datasets/dataset2 \
  --cup-status FINE_ALIGN \
  --target-column action_delta_theta \
  --target-column action_delta_radius \
  --feature-column error_x \
  --feature-column bbox_error \
  --feature-column theta_deg \
  --feature-column servo_radius \
  --feature-column servo_z \
  --max-target-abs action_delta_radius=0.003 \
  --output-dir data/models/finealign_dataset2_cyl_min \
  --epochs 500 \
  --hidden-dim 128
```

`--max-target-abs action_delta_radius=0.003` removes rows whose radial delta
exceeds 3 mm. This filters discontinuities commonly introduced by state
transitions or first-frame initialization.

## 3. Offline evaluation

Evaluate a trained model against its validation split:

```bash
python3 tools/ml/evaluate_structured_policy.py \
  --model data/models/finealign_dataset2_cyl_min/policy.pt \
  --episode-root data/datasets/dataset2 \
  --split val \
  --plot
```

Evaluate a specific episode:

```bash
python3 tools/ml/evaluate_structured_policy.py \
  --model data/models/finealign_dataset2_cyl_min/policy.pt \
  --episode-root data/datasets/dataset2 \
  --episode episode_000021 \
  --plot
```

Offline action error is useful for model comparison, but it is not evidence of
real-arm task success, robustness, or safe motion.

## 4. RViz policy preview

Supply a trained model explicitly:

```bash
ros2 launch ninibot preview_policy_demo.launch.py \
  dry_run:=True \
  policy_model_path:=$PWD/data/models/finealign_dataset2_cyl_min/policy.pt
```

The launch starts the grasp-and-transfer chain in dry-run mode, starts RViz,
and publishes the model prediction as `/policy_joint_states`. The policy ghost
is visualized separately and does not directly control the physical arm.

The main demo remains rule-based. Real-arm policy control is intentionally not
exposed through the standard demo launch.

## 5. Fine-align-prep kinesthetic teaching

The `FINE_ALIGN_PREP` phase moves the arm from its search/head-down pose to the
fine-alignment starting pose. The main demo uses a processed kinesthetic
trajectory for this phase. The earlier built-in trajectory remains only as an
RViz comparison and project artifact.

### 5.1 Record a raw trajectory

Start with the workspace clear and the arm supported. The recording launch
defaults to `dry_run:=True`; physical kinesthetic teaching requires an explicit,
supervised real-arm run:

```bash
ros2 launch ninibot record_prep_trajectory.launch.py dry_run:=False
```

The sequence is:

```text
record_prep_trajectory
  -> publish the search pose to /target_joint_states
  -> motor_driver moves to the search pose
  -> the teaching node publishes /motor_torque_command: disable
  -> the operator manually moves joints 2, 3, and 4
  -> the operator sends the finish command
  -> save raw CSV and metadata under
     data/trajectories/fine_align_prep/sessions/current_recording/
```

Finish the recording:

```bash
ros2 topic pub --once \
  /fine_align_prep_teach/command \
  std_msgs/msg/String \
  "{data: finish}"
```

Monitor teaching and torque state:

```bash
ros2 topic echo /fine_align_prep_teach/status
ros2 topic echo /motor_torque_status
```

The stable raw aliases are `latest_raw.csv` and `latest_raw.json`.

### 5.2 Process and smooth the recording

Convert the raw CSV into a retargetable trajectory template:

```bash
ros2 run ninibot process_prep_trajectory \
  data/trajectories/fine_align_prep/sessions/current_recording/latest_raw.csv \
  --trim-start-sec 12.0 \
  --trim-end-sec 1.0 \
  --path-smoothing spline \
  --joint4-mode horizontal_offset \
  --spline-smoothing 0.0003 \
  --flat-weight 20.0 \
  --no-latest
```

The current processor writes these fixed filenames:

```text
processed_template.json
processed_samples.csv
processing_metrics.json
processed_joint_dynamics.png
raw_vs_processed_comparison.png
```

The two PNG files are written when plotting dependencies are available. Because
the example command uses `--no-latest`, it does not create
`latest_processed.json`, `plot.png`, or `comparison_plot.png`. Omit
`--no-latest` when those convenience aliases are desired.

The processing pipeline is:

```text
raw q(t)
  -> trim invalid time from the beginning and end
  -> resample
  -> compute delta_shape = q(t) - linear_interpolate(start, end)
  -> fit delta_shape with a smoothing spline
  -> fall back to Savitzky–Golay or moving-average smoothing without SciPy
  -> give sustained low-velocity regions additional spline weight
  -> preserve those plateaus instead of smoothing them into reverse motion
  -> recompute joint 4 as horizontal_wrist(joint2, joint3) + offset
  -> decay the initial wrist offset to zero with smootherstep
  -> force delta_shape to zero at both endpoints
  -> save a retargetable template
```

Runtime retargeting uses:

```text
joint2/3 = interpolate(current_start, desired_end) + delta_shape(t)
joint4   = horizontal_wrist(joint2, joint3)
           + start_offset * (1 - smootherstep(t))
```

This preserves the demonstrated shape for joints 2 and 3 while keeping joint 4
consistent with a horizontal-wrist solution and smoothly removing the initial
head-down offset.

`processed_joint_dynamics.png` plots position, velocity, acceleration, and jerk
for the processed taught-start-to-taught-end trajectory.
`raw_vs_processed_comparison.png` compares the retargeted raw shape and the
processed trajectory over those same taught endpoints.

### 5.3 Compare the built-in and processed paths in RViz

```bash
ros2 launch ninibot preview_prep_trajectory.launch.py
```

The preview publishes:

```text
/fine_align_prep_builtin_joint_states
/fine_align_prep_builtin_path
/fine_align_prep_processed_joint_states
/fine_align_prep_processed_path
```

RViz displays two ghosts and paths:

```text
built-in reference
  Earlier rule-based trajectory retained for comparison only

processed
  Joint 2/3 shape from the taught template, with joint 4 recomputed as a
  horizontal wrist plus the smoothed initial offset
```

The preview also saves the built-in comparison trajectory to:

```text
data/trajectories/fine_align_prep/references/rule_based_fine_align_prep_reference.json
```

That file is a reproducible “before” artifact for project presentation. Runtime
deployment uses the processed session template and does not depend on the
reference file.

### 5.4 Processed trajectory used by the main demo

The current default is:

```text
data/trajectories/fine_align_prep/sessions/2026-06-15_8cm_base_raw/processed_template.json
```

The `FINE_ALIGN_PREP` endpoint comes from the template's
`processed_end_rad`; the former `fine_align_prep_joint2/3` parameter path has
been removed.

Start the main chain with:

```bash
ros2 launch ninibot grasp_and_move_cup_demo.launch.py dry_run:=True
```

Playback duration and shape scale are currently configured in
`src/ninibot/ninibot/task_params.py`. A scene-specific YAML layer can be added
when multiple calibrated workcells need stable parameter sets.

The processed template replaces joints 2, 3, and 4 only during
`FINE_ALIGN_PREP`. Joint 1 remains controlled by the vision-derived azimuth,
while joints 5 and 6 retain their existing behavior.
