# Structured policy experiment

This directory contains an **experimental offline behavior-cloning baseline**
using structured episode values rather than images. It is separate from the
rule/state-machine controller used by the main demo.

Raw episodes and trained model files are intentionally excluded from this
public portfolio repository.

## Requirements

- PyTorch for training and evaluation
- Matplotlib only when using the optional evaluation plot
- Episode CSV files produced by the Ninibot episode logger

Install these into an isolated Python environment compatible with the ROS
workspace.

## Train

From the repository root:

```bash
python3 tools/ml/train_structured_policy.py \
  --episode-root data/episodes \
  --output-dir data/models/structured_policy
```

The trainer discovers labeled episode CSV files, makes a deterministic
train/validation split, normalizes the selected structured features, and writes
the model plus metadata to the output directory. Use `--help` to inspect
feature, target, filtering, and optimization options.

## Evaluate offline

```bash
python3 tools/ml/evaluate_structured_policy.py \
  --model data/models/structured_policy/policy.pt \
  --episode-root data/episodes \
  --split val \
  --plot
```

This writes offline metrics and prediction rows. Offline error is not evidence
of real-arm task success or safe motion.

## Preview in RViz

After building and sourcing the workspace:

```bash
ros2 launch ninibot preview_policy_demo.launch.py \
  dry_run:=True \
  policy_model_path:=$PWD/data/models/structured_policy/policy.pt
```

The public preview launch keeps the included manipulation chain in dry-run mode
by default and displays the policy output as a separate RViz ghost. It does not
expose learned-policy real-arm control. Do not set `dry_run:=False` merely to
preview a model.

