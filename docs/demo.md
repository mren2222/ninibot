# Demo operation

Ninibot provides one end-to-end demo and several smaller launches for
independent skill testing. All launch files default to `dry_run:=True`.

From the repository root:

```bash
cd ~/ninibot
source install/setup.bash
```

## 1. Launch organization

```text
src/ninibot/launch/
  demos/
    grasp_and_move_cup_demo.launch.py
    full_demo.launch.py
  debug/
    grasp_cup.launch.py
    move_cup.launch.py
    press_button.launch.py
  learning/
    preview_policy_demo.launch.py
    collect_fine_align_episodes.launch.py
    record_prep_trajectory.launch.py
    preview_prep_trajectory.launch.py
```

The package does not keep duplicate compatibility wrappers at the launch root.
Use the actual launch filename, such as `full_demo.launch.py` or
`press_button.launch.py`.

## 2. Safe end-to-end preview

Start the complete grasp, transfer, and button-press flow without commanding
physical motors:

```bash
ros2 launch ninibot full_demo.launch.py \
  dry_run:=True \
  detector_backend:=hsv
```

The shorter grasp-and-transfer chain is:

```bash
ros2 launch ninibot grasp_and_move_cup_demo.launch.py \
  dry_run:=True \
  detector_backend:=hsv
```

## 3. Cup-detector backends

HSV is the default public backend:

```bash
ros2 launch ninibot grasp_and_move_cup_demo.launch.py \
  dry_run:=True \
  detector_backend:=hsv
```

YOLO and hybrid modes require user-supplied model weights:

```bash
ros2 launch ninibot grasp_and_move_cup_demo.launch.py \
  dry_run:=True \
  detector_backend:=yolo \
  yolo_model_path:=/absolute/path/to/model.pt
```

For a low-confidence diagnostic run:

```bash
ros2 launch ninibot grasp_and_move_cup_demo.launch.py \
  dry_run:=True \
  detector_backend:=yolo \
  yolo_model_path:=/absolute/path/to/model.pt \
  yolo_conf:=0.20
```

Lowering the threshold increases candidate detections and is not, by itself,
evidence of improved perception.

## 4. Supervised physical demo

The full demo performs cup search, visual alignment, grasp, transfer, placement,
and marker-guided button pressing.

After calibrating the camera and arm, validating every phase in dry-run, and
clearing the workcell:

```bash
ros2 launch ninibot full_demo.launch.py \
  dry_run:=False \
  detector_backend:=hsv \
  port:=/dev/ttyACM0 \
  camera_device:="" \
  camera_index:=0
```

Keep an emergency stop within reach. The parameters and motion envelope are
tuned for one workcell and must be recalibrated for another setup.

## 5. Independent skill debugging

Test the grasp action wrapper:

```bash
ros2 launch ninibot grasp_cup.launch.py dry_run:=True
```

Test cup transfer and trigger one transfer command:

```bash
ros2 launch ninibot move_cup.launch.py dry_run:=True
```

In a second terminal:

```bash
ros2 topic pub --once \
  /move_cup_command \
  std_msgs/msg/String \
  "{data: start}"
```

Test the button-press chain:

```bash
ros2 launch ninibot press_button.launch.py dry_run:=True
```

Independent launches make it possible to inspect one state machine at a time
before enabling the complete orchestration layer.

## 6. Runtime monitoring

```bash
ros2 topic echo /cup_servo_status
ros2 topic echo /cup_servo_debug
ros2 topic echo /move_cup_status
ros2 topic echo /task_manager_status
ros2 topic info /joint_states
ros2 topic info /target_joint_states
```

The status topics expose the active phase and make the demo easier to diagnose
than a single opaque end-to-end process.
