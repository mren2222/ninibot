# System bring-up

This guide describes the shortest path from a built ROS 2 workspace to a
running Ninibot manipulation chain.

> **Safety:** all public launch files default to `dry_run:=True`. Keep that
> setting until the camera, transforms, joint limits, target poses, and complete
> motion sequence have been checked. A successful dry-run is not proof of
> collision safety.

## 1. Build and source the workspace

Run all commands from the repository root because several public defaults use
paths relative to the current working directory.

```bash
cd ~/ninibot
source /opt/ros/jazzy/setup.bash

colcon build --packages-select \
  arm_controller \
  audio_controller \
  vision_controller \
  ninibot_interfaces \
  ninibot

source install/setup.bash
```

Rebuild and source the workspace again after changing source code.

## 2. Start with the dry-run chain

This launch starts cup perception, grasp and transfer skills, the action
servers, task orchestration, and the motor bridge without commanding the
physical arm:

```bash
cd ~/ninibot
source install/setup.bash

ros2 launch ninibot grasp_and_move_cup_demo.launch.py \
  dry_run:=True \
  detector_backend:=hsv
```

The HSV backend is the public default and does not require a model file.

## 3. Select the camera

The shared camera defaults are defined in `ninibot.task_params`:

```text
camera_device = ""
camera_index  = 0
```

An empty `camera_device` makes OpenCV use `camera_index`. For a fixed workcell,
prefer a stable `/dev/v4l/by-id/...` device path because `/dev/video0`-style
indices can change when USB devices are reconnected.

Force camera index 0:

```bash
ros2 launch ninibot grasp_and_move_cup_demo.launch.py \
  dry_run:=True \
  camera_device:="" \
  camera_index:=0
```

Use a stable device path:

```bash
ros2 launch ninibot grasp_and_move_cup_demo.launch.py \
  dry_run:=True \
  camera_device:=/dev/v4l/by-id/YOUR_CAMERA_DEVICE
```

## 4. Configure the physical arm

The default SO-101 serial port is:

```text
/dev/ttyACM0
```

Override it at launch time when necessary:

```bash
ros2 launch ninibot grasp_and_move_cup_demo.launch.py \
  dry_run:=True \
  port:=/dev/ttyACM0
```

Only after calibration and supervised dry-run validation should the physical
motor bridge be enabled:

```bash
ros2 launch ninibot grasp_and_move_cup_demo.launch.py \
  dry_run:=False \
  detector_backend:=hsv \
  port:=/dev/ttyACM0
```

Keep an emergency stop within reach and keep people and fragile objects outside
the arm envelope.

## 5. Preview an experimental policy

The policy path is intentionally separate from the main demo. The public
repository does not include trained weights, so provide a model produced by
`tools/ml/train_structured_policy.py`:

```bash
ros2 launch ninibot preview_policy_demo.launch.py \
  dry_run:=True \
  policy_model_path:=$PWD/data/models/structured_policy/policy.pt
```

The predicted pose is published as an RViz ghost on
`/policy_joint_states`; it does not directly command the arm.

## 6. Key topics

```text
/target_joint_states   Target joint commands produced by skills
/joint_states          Commanded motor-driver pose used by RViz
/real_joint_states     Physical feedback, or simulated state in dry-run
/cup_servo_status      Cup-grasp state-machine status
/cup_servo_debug       Numeric cup-servo diagnostics
/move_cup_status       Cup-transfer state
/task_manager_status   End-to-end task state
```

Avoid starting multiple publishers for `/joint_states` or
`/target_joint_states`.

Useful checks:

```bash
ros2 topic info /joint_states
ros2 topic info /target_joint_states
ros2 topic echo /cup_servo_status
ros2 topic echo /move_cup_status
ros2 topic echo /task_manager_status
```

## 7. Package responsibilities

`arm_controller` contains the robot-facing layer:

```text
kinematics/  Cylindrical IK, point IK, and grippertip IK service
motor/       Motor driver, arm-state reader, and RViz joint-state relay
debug/       Keyboard targets, static joints, and gripper-pose diagnostics
```

`vision_controller` contains reusable perception components:

```text
aruco_detection.py  ArUco camera helpers
aruco_tf.py         Camera/base transform helpers
hsv_detector.py     HSV cup detector
yolo_detector.py    YOLO cup detector
hybrid_detector.py  YOLO-gated HSV detector
```

`ninibot` is the application layer that combines the controller, perception,
audio, and interface packages:

```text
skills/         Grasp, transfer, placement, and button-press behavior
actions/        ROS 2 action-server wrappers
orchestration/  Task manager and episode loop
logging/        Structured episode logger
policies/       Experimental policy preview and policy gate
trajectories/   Kinesthetic teaching, processing, replay, and preview
```

Active code imports these modules directly, for example
`ninibot.skills.grasp_cup_node` and `vision_controller.aruco_detection`.
