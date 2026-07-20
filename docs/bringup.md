# Bringup

这份文档记录从空终端到能启动 Ninibot ROS 链路的最小流程。

```bash
cd ~/ninibot
source install/setup.bash

ros2 launch ninibot grasp_and_move_cup_demo.launch.py \
  dry_run:=False \
  detector_backend:=hsv
```

Policy 只做预览时走专门的 launch，不混进主 demo:

```bash
ros2 launch ninibot preview_policy_demo.launch.py
```

  


## 1. 准备环境

```bash
cd ~/ninibot
source install/setup.bash
```

改过源码后重新 build:

```bash
colcon build --packages-select arm_controller audio_controller vision_controller ninibot_interfaces ninibot
source install/setup.bash
```

## 2. 相机

默认相机配置在 `ninibot.task_params`:

```text
camera_device = ""
camera_index  = 0
```

建议优先使用 `camera_device`，因为 `/dev/video0` 这类 index 会随插拔顺序变化。

如果要强制使用 index:

```bash
ros2 launch ninibot grasp_and_move_cup_demo.launch.py camera_device:="" camera_index:=0
```

## 3. 真机串口

默认串口:

```text
/dev/ttyACM0
```

启动时可覆盖:

```bash
ros2 launch ninibot grasp_and_move_cup_demo.launch.py port:=/dev/ttyACM0
```

## 4. 关键 topic

```text
/target_joint_states   # skill 输出的目标关节
/joint_states          # motor_driver 发布的命令姿态，用于 RViz
/real_joint_states     # 真机反馈或 dry-run 当前姿态
/cup_servo_status      # 抓杯状态机状态
/cup_servo_debug       # 抓杯 debug 数值
/move_cup_status       # 移杯状态
/task_manager_status   # 总任务状态
```

避免同时启动多个 `/joint_states` 或 `/target_joint_states` publisher。

## 5. 结构说明

`arm_controller` 现在按职责分包:

```text
kinematics/  # cylindrical IK and IK ROS node
motor/       # motor_driver, read_arm, joint-state relay
debug/       # keyboard target, static joint states, gripper pose debug
```

`vision_controller` 现在只保留视觉相关能力:

```text
aruco_detection.py  # ArUco camera helpers
aruco_tf.py         # camera/base TF helpers
hsv_detector.py     # HSV cup detector
yolo_detector.py    # YOLO cup detector
```

`ninibot` 是主应用 package，调用 `arm_controller`、`audio_controller`、`vision_controller` 和 `ninibot_interfaces`:

```text
skills/         # cup vision, move cup, press button, action wrappers
orchestration/  # task manager, episode loop
logging/        # episode logger
policies/       # policy preview, fine-align policy gate
trajectories/   # teach/replay trajectory tools
```

活跃代码直接 import 子目录模块，例如 `ninibot.skills.grasp_cup_node` 和 `vision_controller.aruco_detection`。
