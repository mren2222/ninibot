# Demo

cd ~/ninibot
source install/setup.bash
ros2 launch ninibot full_demo.launch.py

## 1. Launch 分类

新 launch 结构:

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
```

根目录不再保留兼容 wrapper。命令里直接使用真实 launch 文件名，例如 `full_demo.launch.py` 或 `press_button.launch.py`。

## 2. Dry-run

不动真机，只跑 dry-run 链路:

```bash
ros2 launch ninibot grasp_and_move_cup_demo.launch.py dry_run:=True
```

## 3. 抓杯 + 移杯

默认 HSV:

```bash
ros2 launch ninibot grasp_and_move_cup_demo.launch.py
```

手动切到 YOLO:

```bash
ros2 launch ninibot grasp_and_move_cup_demo.launch.py detector_backend:=yolo
```

YOLO 检出不稳定时降低置信度:

```bash
ros2 launch ninibot grasp_and_move_cup_demo.launch.py detector_backend:=yolo yolo_conf:=0.20
```

## 4. 完整 demo

抓杯、移动杯、按按钮:

```bash
ros2 launch ninibot full_demo.launch.py
```

如果只想先跑抓杯和移杯，用 `grasp_and_move_cup_demo.launch.py`。

## 5. 单独 debug

单独测试抓杯 action wrapper:

```bash
ros2 launch ninibot grasp_cup.launch.py dry_run:=True
```

单独测试移杯:

```bash
ros2 launch ninibot move_cup.launch.py dry_run:=True
ros2 topic pub --once /move_cup_command std_msgs/msg/String "{data: start}"
```

单独测试按按钮:

```bash
ros2 launch ninibot press_button.launch.py dry_run:=True
```

## 6. 监控

```bash
ros2 topic echo /cup_servo_status
ros2 topic echo /cup_servo_debug
ros2 topic echo /move_cup_status
ros2 topic echo /task_manager_status
ros2 topic info /joint_states
ros2 topic info /target_joint_states
```
