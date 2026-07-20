# Structure

当前结构把底层能力和任务编排分开：底层 controller/perception package 不依赖主应用，主应用 `ninibot` 负责把它们组合成技能和 demo。

## ROS 包

```text
src/
  arm_controller/
  audio_controller/
  vision_controller/
  ninibot_interfaces/
  ninibot/
```

## arm_controller

负责机械臂本体、运动学、joint state 和真机 bridge。

```text
arm_controller/
  kinematics/
    cylindrical.py
    point_ik.py
  motor/
    motor_driver.py
    target_to_rviz.py
    read_arm.py
  debug/
    gripper_pose_debug.py
    keyboard_point.py
    static_joints.py
```

console script 直接指向这些子目录模块，包根目录不再保留同名 wrapper。

## vision_controller

负责 perception，不承载任务状态机或 demo 编排。

```text
vision_controller/
  aruco_detection.py
  aruco_tf.py
  hsv_detector.py
  hybrid_detector.py
  yolo_detector.py
  assets/
    aruco/
      marker PNG/PDF print assets
      generate_aruco.py
      pngs_to_letter_pdf.py
```

ArUco 打印资源随 `vision_controller` 安装到 `share/vision_controller/assets/aruco/`。

## ninibot_interfaces

只放 ROS interface/action，避免 runtime 逻辑和消息定义耦合。

```text
action/
  GraspCup.action
  MoveCup.action
```

## ninibot

主应用 package，调用 `arm_controller`、`audio_controller`、`vision_controller` 和 `ninibot_interfaces`，承载上层技能、动作封装、任务编排、学习/策略工具、launch 和配置。

```text
ninibot/
  skills/
    grasp_cup_node.py
    move_cup_node.py
    press_button_node.py
    random_place_cup_node.py
    shutdown_on_task_status.py
  actions/
    grasp_cup_action_server.py
    move_cup_action_server.py
  orchestration/
    full_demo_manager.py
    episode_loop_node.py
    fine_align_episode_collector.py
  logging/
    episode_logger.py
  policies/
    policy_preview.py
    fine_align_policy_gate.py
  trajectories/
    prep_trajectory.py
    record_prep_trajectory.py
    process_prep_trajectory.py
    preview_prep_trajectory.py
  task_params.py
launch/
  demos/
  debug/
  learning/
config/
  aruco_follower.yaml
  motor.yaml
rviz/
```

console scripts 直接指向 `ninibot.*` 子目录模块。launch 文件也安装在 `share/ninibot/launch/...`，所以常用入口是 `ros2 launch ninibot ...`。
