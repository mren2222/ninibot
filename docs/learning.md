# Learning

当前 learning 主线是先用 rule-based controller 采集 episode，再训练 structured imitation baseline。

## 1. 数据采集

`episode_logger` 记录:

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

默认 episode 目录:

```text
data/episodes/
```

`episode_logger` 默认不启动；需要采集数据时显式传 `start_episode_logger:=True`。

建议把不同批次的数据分成独立 dataset，避免旧参数、失败实验和新策略数据混在一起:

```text
data/datasets/dataset1/
data/datasets/dataset2/
data/datasets/dataset3/
```

如果要 archive 当前默认目录:

```bash
cd ~/ninibot
ARCHIVE=data/archive/pre_dataset2_$(date +%Y%m%d_%H%M%S)
mkdir -p "$ARCHIVE"
mv data/episodes "$ARCHIVE/episodes"
mkdir -p data/episodes
```

新的 fine-align 50 次采集可以直接写到 dataset2:

```bash
source install/setup.bash
ros2 launch ninibot collect_fine_align_episodes.launch.py \
  max_episodes:=50 \
  start_episode_logger:=True \
  save_images:=False \
  episode_root:=data/datasets/dataset2
```

采集 demonstration 时默认使用 rule-based finealign。

## 2. 训练 structured policy

```bash
python3 tools/ml/train_structured_policy.py
```

只训练 fine-align:

```bash
python3 tools/ml/train_structured_policy.py \
  --cup-status FINE_ALIGN \
  --output-dir data/models/structured_policy_fine_align
```

面向 integrated control 的推荐模型是 cylindrical fine-align policy。最小输入使用:

```text
error_x
bbox_error
theta_deg
servo_radius
servo_z
```

其中 `error_x` 和 `bbox_error` 来自视觉检测，`theta_deg`、`servo_radius`、`servo_z` 是当前 cylindrical pose。模型只输出:

```text
action_delta_theta
action_delta_radius
```

运行时 `fine_align_policy_gate` 会把 `theta/radius/z + delta` 送进 cylindrical IK，得到 joint1/2/3/4，并保持 z 和 joint4 水平。也就是说训练不需要直接预测关节。

dataset2 的推荐训练命令:

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

`--max-target-abs action_delta_radius=0.003` 会跳过半径 delta 大于 3mm 的异常行，主要用于过滤状态切换或第一帧初始化时的 radius 跳变。

## 3. 离线评估

```bash
python3 tools/ml/evaluate_structured_policy.py --plot
```

评估某个 episode:

```bash
python3 tools/ml/evaluate_structured_policy.py --episode episode_000021 --plot
```

## 4. RViz policy preview

```bash
ros2 launch ninibot preview_policy_demo.launch.py
```

这会打开 `grasp_and_move_cup` 主链路，并启用:

```text
policy_preview
rviz2
```

policy ghost 发布到 `/policy_joint_states`，不会直接控制真机。

## 5. Fine-align policy control

主 demo 现在固定使用 rule-based fine-align。Policy 相关功能先保留为预览入口:

```bash
ros2 launch ninibot preview_policy_demo.launch.py
```

真机策略控制要重新启用时，再把 gate 作为单独入口接回，不放在主 demo 的常用参数里。

## 6. Fine-align-prep kinesthetic teaching

`FINE_ALIGN_PREP` 主 demo 默认使用处理后的示教轨迹：从 search/head-down 姿态移动到 fine-align 起始姿态。原来的 rule-based/builtin 轨迹不再用于真机默认执行，只作为 RViz 对比和项目展示素材保留。

### 6.1 记录 raw 轨迹

启动示教 launch:

```bash
ros2 launch ninibot record_prep_trajectory.launch.py
```

流程:

```text
record_prep_trajectory
  -> 发布 search pose 到 /target_joint_states
  -> motor_driver 移动到 search pose
  -> teach node 发布 /motor_torque_command: disable
  -> 用手移动 joint2/3/4
  -> 手动发送 finish
  -> 保存 raw CSV 到 data/trajectories/fine_align_prep/sessions/current_recording/
```

结束记录:

```bash
ros2 topic pub --once /fine_align_prep_teach/command std_msgs/msg/String "{data: finish}"
```

观察状态:

```bash
ros2 topic echo /fine_align_prep_teach/status
ros2 topic echo /motor_torque_status
```

### 6.2 处理和平滑

把 raw CSV 转成可 retarget 的 processed trajectory:

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

输出:

```text
<raw_name>_processed.json
<raw_name>_processed.csv
<raw_name>_metrics.json
<raw_name>_plot.png
<raw_name>_comparison.png
latest_processed.json
plot.png
comparison_plot.png
```

处理逻辑:

```text
raw q(t)
-> 裁掉前 trim-start-sec 秒无效段
-> 重采样
-> 计算 delta_shape = q(t) - linear_interpolate(start, end)
-> 用 smoothing spline 拟合 delta_shape，缺 SciPy 时 fallback 到 Savitzky-Golay/moving average
-> 对原始记录中低速持续段加 spline 权重，让平台段尽量保持平台，不被平滑成反向小动作
-> joint4 不再使用记录的 joint4 shape，而是实时计算 horizontal_wrist(joint2, joint3) + offset
-> offset 使用 smootherstep 从起点低头 offset 平滑衰减到 0，让终点回到水平 wrist
-> 强制 delta_shape 首尾为 0
-> 保存成可 retarget 模板
```

运行时使用:

```text
joint2/3 = interpolate(current_start, desired_end) + delta_shape(t)
joint4 = horizontal_wrist(joint2, joint3) + start_offset * (1 - smootherstep(t))
```

这样 joint2/3 保留示教轨迹形状，但 joint4 永远跟随当前 joint2/3 的水平 wrist 解，并平滑回到水平。

`plot.png` 是当前 session 内最近一次 processed runtime trajectory 的稳定文件名，内容是已经 retarget 到当前 demo search 起点和 fine-align-prep 终点后的 joint2/3/4 位置、速度、加速度和 jerk。`comparison_plot.png` 对比同一个 retarget 后的 collected shape 和 processed trajectory。

### 6.3 RViz 对比 builtin reference 和 processed trajectory

```bash
ros2 launch ninibot preview_prep_trajectory.launch.py
```

这会发布:

```text
/fine_align_prep_builtin_joint_states
/fine_align_prep_builtin_path
/fine_align_prep_processed_joint_states
/fine_align_prep_processed_path
```

RViz 中显示两条 ghost/path:

```text
builtin reference: 原 rule-based 轨迹，只用于对比和展示
processed: session processed_template/latest_processed JSON 中的处理后轨迹，joint2/3 来自示教 shape，joint4 实时水平 + 平滑 offset
```

preview 启动时会把 builtin reference 单独保存到:

```text
data/trajectories/fine_align_prep/references/rule_based_fine_align_prep_reference.json
```

这个文件是展示项目时可以直接引用的“优化前路径”素材。真正部署时使用 session 里的 processed template，不依赖这个 reference 文件。

### 6.4 主 demo 默认使用 processed FINE_ALIGN_PREP

现在主 demo 默认使用 `data/trajectories/fine_align_prep/sessions/2026-06-15_8cm_base_raw/processed_template.json`，不再需要 `fine_align_prep_mode`。`FINE_ALIGN_PREP` 的终点唯一来自 processed template 里的 `processed_end_rad`；旧的 `fine_align_prep_joint2/3` 参数入口已经删除:

```bash
ros2 launch ninibot grasp_and_move_cup_demo.launch.py
```

播放时长或 shape 强度目前直接在 `src/ninibot/ninibot/task_params.py` 里调，等出现第二套稳定场景后再重新引入场景 YAML。

processed fine-align-prep 只替换 `FINE_ALIGN_PREP` 阶段的 joint2/3/4；joint1 仍由视觉 tracking 的 theta 决定，joint5/6 保持原逻辑。
