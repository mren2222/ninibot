# flake8: noqa
import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String


class JointLimiter:
    """
    One joint independent velocity/acceleration limiter.

    It moves from current position toward goal position.
    Each joint moves independently.
    """

    def __init__(self, max_vel, max_acc):
        self.max_vel = max_vel      # rad/s
        self.max_acc = max_acc      # rad/s^2
        self.pos = 0.0              # rad
        self.vel = 0.0              # rad/s
        self.goal = 0.0             # rad

    def reset(self, pos):
        self.pos = pos
        self.vel = 0.0
        self.goal = pos

    def set_goal(self, goal):
        self.goal = goal

    def step(self, dt):
        error = self.goal - self.pos

        if abs(error) < 1e-6 and abs(self.vel) < 1e-6:
            self.pos = self.goal
            self.vel = 0.0
            return self.pos

        # Braking distance: d = v^2 / (2a)
        stopping_dist = (self.vel * self.vel) / (2.0 * self.max_acc + 1e-9)

        # If close enough that we need to brake, target velocity is zero.
        # Otherwise target velocity is max velocity toward goal.
        if abs(error) <= stopping_dist:
            target_vel = 0.0
        else:
            target_vel = math.copysign(self.max_vel, error)

        # Limit acceleration
        dv = target_vel - self.vel
        max_dv = self.max_acc * dt
        dv = max(-max_dv, min(max_dv, dv))
        self.vel += dv

        next_pos = self.pos + self.vel * dt

        # Avoid overshoot
        if (error > 0 and next_pos >= self.goal) or (error < 0 and next_pos <= self.goal):
            self.pos = self.goal
            self.vel = 0.0
        else:
            self.pos = next_pos

        return self.pos


def load_lerobot_so_follower():
    try:
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        from lerobot.robots.so_follower.so_follower import SOFollower
    except ModuleNotFoundError as exc:
        if exc.name == "lerobot":
            raise RuntimeError(
                "Missing Python package 'lerobot'. Install it in the same Python "
                "environment used by ROS 2, or launch with dry_run:=True."
            ) from exc
        raise

    return SOFollower, SOFollowerRobotConfig


class MotorDriver(Node):
    def __init__(self):
        super().__init__("motor_driver")

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("port", "/dev/ttyACM0")
        self.declare_parameter("input_topic", "/target_joint_states")
        self.declare_parameter("publish_topic", "/joint_states")
        self.declare_parameter("real_state_topic", "/real_joint_states")
        self.declare_parameter("torque_command_topic", "/motor_torque_command")
        self.declare_parameter("torque_status_topic", "/motor_torque_status")
        self.declare_parameter("real_state_rate", 20.0)

        # Default 500 Hz, so you can start without passing parameters.
        self.declare_parameter("control_rate", 500.0)

        # Default to real robot mode.
        # If you only want RViz test, launch with -p dry_run:=True.
        self.declare_parameter("dry_run", False)

        # Easy global speed scaling
        self.declare_parameter("speed_scale", 1.0)
        self.declare_parameter("accel_scale", 1.0)
        self.declare_parameter("slow_status_topics", "")
        self.declare_parameter("slow_statuses", "")
        self.declare_parameter("slow_status_speed_scale", 1.0)
        self.declare_parameter("slow_status_accel_scale", 1.0)
        self.declare_parameter("slow_status_timeout_sec", 0.5)
        self.declare_parameter("torque_limit", -1)
        self.declare_parameter("torque_limit_motors", "")

        # Per-joint max velocity in deg/s
        self.declare_parameter("j1_max_vel_deg", 60.0)
        self.declare_parameter("j2_max_vel_deg", 60.0)
        self.declare_parameter("j3_max_vel_deg", 60.0)
        self.declare_parameter("j4_max_vel_deg", 120.0)
        self.declare_parameter("j5_max_vel_deg", 60.0)
        self.declare_parameter("j6_max_vel_deg", 60.0)

        # Per-joint max acceleration in deg/s^2
        self.declare_parameter("j1_max_acc_deg", 60.0)
        self.declare_parameter("j2_max_acc_deg", 60.0)
        self.declare_parameter("j3_max_acc_deg", 60.0)
        self.declare_parameter("j4_max_acc_deg", 120.0)
        self.declare_parameter("j5_max_acc_deg", 60.0)
        self.declare_parameter("j6_max_acc_deg", 60.0)

        # Motor send deadband in degrees.
        # Keep small. If you want to send every 500 Hz tick, set this to 0.0.
        self.declare_parameter("action_deadband_deg", 0.05)

        self.port = self.get_parameter("port").value
        self.input_topic = self.get_parameter("input_topic").value
        self.publish_topic = self.get_parameter("publish_topic").value
        self.real_state_topic = self.get_parameter("real_state_topic").value
        self.torque_command_topic = self.get_parameter("torque_command_topic").value
        self.torque_status_topic = self.get_parameter("torque_status_topic").value
        self.real_state_rate = self.get_parameter("real_state_rate").value
        self.control_rate = self.get_parameter("control_rate").value
        self.dry_run = self.get_parameter("dry_run").value
        self.speed_scale = self.get_parameter("speed_scale").value
        self.accel_scale = self.get_parameter("accel_scale").value
        self.slow_status_topics = self.parse_csv_parameter(
            self.get_parameter("slow_status_topics").value
        )
        self.slow_statuses = set(
            self.parse_csv_parameter(self.get_parameter("slow_statuses").value)
        )
        self.slow_status_speed_scale = max(
            0.01,
            float(self.get_parameter("slow_status_speed_scale").value),
        )
        self.slow_status_accel_scale = max(
            0.01,
            float(self.get_parameter("slow_status_accel_scale").value),
        )
        self.slow_status_timeout_sec = max(
            0.0,
            float(self.get_parameter("slow_status_timeout_sec").value),
        )
        self.torque_limit = int(self.get_parameter("torque_limit").value)
        self.torque_limit_motors = self.parse_csv_parameter(
            self.get_parameter("torque_limit_motors").value
        )
        self.action_deadband_deg = self.get_parameter("action_deadband_deg").value

        self.dt = 1.0 / self.control_rate

        self.ros_joint_names = ["1", "2", "3", "4", "5", "6"]
        self.input_joint_names = ["1", "2", "3", "4"]

        # Wrist roll is fixed here; wrist pitch and gripper come from targets.
        self.wrist_roll_ros_deg = -90.0
        self.wrist_roll_real_deg = -90.0
    
        self.gripper_percent = 50.0

        # ROS rad -> LeRobot degree mapping for joints 1-4.
        self.real_signs = [1, 1, 1, 1]
        self.real_offsets_deg = [0, 0, 0, 0]

        # Gripper mapping
        self.gripper_min_rad = -0.174533
        self.gripper_max_rad = 1.74533

        # -----------------------------
        # Build per-joint limiters
        # -----------------------------
        max_vel_deg = [
            self.get_parameter("j1_max_vel_deg").value,
            self.get_parameter("j2_max_vel_deg").value,
            self.get_parameter("j3_max_vel_deg").value,
            self.get_parameter("j4_max_vel_deg").value,
            self.get_parameter("j5_max_vel_deg").value,
            self.get_parameter("j6_max_vel_deg").value,
        ]

        max_acc_deg = [
            self.get_parameter("j1_max_acc_deg").value,
            self.get_parameter("j2_max_acc_deg").value,
            self.get_parameter("j3_max_acc_deg").value,
            self.get_parameter("j4_max_acc_deg").value,
            self.get_parameter("j5_max_acc_deg").value,
            self.get_parameter("j6_max_acc_deg").value,
        ]

        self.nominal_max_vel = []
        self.nominal_max_acc = []
        self.limiters = []

        for i in range(6):
            max_vel = math.radians(max_vel_deg[i] * self.speed_scale)
            max_acc = math.radians(max_acc_deg[i] * self.accel_scale)
            self.nominal_max_vel.append(max_vel)
            self.nominal_max_acc.append(max_acc)
            self.limiters.append(JointLimiter(max_vel=max_vel, max_acc=max_acc))

        self.current_positions = None
        self.target_positions = None
        self.pending_target_positions = None
        self.last_action = None
        self.last_real_state_time = 0.0
        self.last_warn_time = 0.0
        self.motor_output_enabled = True
        self.slow_status_active_until = 0.0
        self.active_motion_speed_scale = 1.0
        self.active_motion_accel_scale = 1.0

        # -----------------------------
        # ROS pub/sub
        # -----------------------------
        self.joint_state_pub = self.create_publisher(
            JointState,
            self.publish_topic,
            10,
        )
        self.real_joint_state_pub = self.create_publisher(
            JointState,
            self.real_state_topic,
            10,
        )
        self.torque_status_pub = self.create_publisher(
            String,
            self.torque_status_topic,
            10,
        )

        self.target_sub = self.create_subscription(
            JointState,
            self.input_topic,
            self.target_joint_callback,
            10,
        )
        self.torque_command_sub = self.create_subscription(
            String,
            self.torque_command_topic,
            self.torque_command_callback,
            10,
        )
        self.slow_status_subs = [
            self.create_subscription(
                String,
                topic,
                self.slow_status_callback,
                10,
            )
            for topic in self.slow_status_topics
        ]
        # -----------------------------
        # Robot connection
        # -----------------------------
        self.robot = None

        if self.dry_run:
            self.get_logger().warn("DRY RUN MODE: will not send actions to real robot")
        else:
            try:
                SOFollower, SOFollowerRobotConfig = load_lerobot_so_follower()
            except RuntimeError as exc:
                self.get_logger().error(str(exc))
                raise

            self.get_logger().info(f"Connecting to SO101 on {self.port}")
            config = SOFollowerRobotConfig(port=self.port)
            self.robot = SOFollower(config)
            self.robot.connect(calibrate=False)
            self.apply_torque_limit_override()

            self.current_positions = self.read_current_positions_from_robot_safe(
                retries=10,
                delay_sec=0.1,
            )

            if self.current_positions is not None:
                self.reset_limiters(self.current_positions)
            else:
                self.get_logger().warn(
                    "Connected to SO101, but initial joint read failed; "
                    "waiting for a valid /real_joint_states read before commanding"
                )

            self.get_logger().warn("Connected to SO101: real robot control enabled")

        self.timer = self.create_timer(self.dt, self.timer_callback)

        self.get_logger().info(
            f"motor_driver started: input={self.input_topic}, "
            f"output={self.publish_topic}, torque_command={self.torque_command_topic}, "
            f"rate={self.control_rate:.1f} Hz, "
            f"speed_scale={self.speed_scale:.2f}, accel_scale={self.accel_scale:.2f}, "
            f"slow_statuses={sorted(self.slow_statuses)}"
        )

    def parse_csv_parameter(self, value):
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]

        return [
            item.strip()
            for item in str(value).split(",")
            if item.strip()
        ]

    def apply_torque_limit_override(self):
        if self.robot is None or self.torque_limit < 0 or not self.torque_limit_motors:
            return

        limit = max(0, min(1000, int(self.torque_limit)))
        for motor in self.torque_limit_motors:
            if motor not in self.robot.bus.motors:
                self.get_logger().warn(f"Unknown torque_limit motor: {motor}")
                continue

            try:
                self.robot.bus.write("Torque_Limit", motor, limit)
                self.get_logger().warn(
                    f"Set {motor} Torque_Limit={limit} for this motor session"
                )
            except Exception as exc:
                self.get_logger().error(
                    f"Failed to set {motor} Torque_Limit={limit}: {exc}"
                )

    def publish_torque_status(self, status):
        msg = String()
        msg.data = status
        self.torque_status_pub.publish(msg)

    def torque_command_callback(self, msg):
        command = msg.data.strip().lower()

        if command in ("disable", "off", "free", "release"):
            self.disable_motor_torque()
        elif command in ("enable", "on", "hold"):
            self.enable_motor_torque()
        else:
            self.get_logger().warn(f"Unknown torque command: {msg.data}")

    def disable_motor_torque(self):
        if self.dry_run or self.robot is None:
            self.motor_output_enabled = False
            self.publish_torque_status("DRY_RUN_TORQUE_DISABLED")
            self.get_logger().warn("DRY RUN: simulated torque disable")
            return

        try:
            self.robot.bus.disable_torque()
            self.motor_output_enabled = False
            self.last_action = None
            self.publish_torque_status("TORQUE_DISABLED")
            self.get_logger().warn("SO101 torque disabled; motor actions paused")
        except Exception as exc:
            self.publish_torque_status(f"TORQUE_DISABLE_FAILED: {exc}")
            self.get_logger().error(f"Failed to disable SO101 torque: {exc}")

    def enable_motor_torque(self):
        if self.dry_run or self.robot is None:
            self.motor_output_enabled = True
            self.publish_torque_status("DRY_RUN_TORQUE_ENABLED")
            self.get_logger().warn("DRY RUN: simulated torque enable")
            return

        try:
            self.robot.bus.enable_torque()
            self.sync_controller_to_real_pose()
            self.last_action = None
            self.motor_output_enabled = True
            self.publish_torque_status("TORQUE_ENABLED")
            self.get_logger().warn("SO101 torque enabled; holding current pose")
        except Exception as exc:
            self.publish_torque_status(f"TORQUE_ENABLE_FAILED: {exc}")
            self.get_logger().error(f"Failed to enable SO101 torque: {exc}")

    def sync_controller_to_real_pose(self):
        positions = self.read_current_positions_from_robot_safe(
            retries=5,
            delay_sec=0.05,
        )

        if positions is None:
            return

        self.current_positions = positions
        self.reset_limiters(positions)
        self.set_target_positions(positions)
        self.publish_joint_states(positions)
        self.publish_real_joint_states()

    def reset_limiters(self, positions):
        for limiter, pos in zip(self.limiters, positions):
            limiter.reset(pos)

    def warn_throttled(self, text, period_sec=1.0):
        now = time.monotonic()

        if now - self.last_warn_time >= period_sec:
            self.get_logger().warn(text)
            self.last_warn_time = now

    def set_target_positions(self, target_positions):
        """
        Set target positions for the limiters.

        Joint targets are consumed as-is; upstream nodes own IK and joint4.
        """
        self.target_positions = list(target_positions)

        self.target_positions[4] = math.radians(self.wrist_roll_ros_deg)
        self.target_positions[5] = max(
            self.gripper_min_rad,
            min(self.gripper_max_rad, self.target_positions[5]),
        )
        self.gripper_percent = self.gripper_rad_to_percent(self.target_positions[5])

        for i, limiter in enumerate(self.limiters):
            limiter.set_goal(self.target_positions[i])

    def slow_status_callback(self, msg):
        status = msg.data.strip()
        if status not in self.slow_statuses:
            return

        self.slow_status_active_until = (
            time.monotonic() + self.slow_status_timeout_sec
        )

    def update_motion_scale_from_status(self):
        if time.monotonic() <= self.slow_status_active_until:
            speed_scale = self.slow_status_speed_scale
            accel_scale = self.slow_status_accel_scale
        else:
            speed_scale = 1.0
            accel_scale = 1.0

        if (
            abs(speed_scale - self.active_motion_speed_scale) < 1e-6
            and abs(accel_scale - self.active_motion_accel_scale) < 1e-6
        ):
            return

        self.active_motion_speed_scale = speed_scale
        self.active_motion_accel_scale = accel_scale
        for limiter, max_vel, max_acc in zip(
            self.limiters,
            self.nominal_max_vel,
            self.nominal_max_acc,
        ):
            limiter.max_vel = max_vel * speed_scale
            limiter.max_acc = max_acc * accel_scale

        self.get_logger().info(
            "motor_driver motion scale "
            f"speed={speed_scale:.2f}, accel={accel_scale:.2f}"
        )

    def target_joint_callback(self, msg):
        name_to_position = {}

        for name, position in zip(msg.name, msg.position):
            name_to_position[name] = position

        missing = [
            name for name in self.input_joint_names
            if name not in name_to_position
        ]

        if missing:
            self.get_logger().warn(f"Target JointState missing joints: {missing}")
            return

        new_target = [
            name_to_position["1"],
            name_to_position["2"],
            name_to_position["3"],
            name_to_position["4"],
            math.radians(self.wrist_roll_ros_deg),
            name_to_position.get(
                "6",
                self.gripper_percent_to_rad(self.gripper_percent),
            ),
        ]

        if self.current_positions is None:
            self.pending_target_positions = list(new_target)

            if not self.dry_run and self.robot is not None:
                self.warn_throttled(
                    "Target received, but real joint positions are not ready; "
                    "holding commands until the next successful robot read",
                    1.0,
                )
                return

            # Dry run starts here. Initialize from first target to avoid jump.
            self.current_positions = list(new_target)
            self.reset_limiters(self.current_positions)
            self.set_target_positions(new_target)
            self.publish_joint_states(self.current_positions)
            return

        self.set_target_positions(new_target)

    def timer_callback(self):
        self.update_motion_scale_from_status()

        if self.current_positions is None:
            if not self.dry_run and self.robot is not None:
                positions = self.read_current_positions_from_robot_safe()

                if positions is None:
                    return

                self.current_positions = positions
                self.reset_limiters(self.current_positions)
                self.publish_joint_states(self.current_positions)
                self.publish_real_joint_states()

                if self.pending_target_positions is not None:
                    self.set_target_positions(self.pending_target_positions)
                    self.pending_target_positions = None

                self.get_logger().info("Recovered real joint positions; motor output enabled")

            return

        positions = []

        for i, limiter in enumerate(self.limiters):
            positions.append(limiter.step(self.dt))

        self.current_positions = positions

        # Publish commanded pose to RViz
        self.publish_joint_states(positions)
        self.publish_real_joint_states()

        action = self.ros_positions_to_lerobot_action(positions)

        if not self.dry_run and self.motor_output_enabled:
            if self.action_changed_enough(action):
                self.robot.send_action(action)
                self.last_action = dict(action)

    def publish_joint_states(self, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.ros_joint_names
        msg.position = positions
        self.joint_state_pub.publish(msg)

    def publish_real_joint_states(self):
        now = self.get_clock().now()
        now_sec = now.nanoseconds * 1e-9

        if now_sec - self.last_real_state_time < 1.0 / self.real_state_rate:
            return

        self.last_real_state_time = now_sec

        if self.dry_run or self.robot is None:
            positions = self.current_positions
        else:
            positions = self.read_current_positions_from_robot_safe()

        if positions is None:
            return

        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = self.ros_joint_names
        msg.position = positions
        self.real_joint_state_pub.publish(msg)

    def read_current_positions_from_robot_safe(self, retries=1, delay_sec=0.0):
        last_error = None

        for attempt in range(retries):
            try:
                return self.read_current_positions_from_robot()
            except Exception as exc:
                last_error = exc

                if delay_sec > 0.0 and attempt + 1 < retries:
                    time.sleep(delay_sec)

        self.warn_throttled(
            f"Could not read real joint positions: {last_error}",
            1.0,
        )
        return None

    def read_current_positions_from_robot(self):
        obs = self.robot.get_observation()

        real_degs = [
            obs["shoulder_pan.pos"],
            obs["shoulder_lift.pos"],
            obs["elbow_flex.pos"],
            obs["wrist_flex.pos"],
        ]

        ros_positions = []

        for i in range(4):
            ros_deg = (real_degs[i] - self.real_offsets_deg[i]) / self.real_signs[i]
            ros_positions.append(math.radians(ros_deg))

        self.gripper_percent = max(0.0, min(100.0, obs["gripper.pos"]))

        ros_positions.append(math.radians(self.wrist_roll_ros_deg))
        ros_positions.append(self.gripper_percent_to_rad(self.gripper_percent))

        return ros_positions

    def gripper_percent_to_rad(self, percent):
        percent = max(0.0, min(100.0, percent))
        return self.gripper_min_rad + (percent / 100.0) * (
            self.gripper_max_rad - self.gripper_min_rad
        )

    def gripper_rad_to_percent(self, rad):
        rad = max(self.gripper_min_rad, min(self.gripper_max_rad, rad))

        percent = 100.0 * (
            (rad - self.gripper_min_rad)
            / (self.gripper_max_rad - self.gripper_min_rad)
        )

        return max(0.0, min(100.0, percent))

    def ros_positions_to_lerobot_action(self, positions):
        body_deg = []

        for i in range(4):
            ros_deg = math.degrees(positions[i])
            real_deg = self.real_signs[i] * ros_deg + self.real_offsets_deg[i]
            body_deg.append(real_deg)

        action = {
            "shoulder_pan.pos": body_deg[0],
            "shoulder_lift.pos": body_deg[1],
            "elbow_flex.pos": body_deg[2],
            "wrist_flex.pos": body_deg[3],
            "wrist_roll.pos": self.wrist_roll_real_deg,
            "gripper.pos": self.gripper_rad_to_percent(positions[5]),
        }

        return action

    def action_changed_enough(self, action):
        if self.last_action is None:
            return True

        for key, value in action.items():
            if abs(value - self.last_action[key]) > self.action_deadband_deg:
                return True

        return False

    def destroy_node(self):
        if self.robot is not None:
            try:
                self.robot.disconnect()
            except Exception as e:
                self.get_logger().warn(f"Error disconnecting robot: {e}")

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = MotorDriver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
