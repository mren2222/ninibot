# flake8: noqa
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig


class ReadArm(Node):
    def __init__(self):
        super().__init__("read_arm")

        self.declare_parameter("port", "/dev/ttyACM0")
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("disable_torque", True)
        self.declare_parameter("publish_topic", "/joint_states")
        self.declare_parameter("real_state_topic", "/real_joint_states")

        self.port = self.get_parameter("port").value
        self.publish_rate = self.get_parameter("publish_rate").value
        self.disable_torque = self.get_parameter("disable_torque").value
        self.publish_topic = self.get_parameter("publish_topic").value
        self.real_state_topic = self.get_parameter("real_state_topic").value

        self.get_logger().info(f"Connecting to SO101 on {self.port}")

        config = SOFollowerRobotConfig(port=self.port)
        self.robot = SOFollower(config)

        # Use existing calibration file. Do not enter calibration every time.
        self.robot.connect(calibrate=False)

        self.get_logger().info("Connected to SO101")

        if self.disable_torque:
            self.get_logger().info("Disabling torque for manual movement")
            self.robot.bus.disable_torque()
            self.get_logger().info("Torque disabled")

        self.publisher = self.create_publisher(JointState, self.publish_topic, 10)
        self.real_state_pub = self.create_publisher(
            JointState,
            self.real_state_topic,
            10,
        )
        self.logged_first_state = False

        # LeRobot observation keys from robot.get_observation()
        self.lerobot_joint_keys = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

        # URDF joint names.
        # Your SO101 URDF uses numeric joint names.
        self.ros_joint_names = ["1", "2", "3", "4", "5", "6"]

        # Joint order:
        # 0 shoulder_pan
        # 1 shoulder_lift
        # 2 elbow_flex
        # 3 wrist_flex
        # 4 wrist_roll
        # 5 gripper

        # Direction correction.
        # If a joint moves opposite in RViz, change 1 to -1.
        self.signs = [1, 1, 1, 1, 1, 1]

        self.offsets_deg = [0, 0, 0, 0, 0, 0]

        # Gripper mapping.
        # LeRobot gripper value: 0-100 percent.
        # URDF joint 6 limit:
        # lower = -0.174533 rad
        # upper =  1.74533 rad
        self.gripper_min_rad = -0.174533
        self.gripper_max_rad = 1.74533

        # If RViz gripper opens when the real gripper closes, set this to True.
        self.gripper_reverse = False

        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.publish_joint_states)

    def convert_to_ros_position(self, index, value):
        """
        Convert LeRobot observation value to ROS JointState position.

        Body joints:
            LeRobot gives degrees.
            ROS JointState expects radians.

        Gripper:
            LeRobot gives 0-100 percent.
            URDF joint 6 expects radians.
        """
        joint_key = self.lerobot_joint_keys[index]

        if joint_key == "gripper.pos":
            gripper_percent = value

            # Clamp to 0-100
            gripper_percent = max(0.0, min(100.0, gripper_percent))

            # Reverse if RViz open/close direction is opposite
            if self.gripper_reverse:
                gripper_percent = 100.0 - gripper_percent

            gripper_range = self.gripper_max_rad - self.gripper_min_rad
            gripper_rad = self.gripper_min_rad + (gripper_percent / 100.0) * gripper_range

            return gripper_rad

        corrected_deg = self.signs[index] * value + self.offsets_deg[index]
        return math.radians(corrected_deg)

    def publish_joint_states(self):
        try:
            obs = self.robot.get_observation()

            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.ros_joint_names

            positions = []

            for i, key in enumerate(self.lerobot_joint_keys):
                value = obs[key]
                ros_position = self.convert_to_ros_position(i, value)
                positions.append(ros_position)

            msg.position = positions
            self.publisher.publish(msg)
            self.real_state_pub.publish(msg)

            if not self.logged_first_state:
                degs = [math.degrees(value) for value in positions[:5]]
                self.get_logger().info(
                    "Publishing real joint states to "
                    f"{self.publish_topic} and {self.real_state_topic}: "
                    f"j1-5_deg=[{degs[0]:.1f}, {degs[1]:.1f}, "
                    f"{degs[2]:.1f}, {degs[3]:.1f}, {degs[4]:.1f}]"
                )
                self.logged_first_state = True

        except Exception as e:
            self.get_logger().error(f"Failed to publish joint states: {e}")

    def destroy_node(self):
        if rclpy.ok():
            self.get_logger().info("Disconnecting SO101")

        try:
            self.robot.disconnect()
        except Exception as e:
            if rclpy.ok():
                self.get_logger().warn(f"Error while disconnecting robot: {e}")

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        node = ReadArm()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
