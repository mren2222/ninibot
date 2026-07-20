# flake8: noqa
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point


class PointIK(Node):
    def __init__(self):
        super().__init__("point_ik")

        self.declare_parameter("publish_rate", 100.0)
        self.publish_rate = self.get_parameter("publish_rate").value

        self.publisher = self.create_publisher(JointState, "/target_joint_states", 100)

        self.target_subscriber = self.create_subscription(
            Point,
            "/target_position",
            self.target_position_callback,
            100,
        )

        self.joint_names = ["1", "2", "3", "4"]

        # Link lengths, meters
        self.l1 = 0.1159
        self.l2 = 0.1350

        # Target means gripper/cup target, not wrist target
        self.target_x = 0.1629
        self.target_y = 0.0
        self.target_z = 0.1131

        # Tool offset: gripper target -> wrist target
        self.horizontal_tool_offset = 0.10
        self.z_tool_offset = -0.10

        # Joint safety limits, degrees
        self.base_min_deg = -90.0
        self.base_max_deg = 90.0

        self.shoulder_min_deg = -100.0
        self.shoulder_max_deg = 100.0

        self.elbow_min_deg = -90.0
        self.elbow_max_deg = 90.0

        self.wrist_flex_min_deg = -80.0
        self.wrist_flex_max_deg = 80.0

        self.wrist_pitch_offset_deg = 0.0

        self.current_positions = None

        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.publish_joint_states)

        self.get_logger().info("point_ik started; waiting for /target_position")

    def target_position_callback(self, msg):
        result = self.compute_joint_positions(msg.x, msg.y, msg.z)

        if result is None:
            self.get_logger().warn(
                f"Rejected target: x={msg.x:.3f}, y={msg.y:.3f}, z={msg.z:.3f}"
            )
            return

        self.target_x = msg.x
        self.target_y = msg.y
        self.target_z = msg.z
        self.current_positions = result

        self.get_logger().info(
            f"Accepted target: x={msg.x:.3f}, y={msg.y:.3f}, z={msg.z:.3f}"
        )

    def clamp(self, value, lower, upper):
        return max(lower, min(upper, value))

    def in_range(self, value, lower, upper):
        return lower <= value <= upper

    def apply_target_offset(self, x, y, z):
        r = math.sqrt(x * x + y * y)

        if r < 1e-6:
            dir_x = 1.0
            dir_y = 0.0
        else:
            dir_x = x / r
            dir_y = y / r

        wrist_x = x - self.horizontal_tool_offset * dir_x
        wrist_y = y - self.horizontal_tool_offset * dir_y
        wrist_z = z + self.z_tool_offset

        return wrist_x, wrist_y, wrist_z

    def inverse_kinematics_2link(self, planar_x, planar_z):
        theta1_offset = math.atan2(0.028, 0.11257)
        theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset

        r = math.sqrt(planar_x ** 2 + planar_z ** 2)
        r_max = self.l1 + self.l2
        r_min = abs(self.l1 - self.l2)

        if r > r_max or r < r_min:
            return None

        cos_theta2 = -(r ** 2 - self.l1 ** 2 - self.l2 ** 2) / (
            2.0 * self.l1 * self.l2
        )
        cos_theta2 = self.clamp(cos_theta2, -1.0, 1.0)

        theta2 = math.pi - math.acos(cos_theta2)

        beta = math.atan2(planar_z, planar_x)
        gamma = math.atan2(
            self.l2 * math.sin(theta2),
            self.l1 + self.l2 * math.cos(theta2),
        )

        theta1 = beta + gamma

        joint2 = theta1 + theta1_offset
        joint3 = theta2 + theta2_offset

        joint2_deg = 90.0 - math.degrees(joint2)
        joint3_deg = math.degrees(joint3) - 90.0

        return joint2_deg, joint3_deg

    def compute_joint_positions(self, x, y, z):
        wrist_x, wrist_y, wrist_z = self.apply_target_offset(x, y, z)

        base_deg = -math.degrees(math.atan2(wrist_y, wrist_x))

        if not self.in_range(base_deg, self.base_min_deg, self.base_max_deg):
            self.get_logger().warn(f"Base out of range: {base_deg:.1f} deg")
            return None

        planar_x = math.sqrt(wrist_x * wrist_x + wrist_y * wrist_y)
        planar_z = wrist_z

        ik_result = self.inverse_kinematics_2link(planar_x, planar_z)

        if ik_result is None:
            self.get_logger().warn("Target outside 2-link reachable workspace")
            return None

        shoulder_deg, elbow_deg = ik_result

        wrist_flex_deg = (
            -shoulder_deg
            -elbow_deg
            + self.wrist_pitch_offset_deg
        )

        if not self.in_range(shoulder_deg, self.shoulder_min_deg, self.shoulder_max_deg):
            self.get_logger().warn(f"Shoulder out of range: {shoulder_deg:.1f} deg")
            return None

        if not self.in_range(elbow_deg, self.elbow_min_deg, self.elbow_max_deg):
            self.get_logger().warn(f"Elbow out of range: {elbow_deg:.1f} deg")
            return None

        if not self.in_range(
            wrist_flex_deg,
            self.wrist_flex_min_deg,
            self.wrist_flex_max_deg,
        ):
            self.get_logger().warn(
                f"Horizontal wrist impossible: wrist_flex={wrist_flex_deg:.1f} deg"
            )
            return None

        positions = [
            math.radians(base_deg),
            math.radians(shoulder_deg),
            math.radians(elbow_deg),
            math.radians(wrist_flex_deg),
        ]

        return positions

    def publish_joint_states(self):
        if self.current_positions is None:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.current_positions

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PointIK()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
