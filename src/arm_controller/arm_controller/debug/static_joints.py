# flake8: noqa
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class StaticJoints(Node):
    def __init__(self):
        super().__init__("static_joints")

        self.declare_parameter("publish_topic", "/joint_states")
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("joint1_deg", 0.0)
        self.declare_parameter("joint2_deg", -90.0)
        self.declare_parameter("joint3_deg", 98.0)
        self.declare_parameter("joint4_deg", 40.0)
        self.declare_parameter("joint5_deg", -90.0)
        self.declare_parameter("joint6_rad", 0.0)

        self.publish_topic = self.get_parameter("publish_topic").value
        publish_rate = float(self.get_parameter("publish_rate").value)
        self.joint_names = ["1", "2", "3", "4", "5", "6"]
        self.positions = [
            math.radians(float(self.get_parameter("joint1_deg").value)),
            math.radians(float(self.get_parameter("joint2_deg").value)),
            math.radians(float(self.get_parameter("joint3_deg").value)),
            math.radians(float(self.get_parameter("joint4_deg").value)),
            math.radians(float(self.get_parameter("joint5_deg").value)),
            float(self.get_parameter("joint6_rad").value),
        ]

        self.publisher = self.create_publisher(JointState, self.publish_topic, 10)
        self.timer = self.create_timer(1.0 / max(1e-6, publish_rate), self.publish)

        self.get_logger().info(
            "static_joints started; "
            f"topic={self.publish_topic}, joints={self.joint_names}"
        )

    def publish(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.positions
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StaticJoints()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
