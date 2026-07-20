# flake8: noqa
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class TargetToRviz(Node):
    def __init__(self):
        super().__init__("target_to_rviz")

        self.subscriber = self.create_subscription(
            JointState,
            "/target_joint_states",
            self.callback,
            10,
        )

        self.publisher = self.create_publisher(
            JointState,
            "/joint_states",
            10,
        )

        self.get_logger().info(
            "Relaying /target_joint_states -> /joint_states"
        )

    def callback(self, msg):
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TargetToRviz()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()