# flake8: noqa
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ShutdownOnTaskStatus(Node):
    def __init__(self):
        super().__init__("shutdown_on_task_status")

        self.declare_parameter("status_topic", "/task_manager_status")
        self.declare_parameter("terminal_statuses", "DONE,FAILED,CANCELED")
        self.declare_parameter("delay_sec", 2.0)

        self.status_topic = self.get_parameter("status_topic").value
        self.terminal_statuses = {
            value.strip()
            for value in str(self.get_parameter("terminal_statuses").value).split(",")
            if value.strip()
        }
        self.delay_sec = max(0.0, float(self.get_parameter("delay_sec").value))
        self.shutdown_timer = None
        self.shutdown_status = ""

        self.status_sub = self.create_subscription(
            String,
            self.status_topic,
            self.status_callback,
            10,
        )
        self.get_logger().info(
            f"shutdown_on_task_status watching {self.status_topic}; "
            f"terminal_statuses={sorted(self.terminal_statuses)}, "
            f"delay_sec={self.delay_sec:.1f}"
        )

    def status_callback(self, msg):
        status = msg.data.strip()

        if self.shutdown_timer is not None:
            return

        if status not in self.terminal_statuses:
            return

        self.shutdown_status = status
        self.get_logger().info(
            f"terminal task status {status}; shutting down launch in "
            f"{self.delay_sec:.1f}s"
        )
        self.shutdown_timer = self.create_timer(
            self.delay_sec,
            self.shutdown_now,
        )

    def shutdown_now(self):
        if self.shutdown_timer is not None:
            self.shutdown_timer.cancel()

        self.get_logger().info(
            f"shutdown_on_task_status exiting after {self.shutdown_status}"
        )
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ShutdownOnTaskStatus()

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
