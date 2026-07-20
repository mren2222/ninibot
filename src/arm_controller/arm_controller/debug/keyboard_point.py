# flake8: noqa
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import sys
import select
import termios
import tty


class KeyboardPoint(Node):
    def __init__(self):
        super().__init__('keyboard_point')

        self.pub = self.create_publisher(Point, 'target_position', 10)

        self.x = 0.12
        self.y = 0.0
        self.z = 0.05

        self.step = 0.01

        self.get_logger().info("Keyboard control started")
        self.get_logger().info("W/S: x+/x-  A/D: y+/y-  Q/E: z+/z-  X: exit")

        # ✅ 只在初始化时设置一次 raw mode
        self.settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        self.timer = self.create_timer(0.05, self.loop)

    def get_key(self):
        # 非阻塞读，timeout=0
        rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
        if rlist:
            return sys.stdin.read(1)
        return None

    def loop(self):
        key = self.get_key()

        if key is None:
            return

        if key == 'w':
            self.x += self.step
        elif key == 's':
            self.x -= self.step
        elif key == 'a':
            self.y += self.step
        elif key == 'd':
            self.y -= self.step
        elif key == 'q':
            self.z += self.step
        elif key == 'e':
            self.z -= self.step
        elif key == 'x':
            self.get_logger().info("Exit")
            self.restore_terminal()
            rclpy.shutdown()
            return
        elif key == '\x03':  # Ctrl+C
            self.restore_terminal()
            rclpy.shutdown()
            return
        else:
            return

        msg = Point()
        msg.x = self.x
        msg.y = self.y
        msg.z = self.z

        self.pub.publish(msg)

        self.get_logger().info(
            f"Target: x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}"
        )

    def restore_terminal(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

    def destroy_node(self):
        # ✅ 无论怎么退出都还原终端，防止终端变成乱码状态
        self.restore_terminal()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardPoint()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()