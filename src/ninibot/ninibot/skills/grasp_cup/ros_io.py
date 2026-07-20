import cv2

from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import String


class GraspCupRosIO:
    def __init__(self, node, params):
        self.node = node
        self.params = params
        self.cap = None
        self.camera_ok = False
        self.status_pub = node.create_publisher(String, "/cup_servo_status", 10)
        self.debug_pub = node.create_publisher(String, "/cup_servo_debug", 10)
        self.debug_image_pub = node.create_publisher(
            Image,
            "/cup_servo/debug_image",
            10,
        )
        self.joint_target_pub = node.create_publisher(
            JointState,
            params.target_topic,
            10,
        )

    def open_camera(self):
        if self.cap is not None and self.cap.isOpened():
            self.camera_ok = True
            return True

        self.cap = cv2.VideoCapture(self.params.camera_source, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.params.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.params.frame_height)

        if not self.cap.isOpened():
            self.node.get_logger().error(
                f"Cannot open camera source {self.params.camera_source}"
            )
            self.camera_ok = False
            return False

        self.camera_ok = True
        return True

    def release_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.camera_ok = False
            self.node.get_logger().info("grasp_cup_node released camera")

    def read_frame(self):
        if self.cap is None:
            return False, None

        return self.cap.read()

    def publish_status(self, status):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    def publish_debug(self, text):
        msg = String()
        msg.data = text
        self.debug_pub.publish(msg)

    def image_to_msg(self, image, encoding):
        msg = Image()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.height = image.shape[0]
        msg.width = image.shape[1]
        msg.encoding = encoding
        msg.is_bigendian = False
        msg.step = image.strides[0]
        msg.data = image.tobytes()
        return msg

    def publish_debug_image(self, frame):
        if not self.params.publish_debug_images:
            return

        self.debug_image_pub.publish(self.image_to_msg(frame, "bgr8"))

    def publish_target_positions(self, joint_names, positions):
        joint_msg = JointState()
        joint_msg.header.stamp = self.node.get_clock().now().to_msg()
        joint_msg.name = joint_names
        joint_msg.position = positions
        self.joint_target_pub.publish(joint_msg)
