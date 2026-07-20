# flake8: noqa
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from ninibot_interfaces.action import GraspCup
from ninibot_interfaces.action import MoveCup

from ninibot.task_params import TASK_MANAGER_PARAMS


class FullDemoManager(Node):
    def __init__(self):
        super().__init__("full_demo_manager")

        self.declare_parameters(
            namespace="",
            parameters=list(TASK_MANAGER_PARAMS.items()),
        )

        self.move_cup_command_topic = self.get_parameter("move_cup_command_topic").value
        self.move_cup_status_topic = self.get_parameter("move_cup_status_topic").value
        self.scan_move_target_before_grasp = bool(
            self.get_parameter("scan_move_target_before_grasp").value
        )
        self.move_cup_scan_command = self.get_parameter("move_cup_scan_command").value
        self.grasp_cup_action_name = self.get_parameter("grasp_cup_action_name").value
        self.move_cup_action_name = self.get_parameter("move_cup_action_name").value
        self.grasp_timeout_sec = float(self.get_parameter("grasp_timeout_sec").value)
        self.move_timeout_sec = float(self.get_parameter("move_timeout_sec").value)
        self.press_button_command_topic = self.get_parameter(
            "press_button_command_topic"
        ).value
        self.press_button_status_topic = self.get_parameter(
            "press_button_status_topic"
        ).value
        self.task_status_topic = self.get_parameter("task_status_topic").value
        self.auto_start_grasp = bool(self.get_parameter("auto_start_grasp").value)
        self.auto_start_move = bool(self.get_parameter("auto_start_move").value)
        self.auto_start_press_button = bool(
            self.get_parameter("auto_start_press_button").value
        )

        self.move_cup_command_pub = self.create_publisher(
            String,
            self.move_cup_command_topic,
            10,
        )
        self.press_button_command_pub = self.create_publisher(
            String,
            self.press_button_command_topic,
            10,
        )
        self.task_status_pub = self.create_publisher(
            String,
            self.task_status_topic,
            10,
        )
        self.move_cup_status_sub = self.create_subscription(
            String,
            self.move_cup_status_topic,
            self.move_cup_status_callback,
            10,
        )
        self.press_button_status_sub = self.create_subscription(
            String,
            self.press_button_status_topic,
            self.press_button_status_callback,
            10,
        )

        self.grasp_client = ActionClient(
            self,
            GraspCup,
            self.grasp_cup_action_name,
        )
        self.move_client = ActionClient(
            self,
            MoveCup,
            self.move_cup_action_name,
        )

        self.state = (
            "WAITING_FOR_TARGET_SCAN"
            if self.scan_move_target_before_grasp
            else "WAITING_FOR_GRASP"
        )
        self.target_scan_started = False
        self.target_scan_active = False
        self.last_target_scan_command_time = 0.0
        self.target_locked = False
        self.grasp_started = False
        self.move_started = False
        self.press_button_started = False
        self.last_grasp_feedback_status = ""
        self.last_move_feedback_status = ""

        self.startup_timer = self.create_timer(0.5, self.startup_tick)

        self.publish_task_status()
        self.get_logger().info(
            f"full_demo_manager started; auto_start_grasp={self.auto_start_grasp}, "
            f"scan_move_target_before_grasp={self.scan_move_target_before_grasp}"
        )

    def publish_task_status(self):
        msg = String()
        msg.data = self.state
        self.task_status_pub.publish(msg)

    def set_state(self, state):
        if self.state == state:
            return

        self.state = state
        self.publish_task_status()
        self.get_logger().info(f"task state -> {self.state}")

    def send_move_command(self, command):
        msg = String()
        msg.data = command
        self.move_cup_command_pub.publish(msg)

    def send_move_scan_topic(self):
        self.send_move_command(self.move_cup_scan_command)

    def send_press_button_start(self):
        msg = String()
        msg.data = "start"
        self.press_button_command_pub.publish(msg)

    def startup_tick(self):
        if not self.auto_start_grasp or self.grasp_started:
            return

        if self.state == "FAILED":
            return

        if self.scan_move_target_before_grasp:
            if not self.target_scan_started:
                self.start_target_scan()
                return
            if not self.target_locked:
                if (
                    not self.target_scan_active
                    and time.monotonic() - self.last_target_scan_command_time > 2.0
                ):
                    self.send_move_scan_topic()
                    self.last_target_scan_command_time = time.monotonic()
                    self.get_logger().info(
                        f"resent MoveCup target scan command: {self.move_cup_scan_command}"
                    )
                return

        if not self.grasp_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().info("waiting for GraspCup action server")
            return

        self.start_grasp_action()

    def start_target_scan(self):
        self.target_scan_started = True
        self.target_scan_active = False
        self.target_locked = False
        self.set_state("SCANNING_MOVE_TARGET")
        self.send_move_scan_topic()
        self.last_target_scan_command_time = time.monotonic()
        self.get_logger().info(
            f"sent MoveCup target scan command: {self.move_cup_scan_command}"
        )

    def start_grasp_action(self):
        if self.scan_move_target_before_grasp and not self.target_locked:
            self.get_logger().info("waiting for locked move-cup target before grasp")
            return

        self.grasp_started = True
        self.last_grasp_feedback_status = ""
        self.set_state("GRASPING_CUP")

        goal = GraspCup.Goal()
        goal.timeout_sec = float(self.grasp_timeout_sec)
        goal.succeed_on_existing_finished = False

        future = self.grasp_client.send_goal_async(
            goal,
            feedback_callback=self.grasp_feedback_callback,
        )
        future.add_done_callback(self.grasp_goal_response_callback)
        self.get_logger().info("sent GraspCup action goal")

    def grasp_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.set_state("FAILED")
            self.get_logger().warn("GraspCup action goal rejected")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.grasp_result_callback)

    def grasp_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        status = feedback.status.strip()
        if status == self.last_grasp_feedback_status:
            return

        self.last_grasp_feedback_status = status
        self.get_logger().info(f"grasp feedback: {status}")

    def grasp_result_callback(self, future):
        result = future.result().result
        if not result.success:
            self.set_state("FAILED")
            self.get_logger().warn(
                f"GraspCup failed: status={result.final_status}, "
                f"reason={result.failure_reason}"
            )
            return

        self.get_logger().info(
            f"GraspCup done in {result.elapsed_sec:.2f}s; status={result.final_status}"
        )

        if self.auto_start_move:
            self.start_move_action()
        else:
            self.set_state("GRASP_DONE")

    def start_move_action(self):
        self.move_started = True
        self.last_move_feedback_status = ""
        self.set_state("MOVING_CUP")

        if not self.move_client.wait_for_server(timeout_sec=0.0):
            self.set_state("FAILED")
            self.get_logger().warn("MoveCup action server is not ready")
            return

        goal = MoveCup.Goal()
        goal.timeout_sec = float(self.move_timeout_sec)
        goal.succeed_on_existing_done = False

        future = self.move_client.send_goal_async(
            goal,
            feedback_callback=self.move_feedback_callback,
        )
        future.add_done_callback(self.move_goal_response_callback)
        self.get_logger().info("sent MoveCup action goal")

    def move_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.set_state("FAILED")
            self.get_logger().warn("MoveCup action goal rejected")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.move_result_callback)

    def move_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        status = feedback.status.strip()
        if status == self.last_move_feedback_status:
            return

        self.last_move_feedback_status = status
        self.get_logger().info(f"move feedback: {status}")

    def move_result_callback(self, future):
        result = future.result().result
        if not result.success:
            self.set_state("FAILED")
            self.get_logger().warn(
                f"MoveCup failed: status={result.final_status}, "
                f"reason={result.failure_reason}"
            )
            return

        self.get_logger().info(
            f"MoveCup done in {result.elapsed_sec:.2f}s; status={result.final_status}"
        )

        if self.auto_start_press_button:
            self.press_button_started = True
            self.set_state("PRESSING_BUTTON")
            self.send_press_button_start()
            self.get_logger().info("move cup done; starting press button")
        else:
            self.set_state("DONE")
            self.get_logger().info("task done")

    def move_cup_status_callback(self, msg):
        status = msg.data.strip()

        if (
            self.scan_move_target_before_grasp
            and self.state == "SCANNING_MOVE_TARGET"
        ):
            if status in ("SEARCH_TARGET_PREP", "SEARCH_TARGET"):
                self.target_scan_active = True
            elif status == "TARGET_LOCKED":
                self.target_locked = True
                self.target_scan_active = False
                self.set_state("WAITING_FOR_GRASP")
                self.get_logger().info("move-cup ArUco target locked; starting grasp next")
            elif status in ("FAILED", "CANCELED"):
                self.target_scan_active = False
                self.set_state("FAILED")
                self.get_logger().warn(f"move-cup target scan failed: {status}")
            return

    def press_button_status_callback(self, msg):
        status = msg.data.strip()

        if self.state == "PRESSING_BUTTON" and status == "DONE":
            self.set_state("DONE")
            self.get_logger().info("task done")

        elif self.state == "PRESSING_BUTTON" and status == "FAILED":
            self.set_state("FAILED")
            self.get_logger().warn("press button failed")


def main(args=None):
    rclpy.init(args=args)
    node = FullDemoManager()

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
