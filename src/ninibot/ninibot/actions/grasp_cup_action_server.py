# flake8: noqa
import math
import re
import threading
import time

import rclpy
from rclpy.action import ActionServer
from rclpy.action import CancelResponse
from rclpy.action import GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from ninibot_interfaces.action import GraspCup


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"


class GraspCupActionServer(Node):
    def __init__(self):
        super().__init__("grasp_cup_action_server")

        self.declare_parameter("action_name", "/grasp_cup")
        self.declare_parameter("cup_status_topic", "/cup_servo_status")
        self.declare_parameter("cup_debug_topic", "/cup_servo_debug")
        self.declare_parameter("cup_command_topic", "/cup_servo_command")
        self.declare_parameter("control_cup_vision", True)
        self.declare_parameter("default_timeout_sec", 60.0)
        self.declare_parameter("feedback_period_sec", 0.2)

        self.action_name = self.get_parameter("action_name").value
        self.cup_status_topic = self.get_parameter("cup_status_topic").value
        self.cup_debug_topic = self.get_parameter("cup_debug_topic").value
        self.cup_command_topic = self.get_parameter("cup_command_topic").value
        self.control_cup_vision = bool(
            self.get_parameter("control_cup_vision").value
        )
        self.default_timeout_sec = float(
            self.get_parameter("default_timeout_sec").value
        )
        self.feedback_period_sec = float(
            self.get_parameter("feedback_period_sec").value
        )

        self.success_statuses = {"FINISHED"}
        self.failure_statuses = {"CAMERA_ERROR", "YOLO_ERROR", "FAILED", "CANCELED"}

        self.callback_group = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.active_goal = None
        self.latest_status = ""
        self.latest_debug = ""

        self.status_sub = self.create_subscription(
            String,
            self.cup_status_topic,
            self.status_callback,
            10,
            callback_group=self.callback_group,
        )
        self.debug_sub = self.create_subscription(
            String,
            self.cup_debug_topic,
            self.debug_callback,
            10,
            callback_group=self.callback_group,
        )
        self.command_pub = self.create_publisher(
            String,
            self.cup_command_topic,
            10,
        )
        self.action_server = ActionServer(
            self,
            GraspCup,
            self.action_name,
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )

        self.get_logger().info(
            f"grasp cup action server started: action={self.action_name}, "
            f"status_topic={self.cup_status_topic}, debug_topic={self.cup_debug_topic}, "
            f"command_topic={self.cup_command_topic}, control={self.control_cup_vision}"
        )

    def status_callback(self, msg):
        self.latest_status = msg.data.strip()

    def debug_callback(self, msg):
        self.latest_debug = msg.data.strip()

    def goal_callback(self, goal_request):
        del goal_request
        with self.lock:
            if self.active_goal is not None and self.active_goal.is_active:
                self.get_logger().warn("rejecting grasp cup goal: another goal is active")
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        del goal_handle
        self.get_logger().info(
            "accepted grasp cup action cancel request; "
            "note this wrapper does not stop grasp_cup_node motion"
        )
        return CancelResponse.ACCEPT

    def parse_debug_value(self, name):
        match = re.search(rf"{re.escape(name)}=({FLOAT_RE})", self.latest_debug)
        if not match:
            return math.nan
        return float(match.group(1))

    def make_feedback(self, elapsed_sec):
        feedback = GraspCup.Feedback()
        feedback.status = self.latest_status or "WAITING_FOR_STATUS"
        feedback.debug = self.latest_debug
        feedback.elapsed_sec = float(elapsed_sec)
        feedback.error_x = self.parse_debug_value("error_x")
        feedback.bbox_w = self.parse_debug_value("bbox_w")
        feedback.bbox_error = self.parse_debug_value("bbox_error")
        return feedback

    def make_result(self, success, final_status, failure_reason, elapsed_sec):
        result = GraspCup.Result()
        result.success = bool(success)
        result.final_status = final_status
        result.failure_reason = failure_reason
        result.elapsed_sec = float(elapsed_sec)
        return result

    def resolve_timeout(self, request_timeout_sec):
        timeout_sec = float(request_timeout_sec)
        if timeout_sec <= 0.0:
            timeout_sec = self.default_timeout_sec
        return timeout_sec

    def send_command(self, command):
        if not self.control_cup_vision:
            return

        msg = String()
        msg.data = command
        self.command_pub.publish(msg)

    def execute_callback(self, goal_handle):
        with self.lock:
            self.active_goal = goal_handle

        start_time = time.monotonic()
        timeout_sec = self.resolve_timeout(goal_handle.request.timeout_sec)
        initial_status = self.latest_status
        active_statuses = {
            "SEARCH_PREP",
            "SEARCHING",
            "TRACKING_CENTER",
            "FINE_ALIGN_PREP",
            "FINE_ALIGN",
            "LOST",
            "STARTING",
        }
        seen_active_status = False
        self.get_logger().info(f"grasp cup goal accepted; timeout={timeout_sec:.1f}s")

        if self.control_cup_vision:
            self.send_command("reset")
            time.sleep(0.05)
            self.send_command("start")

        try:
            while rclpy.ok():
                elapsed_sec = time.monotonic() - start_time
                status = self.latest_status
                if status in active_statuses:
                    seen_active_status = True

                feedback = self.make_feedback(elapsed_sec)
                goal_handle.publish_feedback(feedback)

                if goal_handle.is_cancel_requested:
                    self.send_command("cancel")
                    goal_handle.canceled()
                    return self.make_result(
                        False,
                        status,
                        "goal canceled",
                        elapsed_sec,
                    )

                if status in self.success_statuses and (
                    seen_active_status
                    or (
                        not self.control_cup_vision
                        and goal_handle.request.succeed_on_existing_finished
                    )
                ):
                    goal_handle.succeed()
                    return self.make_result(True, status, "", elapsed_sec)

                if status in self.failure_statuses and (
                    seen_active_status or status != initial_status or elapsed_sec > 1.0
                ):
                    goal_handle.abort()
                    return self.make_result(
                        False,
                        status,
                        f"cup vision reported {status}",
                        elapsed_sec,
                    )

                if elapsed_sec >= timeout_sec:
                    goal_handle.abort()
                    return self.make_result(
                        False,
                        status or "NO_STATUS",
                        f"timed out after {timeout_sec:.1f}s",
                        elapsed_sec,
                    )

                time.sleep(self.feedback_period_sec)

            goal_handle.abort()
            return self.make_result(False, self.latest_status, "rclpy shutdown", 0.0)
        finally:
            with self.lock:
                if self.active_goal is goal_handle:
                    self.active_goal = None


def main(args=None):
    rclpy.init(args=args)
    node = GraspCupActionServer()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
