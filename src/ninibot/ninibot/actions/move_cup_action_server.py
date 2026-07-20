# flake8: noqa
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
from ninibot_interfaces.action import MoveCup
from ninibot.skills.move_cup.state_machine import MOVE_ACTION_ACTIVE_STATUSES


class MoveCupActionServer(Node):
    def __init__(self):
        super().__init__("move_cup_action_server")

        self.declare_parameter("action_name", "/move_cup")
        self.declare_parameter("move_status_topic", "/move_cup_status")
        self.declare_parameter("move_command_topic", "/move_cup_command")
        self.declare_parameter("control_move_cup", True)
        self.declare_parameter("start_command", "start")
        self.declare_parameter("reset_before_start", True)
        self.declare_parameter("reset_command", "reset")
        self.declare_parameter("default_timeout_sec", 45.0)
        self.declare_parameter("feedback_period_sec", 0.2)

        self.action_name = self.get_parameter("action_name").value
        self.move_status_topic = self.get_parameter("move_status_topic").value
        self.move_command_topic = self.get_parameter("move_command_topic").value
        self.control_move_cup = bool(
            self.get_parameter("control_move_cup").value
        )
        self.start_command = self.get_parameter("start_command").value
        self.reset_before_start = bool(
            self.get_parameter("reset_before_start").value
        )
        self.reset_command = self.get_parameter("reset_command").value
        self.default_timeout_sec = float(
            self.get_parameter("default_timeout_sec").value
        )
        self.feedback_period_sec = float(
            self.get_parameter("feedback_period_sec").value
        )

        self.success_statuses = {"DONE"}
        self.failure_statuses = {"FAILED", "CANCELED"}

        self.callback_group = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.active_goal = None
        self.latest_status = ""

        self.status_sub = self.create_subscription(
            String,
            self.move_status_topic,
            self.status_callback,
            10,
            callback_group=self.callback_group,
        )
        self.command_pub = self.create_publisher(
            String,
            self.move_command_topic,
            10,
        )
        self.action_server = ActionServer(
            self,
            MoveCup,
            self.action_name,
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )

        self.get_logger().info(
            f"move cup action server started: action={self.action_name}, "
            f"status_topic={self.move_status_topic}, "
            f"command_topic={self.move_command_topic}, control={self.control_move_cup}, "
            f"start_command={self.start_command}, "
            f"reset_before_start={self.reset_before_start}"
        )

    def status_callback(self, msg):
        self.latest_status = msg.data.strip()

    def goal_callback(self, goal_request):
        del goal_request
        with self.lock:
            if self.active_goal is not None and self.active_goal.is_active:
                self.get_logger().warn("rejecting move cup goal: another goal is active")
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        del goal_handle
        self.get_logger().info("accepted move cup action cancel request")
        return CancelResponse.ACCEPT

    def make_feedback(self, elapsed_sec):
        feedback = MoveCup.Feedback()
        feedback.status = self.latest_status or "WAITING_FOR_STATUS"
        feedback.elapsed_sec = float(elapsed_sec)
        return feedback

    def make_result(self, success, final_status, failure_reason, elapsed_sec):
        result = MoveCup.Result()
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
        if not self.control_move_cup:
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
        active_statuses = MOVE_ACTION_ACTIVE_STATUSES
        seen_active_status = False
        self.get_logger().info(f"move cup goal accepted; timeout={timeout_sec:.1f}s")

        if self.control_move_cup:
            if self.reset_before_start:
                self.send_command(self.reset_command)
                time.sleep(0.05)
            self.send_command(self.start_command)

        try:
            while rclpy.ok():
                elapsed_sec = time.monotonic() - start_time
                status = self.latest_status
                if status in active_statuses:
                    seen_active_status = True

                goal_handle.publish_feedback(self.make_feedback(elapsed_sec))

                if goal_handle.is_cancel_requested:
                    self.send_command("cancel")
                    goal_handle.canceled()
                    return self.make_result(False, status, "goal canceled", elapsed_sec)

                if status in self.success_statuses and (
                    seen_active_status
                    or (
                        not self.control_move_cup
                        and goal_handle.request.succeed_on_existing_done
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
                        f"move cup reported {status}",
                        elapsed_sec,
                    )

                if elapsed_sec >= timeout_sec:
                    self.send_command("cancel")
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
    node = MoveCupActionServer()
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
