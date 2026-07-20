# flake8: noqa
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from ninibot_interfaces.action import GraspCup

from ninibot.task_params import FINE_ALIGN_TRAINING_LOOP_PARAMS


class FineAlignEpisodeCollector(Node):
    def __init__(self):
        super().__init__("fine_align_episode_collector")

        self.declare_parameters(
            namespace="",
            parameters=list(FINE_ALIGN_TRAINING_LOOP_PARAMS.items()),
        )

        self.command_topic = self.get_parameter("command_topic").value
        self.status_topic = self.get_parameter("status_topic").value
        self.task_status_topic = self.get_parameter("task_status_topic").value
        self.reset_command_topic = self.get_parameter("reset_command_topic").value
        self.reset_status_topic = self.get_parameter("reset_status_topic").value
        self.grasp_cup_action_name = self.get_parameter("grasp_cup_action_name").value
        self.grasp_timeout_sec = float(self.get_parameter("grasp_timeout_sec").value)
        self.reset_timeout_sec = float(self.get_parameter("reset_timeout_sec").value)
        self.auto_start = bool(self.get_parameter("auto_start").value)
        self.max_episodes = max(0, int(self.get_parameter("max_episodes").value))
        self.settle_sec = float(self.get_parameter("settle_sec").value)
        self.reset_start_delay_sec = float(
            self.get_parameter("reset_start_delay_sec").value
        )
        self.continue_on_grasp_failure = bool(
            self.get_parameter("continue_on_grasp_failure").value
        )
        self.continue_on_reset_failure = bool(
            self.get_parameter("continue_on_reset_failure").value
        )

        self.grasp_client = ActionClient(self, GraspCup, self.grasp_cup_action_name)

        self.command_sub = self.create_subscription(
            String,
            self.command_topic,
            self.command_callback,
            10,
        )
        self.reset_status_sub = self.create_subscription(
            String,
            self.reset_status_topic,
            self.reset_status_callback,
            10,
        )
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.task_status_pub = self.create_publisher(
            String,
            self.task_status_topic,
            10,
        )
        self.reset_command_pub = self.create_publisher(
            String,
            self.reset_command_topic,
            10,
        )

        self.state = "STARTING"
        self.running = self.auto_start
        self.started_once = False
        self.episode_index = 0
        self.active_goal_handle = None
        self.waiting_for_servers_logged = False
        self.latest_reset_status = ""
        self.reset_started_at = None
        self.next_action_time = None
        self.last_failure_reason = ""

        self.timer = self.create_timer(0.2, self.tick)
        self.set_state("WAITING_FOR_SERVERS")
        self.get_logger().info(
            f"fine_align_episode_collector started; auto_start={self.auto_start}, "
            f"max_episodes={self.max_episodes}, command_topic={self.command_topic}"
        )

    def command_callback(self, msg):
        command = msg.data.strip().lower()

        if command in ("start", "resume"):
            self.running = True
            if self.state in ("IDLE", "PAUSED", "FAILED", "STOPPED"):
                self.set_state("IDLE")
                self.start_next_episode_if_ready()
        elif command == "pause":
            self.running = False
            if self.state in ("IDLE", "WAITING_SETTLE"):
                self.set_state("PAUSED")
        elif command in ("stop", "cancel"):
            self.stop_loop()
        elif command == "next":
            if self.state in ("IDLE", "PAUSED", "WAITING_SETTLE"):
                self.running = True
                self.set_state("IDLE")
                self.start_next_episode_if_ready(force=True)
            else:
                self.get_logger().warn(f"cannot start next while state={self.state}")
        else:
            self.get_logger().warn(f"Unknown fine-align training command: {msg.data}")

    def tick(self):
        now = time.monotonic()

        if self.state == "WAITING_FOR_SERVERS":
            if not self.grasp_client.wait_for_server(timeout_sec=0.0):
                if not self.waiting_for_servers_logged:
                    self.get_logger().info("waiting for GraspCup action server")
                    self.waiting_for_servers_logged = True
                return

            self.waiting_for_servers_logged = False
            self.set_state("IDLE")

        if (
            self.state == "WAITING_BEFORE_RESET"
            and self.next_action_time is not None
            and now >= self.next_action_time
        ):
            self.start_reset()
            return

        if self.state == "RESETTING_CUP" and self.reset_started_at is not None:
            if now - self.reset_started_at >= self.reset_timeout_sec:
                self.handle_reset_failure(
                    f"random place timed out after {self.reset_timeout_sec:.1f}s"
                )
                return

        if (
            self.state == "WAITING_SETTLE"
            and self.next_action_time is not None
            and now >= self.next_action_time
        ):
            self.set_state("IDLE")

        if self.running and self.state == "IDLE":
            self.start_next_episode_if_ready()

    def start_next_episode_if_ready(self, force=False):
        del force
        if self.max_episodes > 0 and self.episode_index >= self.max_episodes:
            self.running = False
            self.publish_task_status("DONE")
            self.set_state("DONE")
            self.get_logger().info(
                f"fine-align training done after {self.episode_index} episodes"
            )
            return

        if not self.grasp_client.wait_for_server(timeout_sec=0.0):
            self.set_state("WAITING_FOR_SERVERS")
            return

        self.start_episode()

    def start_episode(self):
        self.started_once = True
        self.episode_index += 1
        self.latest_reset_status = ""
        self.reset_started_at = None
        self.next_action_time = None
        self.last_failure_reason = ""

        self.publish_task_status("GRASPING_CUP")
        self.set_state("GRASPING")
        self.get_logger().info(f"starting fine-align training episode {self.episode_index}")

        goal = GraspCup.Goal()
        goal.timeout_sec = float(self.grasp_timeout_sec)
        goal.succeed_on_existing_finished = False

        future = self.grasp_client.send_goal_async(
            goal,
            feedback_callback=self.grasp_feedback_callback,
        )
        future.add_done_callback(self.grasp_goal_response_callback)

    def grasp_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.handle_grasp_failure("GraspCup action goal rejected")
            return

        self.active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.grasp_result_callback)

    def grasp_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f"fine-align episode {self.episode_index} grasp feedback: "
            f"{feedback.status}, error_x={feedback.error_x:.1f}, bbox_w={feedback.bbox_w:.1f}"
        )

    def grasp_result_callback(self, future):
        self.active_goal_handle = None
        result = future.result().result
        if not result.success:
            reason = (
                f"GraspCup failed: status={result.final_status}, "
                f"reason={result.failure_reason}"
            )
            self.handle_grasp_failure(reason)
            return

        self.get_logger().info(
            f"fine-align episode {self.episode_index} reached FINISHED "
            f"in {result.elapsed_sec:.2f}s"
        )
        self.publish_task_status("DONE")
        self.set_state("WAITING_BEFORE_RESET")
        self.next_action_time = time.monotonic() + max(0.0, self.reset_start_delay_sec)

    def handle_grasp_failure(self, reason):
        self.last_failure_reason = reason
        self.publish_task_status("FAILED")
        self.get_logger().warn(reason)

        if self.continue_on_grasp_failure:
            self.set_state("WAITING_BEFORE_RESET")
            self.next_action_time = time.monotonic() + max(0.0, self.reset_start_delay_sec)
            return

        self.running = False
        self.set_state("FAILED")

    def start_reset(self):
        self.latest_reset_status = ""
        self.reset_started_at = time.monotonic()
        self.next_action_time = None
        self.publish_task_status("RESETTING_CUP")
        self.set_state("RESETTING_CUP")
        self.publish_reset_command("reset")
        self.publish_reset_command("start")
        self.get_logger().info(
            f"episode {self.episode_index} fine-align data done; randomizing cup"
        )

    def reset_status_callback(self, msg):
        status = msg.data.strip()
        self.latest_reset_status = status

        if self.state != "RESETTING_CUP":
            return

        if status == "DONE":
            self.get_logger().info(
                f"random place done for episode {self.episode_index}; settling"
            )
            self.set_state("WAITING_SETTLE")
            self.next_action_time = time.monotonic() + max(0.0, self.settle_sec)
            self.reset_started_at = None
        elif status in ("FAILED", "CANCELED"):
            self.handle_reset_failure(f"random place reported {status}")

    def handle_reset_failure(self, reason):
        self.last_failure_reason = reason
        self.publish_task_status("FAILED")
        self.get_logger().warn(reason)
        self.reset_started_at = None
        self.next_action_time = None

        if self.continue_on_reset_failure:
            self.set_state("WAITING_SETTLE")
            self.next_action_time = time.monotonic() + max(0.0, self.settle_sec)
            return

        self.running = False
        self.set_state("FAILED")

    def stop_loop(self):
        self.running = False
        if self.active_goal_handle is not None:
            self.active_goal_handle.cancel_goal_async()
            self.active_goal_handle = None
        self.publish_reset_command("cancel")
        self.publish_task_status("CANCELED")
        self.set_state("STOPPED")
        self.get_logger().info("fine-align training loop stopped")

    def publish_reset_command(self, command):
        msg = String()
        msg.data = command
        self.reset_command_pub.publish(msg)

    def publish_task_status(self, status):
        msg = String()
        msg.data = status
        self.task_status_pub.publish(msg)

    def set_state(self, state):
        self.state = state
        self.publish_status()

    def publish_status(self):
        msg = String()
        msg.data = (
            f"state={self.state}, "
            f"running={self.running}, "
            f"episode={self.episode_index}, "
            f"reset_status={self.latest_reset_status}, "
            f"last_failure={self.last_failure_reason}"
        )
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FineAlignEpisodeCollector()

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
