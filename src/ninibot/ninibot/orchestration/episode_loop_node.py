# flake8: noqa
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from ninibot_interfaces.action import GraspCup
from ninibot_interfaces.action import MoveCup


class EpisodeLoopNode(Node):
    def __init__(self):
        super().__init__("episode_loop_node")

        self.declare_parameter("command_topic", "/episode_loop/command")
        self.declare_parameter("status_topic", "/episode_loop/status")
        self.declare_parameter("task_status_topic", "/task_manager_status")
        self.declare_parameter("grasp_cup_action_name", "/grasp_cup")
        self.declare_parameter("move_cup_action_name", "/move_cup")
        self.declare_parameter("grasp_timeout_sec", 60.0)
        self.declare_parameter("move_timeout_sec", 45.0)
        self.declare_parameter("auto_start_first_episode", True)

        self.command_topic = self.get_parameter("command_topic").value
        self.status_topic = self.get_parameter("status_topic").value
        self.task_status_topic = self.get_parameter("task_status_topic").value
        self.grasp_cup_action_name = self.get_parameter("grasp_cup_action_name").value
        self.move_cup_action_name = self.get_parameter("move_cup_action_name").value
        self.grasp_timeout_sec = float(self.get_parameter("grasp_timeout_sec").value)
        self.move_timeout_sec = float(self.get_parameter("move_timeout_sec").value)
        self.auto_start_first_episode = bool(
            self.get_parameter("auto_start_first_episode").value
        )

        self.grasp_client = ActionClient(self, GraspCup, self.grasp_cup_action_name)
        self.move_client = ActionClient(self, MoveCup, self.move_cup_action_name)

        self.command_sub = self.create_subscription(
            String,
            self.command_topic,
            self.command_callback,
            10,
        )
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.task_status_pub = self.create_publisher(
            String,
            self.task_status_topic,
            10,
        )

        self.state = "STARTING"
        self.episode_index = 0
        self.active_goal_handle = None
        self.waiting_for_servers_logged = False
        self.first_episode_started = False

        self.timer = self.create_timer(0.5, self.tick)
        self.set_state("WAITING_FOR_SERVERS")
        self.get_logger().info(
            f"episode_loop_node started; command_topic={self.command_topic}, "
            f"auto_start_first_episode={self.auto_start_first_episode}"
        )

    def command_callback(self, msg):
        command = msg.data.strip().lower()

        if command in ("next", "start"):
            self.start_next_episode_from_command()
        elif command in ("cancel", "stop"):
            self.cancel_active_goal()
        else:
            self.get_logger().warn(f"Unknown episode loop command: {msg.data}")

    def tick(self):
        if self.state == "WAITING_FOR_SERVERS":
            if not self.action_servers_ready():
                if not self.waiting_for_servers_logged:
                    self.get_logger().info("waiting for GraspCup/MoveCup action servers")
                    self.waiting_for_servers_logged = True
                return

            self.waiting_for_servers_logged = False
            self.set_state("WAITING_FOR_RESET")

        if (
            self.auto_start_first_episode
            and not self.first_episode_started
            and self.state == "WAITING_FOR_RESET"
        ):
            self.start_episode()

    def action_servers_ready(self):
        return (
            self.grasp_client.wait_for_server(timeout_sec=0.0)
            and self.move_client.wait_for_server(timeout_sec=0.0)
        )

    def start_next_episode_from_command(self):
        if self.state == "WAITING_FOR_SERVERS":
            self.get_logger().warn("action servers are not ready yet")
            return

        if self.state != "WAITING_FOR_RESET":
            self.get_logger().warn(f"cannot start next episode while state={self.state}")
            return

        self.start_episode()

    def start_episode(self):
        self.first_episode_started = True
        self.episode_index += 1
        self.get_logger().info(f"starting episode loop round {self.episode_index}")
        self.start_grasp_action()

    def start_grasp_action(self):
        self.set_state("GRASPING_CUP")

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
            self.fail_and_wait("GraspCup action goal rejected")
            self.get_logger().warn("GraspCup action goal rejected")
            return

        self.active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.grasp_result_callback)

    def grasp_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f"loop grasp feedback: {feedback.status}, "
            f"error_x={feedback.error_x:.1f}, bbox_w={feedback.bbox_w:.1f}"
        )

    def grasp_result_callback(self, future):
        self.active_goal_handle = None
        result = future.result().result
        if not result.success:
            reason = (
                f"GraspCup failed: status={result.final_status}, "
                f"reason={result.failure_reason}"
            )
            self.fail_and_wait(reason)
            self.get_logger().warn(reason)
            return

        self.get_logger().info("GraspCup succeeded; starting MoveCup")
        self.start_move_action()

    def start_move_action(self):
        self.set_state("MOVING_CUP")

        goal = MoveCup.Goal()
        goal.timeout_sec = float(self.move_timeout_sec)
        goal.succeed_on_existing_done = False

        future = self.move_client.send_goal_async(
            goal,
            feedback_callback=self.move_feedback_callback,
        )
        future.add_done_callback(self.move_goal_response_callback)

    def move_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.fail_and_wait("MoveCup action goal rejected")
            self.get_logger().warn("MoveCup action goal rejected")
            return

        self.active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.move_result_callback)

    def move_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f"loop move feedback: {feedback.status}, elapsed={feedback.elapsed_sec:.1f}s"
        )

    def move_result_callback(self, future):
        self.active_goal_handle = None
        result = future.result().result
        if not result.success:
            reason = (
                f"MoveCup failed: status={result.final_status}, "
                f"reason={result.failure_reason}"
            )
            self.fail_and_wait(reason)
            self.get_logger().warn(reason)
            return

        self.set_state("DONE")
        self.get_logger().info(
            "episode round done; move the cup to a new position, then publish "
            f"`ros2 topic pub --once {self.command_topic} std_msgs/msg/String "
            "\"{data: next}\"`"
        )
        self.set_state("WAITING_FOR_RESET")

    def cancel_active_goal(self):
        if self.active_goal_handle is None:
            self.set_state("WAITING_FOR_RESET")
            return

        self.active_goal_handle.cancel_goal_async()
        self.active_goal_handle = None
        self.set_state("CANCELED")
        self.set_state("WAITING_FOR_RESET")

    def fail_and_wait(self, reason):
        self.get_logger().warn(
            f"{reason}; move/reset the cup if needed, then send next"
        )
        self.set_state("FAILED")
        self.set_state("WAITING_FOR_RESET")

    def set_state(self, state):
        self.state = state
        self.publish_status(state)

    def publish_status(self, status):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.task_status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = EpisodeLoopNode()

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
