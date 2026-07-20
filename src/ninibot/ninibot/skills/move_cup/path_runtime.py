import time

from arm_controller.kinematics.cylindrical import CylindricalPose
from ninibot.skills.move_cup.phases import EXTENDING_RADIUS
from ninibot.skills.move_cup.phases import PATH_PHASES
from ninibot.skills.move_cup.phases import POST_RELEASE_CLEARANCE
from ninibot.skills.move_cup.phases import RETRACTING_RADIUS


class MoveCupPathRuntime:
    def __init__(self, node):
        self.node = node

    def clear_path(self):
        node = self.node
        node.path_start_pose = None
        node.path_goal_pose = None
        node.path_duration_sec = 0.0
        node.path_start_gripper_percent = node.hold_gripper_percent
        node.path_goal_gripper_percent = node.hold_gripper_percent

    def pose_or_path_goal(self):
        node = self.node
        if node.path_goal_pose is not None:
            return node.path_goal_pose

        pose = node.current_cylindrical_pose()
        if pose is not None:
            return pose

        return CylindricalPose(
            radius=node.target_radius,
            theta_deg=node.target_theta_deg,
            z=node.target_z,
        )

    def fail_path(self, text):
        node = self.node
        node.phase = "FAILED"
        node.current_target = None
        self.clear_path()
        node.publish_status(node.phase)
        node.get_logger().warn(text)

    def set_cylindrical_path_phase(
        self,
        phase,
        start_pose,
        goal_pose,
        duration_sec,
        gripper_percent,
        goal_gripper_percent=None,
    ):
        node = self.node
        if start_pose is None or goal_pose is None:
            self.fail_path("move cup failed: invalid cylindrical path")
            return

        node.phase = phase
        node.path_start_pose = start_pose
        node.path_goal_pose = goal_pose
        node.path_duration_sec = max(0.0, float(duration_sec))
        node.path_start_gripper_percent = gripper_percent
        node.path_goal_gripper_percent = (
            gripper_percent
            if goal_gripper_percent is None
            else goal_gripper_percent
        )
        node.settle_count = 0
        node.phase_started_at = time.monotonic()
        node.last_progress_log_time = 0.0
        node.current_target = node.make_pose_from_cylindrical(
            start_pose,
            gripper_percent,
        )

        if node.current_target is None:
            self.fail_path("move cup failed: invalid path start target")
            return

        node.publish_status(node.phase)
        node.get_logger().info(
            f"move cup: {node.phase} path "
            f"start=(r={start_pose.radius:.3f}, theta={start_pose.theta_deg:.1f}, z={start_pose.z:.3f}) "
            f"goal=(r={goal_pose.radius:.3f}, theta={goal_pose.theta_deg:.1f}, z={goal_pose.z:.3f}) "
            f"duration={node.path_duration_sec:.2f}s"
        )

    def path_complete(self):
        node = self.node
        if node.path_goal_pose is None or node.phase_started_at is None:
            return True

        return node.phase_elapsed_sec() >= node.path_duration_sec

    def update_path_target(self):
        node = self.node
        if node.phase not in PATH_PHASES or node.path_goal_pose is None:
            return

        if node.path_duration_sec <= 0.0:
            alpha = 1.0
        else:
            alpha = node.clamp(
                node.phase_elapsed_sec() / node.path_duration_sec,
                0.0,
                1.0,
            )

        pose = node.carry_planner.path_pose_at(
            node.path_start_pose,
            node.path_goal_pose,
            alpha,
        )
        gripper_percent = node.carry_planner.gripper_percent_at(
            node.path_start_gripper_percent,
            node.path_goal_gripper_percent,
            alpha,
        )
        target = node.make_pose_from_cylindrical(
            pose,
            gripper_percent,
        )

        if target is None:
            self.fail_path("move cup failed: invalid interpolated target")
            return

        node.current_target = target

    def fail_missing_pose(self, text):
        node = self.node
        node.phase = "FAILED"
        node.publish_status(node.phase)
        node.get_logger().warn(text)

    def start_lifting_phase(self):
        node = self.node
        start_pose = self.pose_or_path_goal()
        if start_pose is None:
            self.fail_missing_pose("move cup failed: no current pose for lifting")
            return

        node.active_carry_z = start_pose.z + node.lift_z_offset
        goal_pose = node.carry_planner.lift_goal(start_pose)
        self.set_cylindrical_path_phase(
            "LIFTING",
            start_pose,
            goal_pose,
            node.lift_duration_sec,
            node.hold_gripper_percent,
        )

    def start_retract_radius_phase(self):
        node = self.node
        start_pose = self.pose_or_path_goal()
        if start_pose is None:
            self.fail_missing_pose("move cup failed: no current pose for radius retract")
            return

        goal_pose = node.carry_planner.retract_radius_goal(start_pose)
        self.set_cylindrical_path_phase(
            RETRACTING_RADIUS,
            start_pose,
            goal_pose,
            node.pre_pan_retract_duration_sec,
            node.hold_gripper_percent,
        )

    def start_horizontal_move_phase(self):
        node = self.node
        start_pose = self.pose_or_path_goal()
        if start_pose is None:
            self.fail_missing_pose("move cup failed: no current pose for pan move")
            return

        if node.use_dynamic_carry_z:
            node.active_carry_z = start_pose.z
        elif node.active_carry_z is None:
            node.active_carry_z = node.carry_z

        goal_pose = node.carry_planner.horizontal_move_goal(
            start_pose,
            node.target_theta_deg,
            node.active_carry_z,
        )
        self.set_cylindrical_path_phase(
            "MOVING_TO_TARGET",
            start_pose,
            goal_pose,
            node.horizontal_move_duration_sec,
            node.hold_gripper_percent,
        )

    def start_extend_radius_phase(self):
        node = self.node
        start_pose = self.pose_or_path_goal()
        if start_pose is None:
            self.fail_missing_pose("move cup failed: no current pose for radius extend")
            return

        target_radius = node.clamp_radius_for_reach_margin(
            node.target_radius,
            node.active_carry_z if node.active_carry_z is not None else start_pose.z,
            EXTENDING_RADIUS,
        )
        node.target_radius = target_radius
        goal_pose = node.carry_planner.extend_radius_goal(
            start_pose,
            target_radius,
            node.active_carry_z,
        )
        self.set_cylindrical_path_phase(
            EXTENDING_RADIUS,
            start_pose,
            goal_pose,
            node.post_pan_extend_duration_sec,
            node.hold_gripper_percent,
        )

    def start_lowering_phase(self):
        node = self.node
        start_pose = self.pose_or_path_goal()
        if start_pose is None:
            self.fail_missing_pose("move cup failed: no current pose for lowering")
            return

        goal_pose = node.carry_planner.lower_goal(start_pose)
        self.set_cylindrical_path_phase(
            "LOWERING_TO_RELEASE",
            start_pose,
            goal_pose,
            node.lower_duration_sec,
            node.hold_gripper_percent,
        )

    def pose_with_base_pan_offset(self, start_pose, base_pan_offset_deg):
        node = self.node
        plan = node.carry_planner.release_goal(start_pose, base_pan_offset_deg)
        if plan.warning_text:
            node.get_logger().warn(plan.warning_text)
        return plan.pose

    def start_releasing_phase(self, start_pose):
        node = self.node
        if start_pose is None:
            self.fail_missing_pose("move cup failed: no current pose for releasing")
            return

        goal_pose = self.pose_with_base_pan_offset(
            start_pose,
            node.release_pan_offset_deg,
        )
        node.release_cylindrical_pose = goal_pose
        self.set_cylindrical_path_phase(
            "RELEASING",
            start_pose,
            goal_pose,
            node.release_settle_sec,
            node.hold_gripper_percent,
            goal_gripper_percent=node.release_gripper_percent,
        )

    def start_post_release_clearance_phase(self):
        node = self.node
        start_pose = (
            node.current_cylindrical_pose()
            or node.release_cylindrical_pose
            or self.pose_or_path_goal()
        )
        if start_pose is None:
            self.fail_missing_pose(
                "move cup failed: no current pose for post-release clearance"
            )
            return

        goal_pose = node.carry_planner.post_release_clearance_goal(start_pose)
        self.set_cylindrical_path_phase(
            POST_RELEASE_CLEARANCE,
            start_pose,
            goal_pose,
            node.post_release_clearance_duration_sec,
            node.release_gripper_percent,
        )
