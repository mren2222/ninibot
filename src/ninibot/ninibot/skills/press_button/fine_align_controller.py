import time

import numpy as np

from ninibot.skills.press_button.fine_align import average_samples
from ninibot.skills.press_button.ik_service_client import FineAlignIkServiceClient
from ninibot.skills.press_button.state_machine import Phase


class FineAlignController:
    def __init__(self, node):
        self.node = node

    def make_fine_align_context(
        self,
        visual_step_target,
        ik_step_target,
        distance,
        max_step,
        command_ahead,
    ):
        return {
            "visual_step_target": np.asarray(visual_step_target, dtype=float),
            "ik_step_target": np.asarray(ik_step_target, dtype=float),
            "distance": float(distance),
            "max_step": float(max_step),
            "command_ahead": float(command_ahead),
        }

    def start_fine_align_move_from_pose(self, pose, context, source):
        node = self.node
        node.fine_align_goal = list(pose)
        node.fine_align_target_base_xyz = np.asarray(
            context["visual_step_target"],
            dtype=float,
        )
        node.fine_align_cycle_index += 1
        node.settle_count = 0
        node.set_phase(
            Phase.FINE_ALIGN_MOVE,
            "press button: FINE_ALIGN_MOVE "
            f"source={source} "
            f"cycle={node.fine_align_cycle_index}/{node.fine_align_max_cycles} "
            f"tip-target={context['distance'] * 100.0:.1f}cm "
            f"step={context['max_step'] * 100.0:.1f}cm "
            f"command_ahead={context['command_ahead'] * 100.0:.1f}cm "
            f"visual_step_target={node.format_cm(context['visual_step_target'])} "
            f"ik_step_target={node.format_cm(context['ik_step_target'])} "
            f"q1234_goal_deg={node.format_joint_target_deg(node.fine_align_goal)} "
            f"{node.format_joint_error_to_target_deg(node.fine_align_goal)}",
        )

    def fallback_fine_align_ik(self, context, reason):
        node = self.node
        pose = node.make_fine_align_pose(context["ik_step_target"])
        if pose is None:
            node.set_phase(
                Phase.FAILED,
                "press button: FAILED fine align step unreachable "
                f"reason={reason} "
                f"visual_step_target={node.format_cm(context['visual_step_target'])} "
                f"ik_step_target={node.format_cm(context['ik_step_target'])} "
                f"tip-target={context['distance'] * 100.0:.1f}cm",
            )
            return True
        node.get_logger().warn(
            "press button: using internal fine-align IK fallback "
            f"reason={reason}"
        )
        self.start_fine_align_move_from_pose(
            pose,
            context,
            "internal_fallback",
        )
        return True

    def start_fine_align_ik_service_request(self, context):
        node = self.node
        started = node.ik_service.start_request(
            context,
            target_xyz=context["ik_step_target"],
            seed_q1234=node.local_ik_reference(),
            lock_joint4=node.fine_align_lock_joint4,
            max_joint_step_rad=node.fine_align_max_joint_step_rad,
            ik_position_tolerance=node.fine_align_local_tolerance(),
            joint4_min_rad=node.ik_lower_bounds[3],
            joint4_max_rad=node.ik_upper_bounds[3],
            fixed_joint5_rad=node.fixed_joint5_rad,
        )
        if not started:
            return False
        node.set_phase(
            Phase.FINE_ALIGN_IK_WAIT,
            "press button: FINE_ALIGN_IK_WAIT "
            f"service={node.fine_align_ik_service_name} "
            f"target={node.format_cm(context['ik_step_target'])} "
            f"lock_joint4={node.fine_align_lock_joint4}",
        )
        return True

    def finish_fine_align_ik_service_wait(self):
        node = self.node
        result = node.ik_service.poll(node.fine_align_near_miss_tolerance())
        if result.kind == FineAlignIkServiceClient.WAITING:
            return False
        if result.kind == FineAlignIkServiceClient.FALLBACK:
            return self.fallback_fine_align_ik(result.context, result.reason)
        if result.kind == FineAlignIkServiceClient.FAILED:
            node.set_phase(Phase.FAILED, f"press button: FAILED {result.reason}")
            return True
        if result.warning:
            node.get_logger().warn(result.warning)
        node.ik_planner.record_result(
            result.context["ik_step_target"],
            result.residual_xyz,
            result.error,
            result.q1234,
        )
        self.start_fine_align_move_from_pose(
            result.pose,
            result.context,
            result.source,
        )
        return True

    def start_fine_align_sample(self, reason=""):
        node = self.node
        node.freeze_current_target()
        node.fine_align_goal = None
        node.fine_align_target_base_xyz = None
        node.fine_align_samples = []
        node.fine_align_sample_start_time = time.monotonic()
        node.settle_count = 0
        reason_text = f"; {reason}" if reason else ""
        sample_cycle = min(
            node.fine_align_cycle_index + 1,
            node.fine_align_max_cycles,
        )
        node.set_phase(
            Phase.FINE_ALIGN_SAMPLE,
            "press button: FINE_ALIGN_SAMPLE "
            f"cycle={sample_cycle}/{node.fine_align_max_cycles} "
            f"samples={node.fine_align_marker_samples}{reason_text}",
        )

    def collect_fine_align_sample(self):
        node = self.node
        marker_base_xyz = node.detect_marker()
        if marker_base_xyz is None or node.marker_target_base_xyz is None:
            return False
        node.fine_align_samples.append(
            np.asarray(node.marker_target_base_xyz, dtype=float)
        )
        return True

    def fine_align_sample_ready(self):
        node = self.node
        if node.fine_align_sample_start_time is None:
            return False
        waited = time.monotonic() - node.fine_align_sample_start_time
        return (
            waited >= node.fine_align_sample_wait_sec
            and len(node.fine_align_samples) >= node.fine_align_marker_samples
        )

    def make_fine_align_ik_step_target(self, fk_tip_xyz, step_direction, max_step):
        node = self.node
        target, command_ahead = node.fine_align_command_planner.step_target(
            fk_tip_xyz,
            step_direction,
            max_step,
        )
        node.fine_align_command_ahead = (
            node.fine_align_command_planner.command_ahead
        )
        node.fine_align_command_direction = (
            None
            if node.fine_align_command_planner.command_direction is None
            else node.fine_align_command_planner.command_direction.copy()
        )
        return target, command_ahead

    def finish_fine_align_sample(self):
        node = self.node
        if not node.fine_align_samples:
            node.log_progress("press button: FINE_ALIGN waiting for marker samples")
            return False

        target_base_xyz = average_samples(node.fine_align_samples)
        node.move_target_base_xyz = target_base_xyz
        node.marker_target_base_xyz = target_base_xyz

        tip_base_xyz = node.lookup_gripper_base_xyz()
        if tip_base_xyz is None:
            node.log_progress("press button: FINE_ALIGN waiting for grippertip TF")
            return False
        node.gripper_base_xyz = tip_base_xyz

        error = target_base_xyz - tip_base_xyz
        distance = float(np.linalg.norm(error))
        if distance <= node.fine_align_tolerance:
            node.move_goal = list(node.current_target or node.move_goal or [])
            node.get_logger().info(
                "press button: FINE_ALIGN complete "
                f"tip-target={distance * 100.0:.2f}cm "
                f"target={node.format_cm(target_base_xyz)} "
                f"tip={node.format_cm(tip_base_xyz)}"
            )
            node.start_press_wait()
            return True

        if node.fine_align_cycle_index >= node.fine_align_max_cycles:
            node.set_phase(
                Phase.FAILED,
                "press button: FAILED fine align max cycles reached "
                f"tip-target={distance * 100.0:.1f}cm "
                f"tol={node.fine_align_tolerance * 100.0:.1f}cm",
            )
            return True

        max_step = min(node.fine_align_max_step, distance)
        if max_step <= 1e-6:
            node.start_press_wait()
            return True
        step_direction = error / distance
        visual_step_target = tip_base_xyz + step_direction * max_step
        fk_tip_xyz = node.fk_grippertip_xyz_from_q1234(node.local_ik_reference())
        ik_step_target, command_ahead = self.make_fine_align_ik_step_target(
            fk_tip_xyz,
            step_direction,
            max_step,
        )
        context = self.make_fine_align_context(
            visual_step_target,
            ik_step_target,
            distance,
            max_step,
            command_ahead,
        )
        if self.start_fine_align_ik_service_request(context):
            return True
        if node.fine_align_use_ik_service and not node.fine_align_ik_service_fallback:
            node.set_phase(Phase.FAILED, "press button: FAILED IK service unavailable")
            return True
        return self.fallback_fine_align_ik(context, "service_unavailable")

    def ramp_fine_align_target(self):
        node = self.node
        return node.ramp_joint_goal(
            node.fine_align_goal,
            range(4),
            node.fine_align_speed_scale,
        )
