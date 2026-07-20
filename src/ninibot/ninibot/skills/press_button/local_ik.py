import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from urdf_parser_py.urdf import URDF

from ninibot.skills.press_button.debug_view import format_cm_delta
from ninibot.skills.press_button.debug_view import format_degrees
from ninibot.skills.press_button.target_planner import joint4_rad_from_joint23


def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def axis_angle_to_matrix(axis, angle):
    axis = np.asarray(axis if axis is not None else [1.0, 0.0, 0.0], dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x + z * s - y * x * c, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def transform_from_xyz_rpy(xyz=None, rpy=None):
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rpy_to_matrix(*(rpy or [0.0, 0.0, 0.0]))
    transform[:3, 3] = np.asarray(xyz or [0.0, 0.0, 0.0], dtype=float)
    return transform


def transform_from_axis_angle(axis, angle):
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = axis_angle_to_matrix(axis, angle)
    return transform


@dataclass(frozen=True)
class LocalIkConfig:
    urdf_path: str
    base_frame: str
    gripper_frame: str
    prep_joint1_deg: float
    prep_joint2_deg: float
    prep_joint3_deg: float
    prep_joint4_deg: float
    search_pan_min_deg: float
    search_pan_max_deg: float
    align_wrist_pitch_deg: float
    ik_joint4_min_deg: float
    ik_joint4_max_deg: float
    fixed_joint5_rad: float
    joint6_rad: float
    ik_position_tolerance: float
    fine_align_max_joint_step_rad: float
    fine_align_lock_joint4: bool
    fine_align_max_step: float


class PressButtonLocalIk:
    def __init__(self, config: LocalIkConfig, log_progress=None, warn=None):
        self.config = config
        self.log_progress = log_progress or (lambda text, period_sec=1.0: None)
        self.warn = warn or (lambda text: None)
        self.robot_model = URDF.from_xml_file(config.urdf_path)
        self.urdf_joints = {joint.name: joint for joint in self.robot_model.joints}
        self.fk_chain = self.build_fk_chain(config.base_frame, config.gripper_frame)
        self.lower_bounds, self.upper_bounds = self.load_joint_bounds(["1", "2", "3", "4"])
        self.joint4_lower = np.asarray([self.lower_bounds[3]], dtype=float)
        self.joint4_upper = np.asarray([self.upper_bounds[3]], dtype=float)
        self.lower_bounds[3] = max(
            self.lower_bounds[3],
            math.radians(config.ik_joint4_min_deg),
        )
        self.upper_bounds[3] = min(
            self.upper_bounds[3],
            math.radians(config.ik_joint4_max_deg),
        )
        if self.lower_bounds[3] > self.upper_bounds[3]:
            raise RuntimeError(
                "Configured joint4 IK range does not overlap URDF limit: "
                f"[{config.ik_joint4_min_deg:.1f}, {config.ik_joint4_max_deg:.1f}] deg"
            )
        self.last_target_xyz = None
        self.last_residual = None
        self.last_error = None
        self.last_solution = None

    def reset_last_result(self):
        self.last_target_xyz = None
        self.last_residual = None
        self.last_error = None
        self.last_solution = None

    def record_result(self, target_xyz, residual, error, q1234):
        self.last_target_xyz = np.asarray(target_xyz, dtype=float).copy()
        self.last_residual = np.asarray(residual, dtype=float).copy()
        self.last_error = float(error)
        self.last_solution = list(q1234[:4])

    def build_fk_chain(self, root_link, tip_link):
        child_to_joint = {joint.child: joint for joint in self.robot_model.joints}
        chain = []
        link = tip_link
        while link != root_link:
            joint = child_to_joint.get(link)
            if joint is None:
                raise RuntimeError(
                    f"Could not find URDF chain {root_link}->{tip_link}; stopped at {link}"
                )
            chain.append(joint)
            link = joint.parent
        chain.reverse()
        return chain

    def load_joint_bounds(self, joint_names):
        lower = []
        upper = []
        for joint_name in joint_names:
            joint = self.urdf_joints[joint_name]
            if joint.limit is None:
                raise RuntimeError(f"URDF joint {joint_name} has no limits")
            lower.append(float(joint.limit.lower))
            upper.append(float(joint.limit.upper))
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

    def forward_grippertip_xyz(self, q1234):
        joint_values = {
            "1": q1234[0],
            "2": q1234[1],
            "3": q1234[2],
            "4": q1234[3],
            "5": self.config.fixed_joint5_rad,
            "6": self.config.joint6_rad,
        }
        transform = np.eye(4, dtype=float)
        for joint in self.fk_chain:
            origin = joint.origin
            transform = transform @ transform_from_xyz_rpy(
                None if origin is None else origin.xyz,
                None if origin is None else origin.rpy,
            )
            if joint.type in ("revolute", "continuous"):
                transform = transform @ transform_from_axis_angle(
                    joint.axis,
                    float(joint_values.get(joint.name, 0.0)),
                )
        return transform[:3, 3].copy()

    def joint4_rad_from_joint23(self, joint2_rad, joint3_rad):
        return joint4_rad_from_joint23(
            joint2_rad,
            joint3_rad,
            self.config.align_wrist_pitch_deg,
        )

    def is_joint4_in_range(self, joint4):
        return self.joint4_lower[0] <= joint4 <= self.joint4_upper[0]

    def initial_guess(self, latest_joint_positions=None, current_target=None):
        if latest_joint_positions is not None:
            return np.asarray(latest_joint_positions[:4], dtype=float)
        if current_target is not None:
            return np.asarray(current_target[:4], dtype=float)
        return np.asarray(
            [
                math.radians(self.config.prep_joint1_deg),
                math.radians(self.config.prep_joint2_deg),
                math.radians(self.config.prep_joint3_deg),
                math.radians(self.config.prep_joint4_deg),
            ],
            dtype=float,
        )

    def seed_guesses(self, q0):
        seeds = [np.asarray(q0, dtype=float)]
        seeds.append(
            np.asarray(
                [
                    math.radians(self.config.prep_joint1_deg),
                    math.radians(self.config.prep_joint2_deg),
                    math.radians(self.config.prep_joint3_deg),
                    math.radians(self.config.prep_joint4_deg),
                ],
                dtype=float,
            )
        )

        base_candidates = [
            q0[0],
            math.radians(self.config.prep_joint1_deg),
            math.radians(self.config.search_pan_min_deg),
            math.radians(self.config.search_pan_max_deg),
            0.0,
        ]
        arm_shape_candidates = [
            (q0[1], q0[2]),
            (math.radians(-90.0), math.radians(98.0)),
            (math.radians(-75.0), math.radians(90.0)),
            (math.radians(-60.0), math.radians(80.0)),
            (math.radians(-45.0), math.radians(60.0)),
            (math.radians(-30.0), math.radians(45.0)),
            (math.radians(0.0), math.radians(30.0)),
        ]
        joint4_mid = 0.5 * (self.lower_bounds[3] + self.upper_bounds[3])
        joint4_candidates = [
            q0[3],
            math.radians(self.config.prep_joint4_deg),
            self.lower_bounds[3],
            joint4_mid,
            self.upper_bounds[3],
            self.joint4_rad_from_joint23(q0[1], q0[2]),
        ]

        for base in base_candidates:
            for shoulder, elbow in arm_shape_candidates:
                for joint4 in joint4_candidates:
                    seeds.append(np.asarray([base, shoulder, elbow, joint4], dtype=float))

        unique_seeds = []
        seen = set()
        for seed in seeds:
            clipped = np.clip(seed, self.lower_bounds, self.upper_bounds)
            key = tuple(np.round(clipped, 5))
            if key in seen:
                continue
            seen.add(key)
            unique_seeds.append(clipped)
        return unique_seeds

    def format_bound_flags(self, q):
        flags = []
        for name, value, lower, upper in zip(
            ["j1", "j2", "j3", "j4"],
            q,
            self.lower_bounds,
            self.upper_bounds,
        ):
            if abs(value - lower) < math.radians(1.0):
                flags.append(f"{name}=min")
            elif abs(value - upper) < math.radians(1.0):
                flags.append(f"{name}=max")
        return ",".join(flags) if flags else "none"

    def solve_grippertip_ik(self, target_xyz, latest_joint_positions=None, current_target=None):
        target_xyz = np.asarray(target_xyz, dtype=float)
        q0 = np.clip(
            self.initial_guess(latest_joint_positions, current_target),
            self.lower_bounds,
            self.upper_bounds,
        )

        def residual(q):
            return self.forward_grippertip_xyz(q) - target_xyz

        best = None
        best_error = float("inf")
        best_residual = None
        best_joint4 = None
        best_seed_index = 0
        best_any = None
        seed_guesses = self.seed_guesses(q0)
        for seed_index, seed in enumerate(seed_guesses):
            result = least_squares(
                residual,
                seed,
                bounds=(self.lower_bounds, self.upper_bounds),
                max_nfev=160,
                xtol=1e-7,
                ftol=1e-7,
                gtol=1e-7,
            )
            result_residual = residual(result.x)
            error = float(np.linalg.norm(result_residual))
            joint4 = result.x[3]
            any_candidate = (error, result, result_residual, joint4, seed_index)
            if best_any is None or error < best_any[0]:
                best_any = any_candidate
            if error < best_error:
                best_error = error
                best = result
                best_residual = result_residual
                best_joint4 = joint4
                best_seed_index = seed_index
            if error <= self.config.ik_position_tolerance:
                break

        if best is None:
            error, result, result_residual, joint4, seed_index = best_any
            self.log_progress(
                "press button: URDF IK failed; no valid joint4 solution "
                f"best_error={error * 100.0:.1f} cm "
                f"residual_cm={format_cm_delta(result_residual)} "
                f"q123_deg={format_degrees(result.x)} "
                f"joint4_deg={math.degrees(joint4):+.1f} "
                f"seed={seed_index}/{len(seed_guesses)} "
                f"success={result.success} status={result.status}"
            )
            return None

        error = best_error
        self.record_result(target_xyz, best_residual, error, best.x)
        if not best.success or error > self.config.ik_position_tolerance:
            self.log_progress(
                "press button: URDF IK failed "
                f"error={error * 100.0:.1f} cm "
                f"tol={self.config.ik_position_tolerance * 100.0:.1f} cm "
                f"residual_cm={format_cm_delta(best_residual)} "
                f"q123_deg={format_degrees(best.x)} "
                f"joint4_deg={math.degrees(best_joint4):+.1f} "
                f"bounds={self.format_bound_flags(best.x)} "
                f"seed={best_seed_index}/{len(seed_guesses)} "
                f"success={best.success} status={best.status}"
            )
            return None

        return best.x[0], best.x[1], best.x[2], best.x[3]

    def make_target_pose(self, target_base_xyz, latest_joint_positions=None, current_target=None):
        target_xyz = np.asarray(target_base_xyz, dtype=float).copy()
        ik = self.solve_grippertip_ik(target_xyz, latest_joint_positions, current_target)
        if ik is None:
            return None
        return [ik[0], ik[1], ik[2], ik[3], self.config.joint6_rad]

    def local_reference(self, latest_joint_positions=None, current_target=None):
        if latest_joint_positions is not None:
            return np.asarray(latest_joint_positions[:4], dtype=float)
        if current_target is not None:
            return np.asarray(current_target[:4], dtype=float)
        return self.initial_guess()

    def fine_align_local_tolerance(self):
        return min(
            self.config.ik_position_tolerance,
            max(0.001, 0.5 * self.config.fine_align_max_step),
        )

    def fine_align_near_miss_tolerance(self):
        return min(
            self.config.ik_position_tolerance,
            self.fine_align_local_tolerance() + 0.001,
        )

    def make_fine_align_pose(self, target_base_xyz, latest_joint_positions=None, current_target=None):
        target_xyz = np.asarray(target_base_xyz, dtype=float)
        q0 = np.clip(
            self.local_reference(latest_joint_positions, current_target),
            self.lower_bounds,
            self.upper_bounds,
        )
        joint_step = max(self.config.fine_align_max_joint_step_rad, math.radians(0.5))
        lower = np.maximum(self.lower_bounds, q0 - joint_step)
        upper = np.minimum(self.upper_bounds, q0 + joint_step)
        q0 = np.clip(q0, lower, upper)

        if self.config.fine_align_lock_joint4:
            fixed_joint4 = q0[3]

            def residual(q123):
                q = np.asarray([q123[0], q123[1], q123[2], fixed_joint4], dtype=float)
                return self.forward_grippertip_xyz(q) - target_xyz

            result = least_squares(
                residual,
                q0[:3],
                bounds=(lower[:3], upper[:3]),
                max_nfev=80,
                xtol=1e-7,
                ftol=1e-7,
                gtol=1e-7,
            )
            result_q = np.asarray([result.x[0], result.x[1], result.x[2], fixed_joint4], dtype=float)
            result_residual = residual(result.x)
        else:
            def residual(q):
                return self.forward_grippertip_xyz(q) - target_xyz

            result = least_squares(
                residual,
                q0,
                bounds=(lower, upper),
                max_nfev=80,
                xtol=1e-7,
                ftol=1e-7,
                gtol=1e-7,
            )
            result_q = np.asarray(result.x, dtype=float)
            result_residual = residual(result_q)

        error = float(np.linalg.norm(result_residual))
        self.record_result(target_xyz, result_residual, error, result_q)

        local_tolerance = self.fine_align_local_tolerance()
        if not result.success or error > local_tolerance:
            dq = result_q - q0
            self.warn(
                "press button: fine align local IK failed "
                f"error={error * 100.0:.1f}cm "
                f"tol={local_tolerance * 100.0:.1f}cm "
                f"residual_cm={format_cm_delta(result_residual)} "
                f"dq1234_deg={format_degrees(dq)} "
                f"lock_joint4={self.config.fine_align_lock_joint4} "
                f"bounds={self.format_bound_flags(result_q)} "
                f"success={result.success} status={result.status}"
            )
            return None

        return [
            result_q[0],
            result_q[1],
            result_q[2],
            result_q[3],
            self.config.joint6_rad,
        ]

