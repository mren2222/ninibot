# flake8: noqa
import math

import numpy as np
from scipy.optimize import least_squares
from urdf_parser_py.urdf import URDF


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
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
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


class GrippertipIKSolver:
    def __init__(self, urdf_path, base_frame="base", gripper_frame="grippertip"):
        self.urdf_path = urdf_path
        self.base_frame = base_frame
        self.gripper_frame = gripper_frame
        self.robot_model = URDF.from_xml_file(urdf_path)
        self.urdf_joints = {joint.name: joint for joint in self.robot_model.joints}
        self.fk_chain = self.build_fk_chain(base_frame, gripper_frame)
        self.lower_bounds, self.upper_bounds = self.load_joint_bounds(["1", "2", "3", "4"])

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

    def forward_grippertip_xyz(self, q1234, fixed_joint5_rad, joint6_rad):
        joint_values = {
            "1": q1234[0],
            "2": q1234[1],
            "3": q1234[2],
            "4": q1234[3],
            "5": fixed_joint5_rad,
            "6": joint6_rad,
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

    def solve(
        self,
        target_xyz,
        seed_q1234=None,
        lock_joint4=False,
        max_joint_step_deg=0.0,
        ik_position_tolerance=0.01,
        joint4_min_deg=None,
        joint4_max_deg=None,
        fixed_joint5_rad=0.0,
        joint6_rad=0.0,
    ):
        target_xyz = np.asarray(target_xyz, dtype=float)
        if seed_q1234 is None or len(seed_q1234) < 4:
            q0 = np.zeros(4, dtype=float)
        else:
            q0 = np.asarray(seed_q1234[:4], dtype=float)

        lower = self.lower_bounds.copy()
        upper = self.upper_bounds.copy()
        if joint4_min_deg is not None and joint4_max_deg is not None:
            joint4_min = math.radians(float(joint4_min_deg))
            joint4_max = math.radians(float(joint4_max_deg))
            if joint4_min > joint4_max:
                joint4_min, joint4_max = joint4_max, joint4_min
            lower[3] = max(lower[3], joint4_min)
            upper[3] = min(upper[3], joint4_max)

        q0 = np.clip(q0, lower, upper)

        if max_joint_step_deg and max_joint_step_deg > 0.0:
            joint_step = math.radians(float(max_joint_step_deg))
            lower = np.maximum(lower, q0 - joint_step)
            upper = np.minimum(upper, q0 + joint_step)
            q0 = np.clip(q0, lower, upper)

        if lock_joint4:
            fixed_joint4 = q0[3]

            def residual(q123):
                q = np.asarray([q123[0], q123[1], q123[2], fixed_joint4], dtype=float)
                return self.forward_grippertip_xyz(q, fixed_joint5_rad, joint6_rad) - target_xyz

            result = least_squares(
                residual,
                q0[:3],
                bounds=(lower[:3], upper[:3]),
                max_nfev=80,
                xtol=1e-7,
                ftol=1e-7,
                gtol=1e-7,
            )
            q = np.asarray([result.x[0], result.x[1], result.x[2], fixed_joint4], dtype=float)
            result_residual = residual(result.x)
        else:
            def residual(q):
                return self.forward_grippertip_xyz(q, fixed_joint5_rad, joint6_rad) - target_xyz

            result = least_squares(
                residual,
                q0,
                bounds=(lower, upper),
                max_nfev=120,
                xtol=1e-7,
                ftol=1e-7,
                gtol=1e-7,
            )
            q = np.asarray(result.x, dtype=float)
            result_residual = residual(q)

        error = float(np.linalg.norm(result_residual))
        success = bool(result.success and error <= float(ik_position_tolerance))
        return {
            "success": success,
            "q1234": q,
            "error": error,
            "residual_xyz": np.asarray(result_residual, dtype=float),
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "message": str(result.message),
        }
