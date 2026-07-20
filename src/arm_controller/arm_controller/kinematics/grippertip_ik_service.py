# flake8: noqa
from ament_index_python.packages import get_package_share_directory
import math
import os

import numpy as np
import rclpy
from rclpy.node import Node

from ninibot_interfaces.srv import SolveGrippertipIk

from arm_controller.kinematics.grippertip_ik_solver import GrippertipIKSolver


def format_deg(values):
    return "[" + ", ".join(f"{math.degrees(value):+.1f}" for value in values) + "]"


def default_urdf_path():
    return os.path.join(
        get_package_share_directory("arm_controller"),
        "urdf",
        "so101.urdf",
    )


class GrippertipIKService(Node):
    def __init__(self):
        super().__init__("grippertip_ik_service")
        self.declare_parameter("service_name", "/solve_grippertip_ik")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("base_frame", "base")
        self.declare_parameter("gripper_frame", "grippertip")

        self.service_name = self.get_parameter("service_name").value
        self.urdf_path = self.get_parameter("urdf_path").value or default_urdf_path()
        self.base_frame = self.get_parameter("base_frame").value
        self.gripper_frame = self.get_parameter("gripper_frame").value

        self.solver = GrippertipIKSolver(
            self.urdf_path,
            self.base_frame,
            self.gripper_frame,
        )
        self.service = self.create_service(
            SolveGrippertipIk,
            self.service_name,
            self.solve_callback,
        )
        self.get_logger().info(
            "grippertip_ik_service started: "
            f"service={self.service_name}, urdf={self.urdf_path}, "
            f"chain={self.base_frame}->{self.gripper_frame}"
        )

    def solve_callback(self, request, response):
        target_xyz = np.asarray(
            [request.target_x, request.target_y, request.target_z],
            dtype=float,
        )
        seed = list(request.seed_q1234)
        if len(seed) < 4:
            seed = [0.0, 0.0, 0.0, 0.0]

        try:
            result = self.solver.solve(
                target_xyz,
                seed_q1234=seed,
                lock_joint4=bool(request.lock_joint4),
                max_joint_step_deg=float(request.max_joint_step_deg),
                ik_position_tolerance=float(request.ik_position_tolerance),
                joint4_min_deg=float(request.joint4_min_deg),
                joint4_max_deg=float(request.joint4_max_deg),
                fixed_joint5_rad=float(request.fixed_joint5_rad),
                joint6_rad=float(request.joint6_rad),
            )
        except Exception as exc:
            response.success = False
            response.message = f"IK exception: {exc}"
            response.q1234 = []
            response.error = float("inf")
            response.residual_xyz = []
            self.get_logger().warn(response.message)
            return response

        response.success = bool(result["success"])
        response.q1234 = [float(value) for value in result["q1234"]]
        response.error = float(result["error"])
        response.residual_xyz = [float(value) for value in result["residual_xyz"]]
        response.message = (
            f"error={response.error * 100.0:.2f}cm "
            f"q1234_deg={format_deg(response.q1234)} "
            f"optimizer_success={result['optimizer_success']} "
            f"status={result['optimizer_status']} "
            f"lock_joint4={request.lock_joint4}"
        )

        if response.success:
            self.get_logger().debug(response.message)
        else:
            self.get_logger().warn(f"IK failed: {response.message}")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GrippertipIKService()
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
