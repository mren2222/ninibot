# flake8: noqa
import math
from pathlib import Path
import re
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import torch
from torch import nn

from arm_controller.kinematics.cylindrical import CylindricalArmKinematics
from ninibot.task_params import CYLINDRICAL_VISION_PARAMS


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
DEBUG_FIELD_RE = re.compile(rf"([A-Za-z_][A-Za-z0-9_]*)=({FLOAT_RE})")


class StructuredPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class FineAlignPolicyGate(Node):
    def __init__(self):
        super().__init__("fine_align_policy_gate")

        self.declare_parameter(
            "model_path",
            "data/models/structured_policy_fine_align/policy.pt",
        )
        self.declare_parameter("rule_based_target_topic", "/rule_based_target_joint_states")
        self.declare_parameter("output_topic", "/target_joint_states")
        self.declare_parameter("real_joint_states_topic", "/real_joint_states")
        self.declare_parameter("fallback_joint_states_topic", "/joint_states")
        self.declare_parameter("cup_status_topic", "/cup_servo_status")
        self.declare_parameter("cup_debug_topic", "/cup_servo_debug")
        self.declare_parameter("debug_topic", "/fine_align_policy_gate_debug")
        self.declare_parameter("control_rate", 10.0)
        self.declare_parameter("mode", "one_step")
        self.declare_parameter("active_statuses", "FINE_ALIGN")
        self.declare_parameter(
            "forward_statuses",
            "STARTING,SEARCH_PREP,SEARCHING,LOST,TRACKING_CENTER,FINE_ALIGN_PREP",
        )
        self.declare_parameter("policy_start_source", "real")
        self.declare_parameter("policy_feature_source", "real")
        self.declare_parameter("debug_timeout_sec", 1.0)
        self.declare_parameter("rule_target_timeout_sec", 1.0)
        self.declare_parameter("fallback_to_rule_based", True)
        self.declare_parameter("freeze_gripper", True)
        self.declare_parameter("maintain_wrist_level", True)
        self.declare_parameter(
            "shoulder_max_deg",
            CYLINDRICAL_VISION_PARAMS["shoulder_max_deg"],
        )
        self.declare_parameter("action_scale", 1.0)
        self.declare_parameter("theta_action_scale", 1.0)
        self.declare_parameter("radius_action_scale", 1.0)
        self.declare_parameter("max_delta_theta_rad", 0.006)
        self.declare_parameter("max_delta_radius_m", 0.002)
        self.declare_parameter("max_delta_joint2_rad", 0.012)
        self.declare_parameter("max_delta_joint3_rad", 0.025)
        self.declare_parameter("max_delta_gripper_rad", 0.005)

        self.model_path = Path(str(self.get_parameter("model_path").value)).expanduser()
        self.rule_based_target_topic = self.get_parameter(
            "rule_based_target_topic"
        ).value
        self.output_topic = self.get_parameter("output_topic").value
        self.real_joint_states_topic = self.get_parameter(
            "real_joint_states_topic"
        ).value
        self.fallback_joint_states_topic = self.get_parameter(
            "fallback_joint_states_topic"
        ).value
        self.cup_status_topic = self.get_parameter("cup_status_topic").value
        self.cup_debug_topic = self.get_parameter("cup_debug_topic").value
        self.debug_topic = self.get_parameter("debug_topic").value
        self.control_rate = float(self.get_parameter("control_rate").value)
        self.mode = str(self.get_parameter("mode").value).strip().lower()
        self.active_statuses = self.csv_set(
            self.get_parameter("active_statuses").value
        )
        self.forward_statuses = self.csv_set(
            self.get_parameter("forward_statuses").value
        )
        self.policy_start_source = str(
            self.get_parameter("policy_start_source").value
        ).strip().lower()
        self.policy_feature_source = str(
            self.get_parameter("policy_feature_source").value
        ).strip().lower()
        self.debug_timeout_sec = float(
            self.get_parameter("debug_timeout_sec").value
        )
        self.rule_target_timeout_sec = float(
            self.get_parameter("rule_target_timeout_sec").value
        )
        self.fallback_to_rule_based = bool(
            self.get_parameter("fallback_to_rule_based").value
        )
        self.freeze_gripper = bool(self.get_parameter("freeze_gripper").value)
        self.maintain_wrist_level = bool(
            self.get_parameter("maintain_wrist_level").value
        )
        self.shoulder_max_deg = float(self.get_parameter("shoulder_max_deg").value)
        self.action_scale = float(self.get_parameter("action_scale").value)
        self.theta_action_scale = float(
            self.get_parameter("theta_action_scale").value
        )
        self.radius_action_scale = float(
            self.get_parameter("radius_action_scale").value
        )
        self.max_delta_theta_rad = float(
            self.get_parameter("max_delta_theta_rad").value
        )
        self.max_delta_radius_m = float(
            self.get_parameter("max_delta_radius_m").value
        )
        self.max_delta_joint2_rad = float(
            self.get_parameter("max_delta_joint2_rad").value
        )
        self.max_delta_joint3_rad = float(
            self.get_parameter("max_delta_joint3_rad").value
        )
        self.max_delta_gripper_rad = float(
            self.get_parameter("max_delta_gripper_rad").value
        )

        if self.mode not in ("one_step", "integrated"):
            self.get_logger().warn(
                f"Unknown mode={self.mode!r}; falling back to one_step"
            )
            self.mode = "one_step"

        for attr in ("policy_start_source", "policy_feature_source"):
            value = getattr(self, attr)

            if value not in ("real", "rule", "fallback", "policy"):
                self.get_logger().warn(
                    f"Unknown {attr}={value!r}; falling back to real"
                )
                setattr(self, attr, "real")

        self.joint_names = ["1", "2", "3", "4", "5", "6"]
        self.latest_rule_positions = None
        self.latest_rule_time = 0.0
        self.latest_real_positions = None
        self.latest_fallback_positions = None
        self.policy_positions = None
        self.cup_status = ""
        self.debug_values = {}
        self.last_debug_time = 0.0
        self.last_status = ""
        self.last_debug_publish_time = 0.0
        self.last_ik_warn_time = 0.0

        self.kinematics = CylindricalArmKinematics(
            l1=CYLINDRICAL_VISION_PARAMS["l1"],
            l2=CYLINDRICAL_VISION_PARAMS["l2"],
            horizontal_tool_offset=CYLINDRICAL_VISION_PARAMS[
                "horizontal_tool_offset"
            ],
            z_tool_offset=CYLINDRICAL_VISION_PARAMS["z_tool_offset"],
            theta_joint_sign=CYLINDRICAL_VISION_PARAMS["theta_joint_sign"],
            theta_joint_offset_deg=CYLINDRICAL_VISION_PARAMS[
                "theta_joint_offset_deg"
            ],
            base_min_deg=CYLINDRICAL_VISION_PARAMS["base_min_deg"],
            base_max_deg=CYLINDRICAL_VISION_PARAMS["base_max_deg"],
            shoulder_min_deg=CYLINDRICAL_VISION_PARAMS["shoulder_min_deg"],
            shoulder_max_deg=self.shoulder_max_deg,
            elbow_min_deg=CYLINDRICAL_VISION_PARAMS["elbow_min_deg"],
            elbow_max_deg=CYLINDRICAL_VISION_PARAMS["elbow_max_deg"],
            wrist_flex_min_deg=CYLINDRICAL_VISION_PARAMS["wrist_flex_min_deg"],
            wrist_flex_max_deg=CYLINDRICAL_VISION_PARAMS["wrist_flex_max_deg"],
            horizontal_wrist_pitch_offset_deg=CYLINDRICAL_VISION_PARAMS[
                "horizontal_wrist_pitch_offset_deg"
            ],
        )

        self.model, self.model_blob = self.load_model(self.model_path)
        self.feature_columns = list(self.model_blob["feature_columns"])
        self.target_columns = list(self.model_blob["target_columns"])
        self.status_names = list(self.model_blob["status_names"])
        self.x_mean = torch.tensor(self.model_blob["x_mean"], dtype=torch.float32)
        self.x_std = torch.tensor(self.model_blob["x_std"], dtype=torch.float32)
        self.y_mean = torch.tensor(self.model_blob["y_mean"], dtype=torch.float32)
        self.y_std = torch.tensor(self.model_blob["y_std"], dtype=torch.float32)

        self.target_pub = self.create_publisher(JointState, self.output_topic, 10)
        self.debug_pub = self.create_publisher(String, self.debug_topic, 10)
        self.rule_sub = self.create_subscription(
            JointState,
            self.rule_based_target_topic,
            self.rule_target_callback,
            10,
        )
        self.real_sub = self.create_subscription(
            JointState,
            self.real_joint_states_topic,
            self.real_joint_callback,
            10,
        )
        self.fallback_sub = self.create_subscription(
            JointState,
            self.fallback_joint_states_topic,
            self.fallback_joint_callback,
            10,
        )
        self.status_sub = self.create_subscription(
            String,
            self.cup_status_topic,
            self.status_callback,
            10,
        )
        self.cup_debug_sub = self.create_subscription(
            String,
            self.cup_debug_topic,
            self.cup_debug_callback,
            10,
        )
        self.timer = self.create_timer(1.0 / self.control_rate, self.loop)

        self.get_logger().warn(
            "fine_align_policy_gate is allowed to publish /target_joint_states "
            f"during {sorted(self.active_statuses)}; mode={self.mode}, "
            f"action_scale={self.action_scale:.2f}, "
            f"theta_scale={self.theta_action_scale:.2f}, "
            f"radius_scale={self.radius_action_scale:.2f}"
        )

    def csv_set(self, value):
        return {
            item.strip()
            for item in str(value).split(",")
            if item.strip()
        }

    def torch_load(self, path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def load_model(self, path):
        if not path.exists():
            raise FileNotFoundError(f"policy model not found: {path}")

        blob = self.torch_load(path)
        model = StructuredPolicy(
            input_dim=int(blob["input_dim"]),
            output_dim=int(blob["output_dim"]),
            hidden_dim=int(blob["hidden_dim"]),
            dropout=0.0,
        )
        model.load_state_dict(blob["model_state_dict"])
        model.eval()
        return model, blob

    def positions_from_joint_state(self, msg):
        name_to_position = {
            name: position
            for name, position in zip(msg.name, msg.position)
        }

        if not all(name in name_to_position for name in self.joint_names):
            return None

        return [
            float(name_to_position[name])
            for name in self.joint_names
        ]

    def rule_target_callback(self, msg):
        positions = self.positions_from_joint_state(msg)

        if positions is not None:
            self.latest_rule_positions = positions
            self.latest_rule_time = time.monotonic()

    def real_joint_callback(self, msg):
        positions = self.positions_from_joint_state(msg)

        if positions is not None:
            self.latest_real_positions = positions

    def fallback_joint_callback(self, msg):
        positions = self.positions_from_joint_state(msg)

        if positions is not None:
            self.latest_fallback_positions = positions

    def status_callback(self, msg):
        status = msg.data.strip()

        if status != self.last_status:
            self.policy_positions = None
            self.last_status = status

        self.cup_status = status

    def cup_debug_callback(self, msg):
        values = {}

        for key, value_text in DEBUG_FIELD_RE.findall(msg.data):
            try:
                values[key] = float(value_text)
            except ValueError:
                continue

        self.debug_values = values
        self.last_debug_time = time.monotonic()

    def debug_is_fresh(self):
        return (
            self.last_debug_time > 0.0
            and time.monotonic() - self.last_debug_time <= self.debug_timeout_sec
        )

    def rule_target_is_fresh(self):
        return (
            self.latest_rule_time > 0.0
            and time.monotonic() - self.latest_rule_time <= self.rule_target_timeout_sec
        )

    def positions_for_source(self, source):
        if source == "rule":
            positions = self.latest_rule_positions
        elif source == "target":
            positions = self.latest_rule_positions
        elif source == "fallback":
            positions = self.latest_fallback_positions
        elif source == "policy":
            positions = self.policy_positions
        else:
            positions = self.latest_real_positions

        if positions is None:
            return None

        return list(positions)

    def source_positions(self, preferred_source):
        fallback_orders = {
            "real": ("real", "fallback", "rule"),
            "rule": ("rule", "real", "fallback"),
            "target": ("rule", "real", "fallback"),
            "fallback": ("fallback", "real", "rule"),
            "policy": ("policy", "real", "fallback", "rule"),
        }

        for source in fallback_orders.get(preferred_source, ("real", "fallback", "rule")):
            positions = self.positions_for_source(source)

            if positions is not None:
                return positions

        return None

    def policy_ready(self):
        return (
            self.cup_status in self.active_statuses
            and self.debug_is_fresh()
            and self.source_positions(self.policy_start_source) is not None
        )

    def clamp(self, value, lower, upper):
        return max(lower, min(upper, value))

    def clamp_joint_positions(self, positions):
        positions[0] = self.clamp(
            positions[0],
            math.radians(CYLINDRICAL_VISION_PARAMS["base_min_deg"]),
            math.radians(CYLINDRICAL_VISION_PARAMS["base_max_deg"]),
        )
        positions[1] = self.clamp(
            positions[1],
            math.radians(CYLINDRICAL_VISION_PARAMS["shoulder_min_deg"]),
            math.radians(self.shoulder_max_deg),
        )
        positions[2] = self.clamp(
            positions[2],
            math.radians(CYLINDRICAL_VISION_PARAMS["elbow_min_deg"]),
            math.radians(CYLINDRICAL_VISION_PARAMS["elbow_max_deg"]),
        )
        positions[3] = self.clamp(
            positions[3],
            math.radians(CYLINDRICAL_VISION_PARAMS["wrist_flex_min_deg"]),
            math.radians(CYLINDRICAL_VISION_PARAMS["wrist_flex_max_deg"]),
        )
        return positions

    def value_for_feature(self, column, target_positions):
        if column in self.debug_values:
            return self.debug_values[column], True

        if column == "gripper_target_rad" and len(target_positions) >= 6:
            return target_positions[5], True

        for prefix, positions in (
            ("real", self.latest_real_positions),
            ("target", target_positions),
        ):
            if column.startswith(f"{prefix}_j"):
                index_text = column[len(prefix) + 2:]

                if index_text.isdigit():
                    index = int(index_text) - 1

                    if positions is not None and 0 <= index < len(positions):
                        return positions[index], True

        if column.startswith("cmd_delta_j"):
            index_text = column[len("cmd_delta_j"):]

            if index_text.isdigit():
                index = int(index_text) - 1

                if (
                    self.latest_real_positions is not None
                    and 0 <= index < len(target_positions)
                    and 0 <= index < len(self.latest_real_positions)
                ):
                    return (
                        target_positions[index] - self.latest_real_positions[index],
                        True,
                    )

        return 0.0, False

    def make_features(self, target_positions):
        features = []

        for status in self.status_names:
            features.append(1.0 if self.cup_status == status else 0.0)

        for column in self.feature_columns:
            value, valid = self.value_for_feature(column, target_positions)
            features.append(float(value) if valid else 0.0)
            features.append(1.0 if valid else 0.0)

        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def predict_action(self, feature_positions):
        features = self.make_features(feature_positions)
        features = (features - self.x_mean) / self.x_std

        with torch.no_grad():
            pred_norm = self.model(features)

        pred = pred_norm * self.y_std + self.y_mean
        action = {
            name: float(value)
            for name, value in zip(self.target_columns, pred.squeeze(0).tolist())
        }
        return self.scale_action(action)

    def scale_action(self, action):
        scaled = {}

        for name, value in action.items():
            scale = self.action_scale

            if name == "action_delta_theta":
                scale *= self.theta_action_scale
            elif name == "action_delta_radius":
                scale *= self.radius_action_scale

            scaled[name] = value * scale

        return scaled

    def clipped_delta(self, name, value):
        limits = {
            "action_delta_theta": self.max_delta_theta_rad,
            "action_delta_radius": self.max_delta_radius_m,
            "action_delta_joint2": self.max_delta_joint2_rad,
            "action_delta_joint3": self.max_delta_joint3_rad,
            "action_delta_gripper": self.max_delta_gripper_rad,
        }
        limit = limits.get(name)

        if limit is None or limit <= 0.0:
            return value

        return self.clamp(value, -limit, limit)

    def warn_ik_throttled(self, text):
        now = time.monotonic()

        if now - self.last_ik_warn_time < 1.0:
            return

        self.get_logger().warn(text)
        self.last_ik_warn_time = now

    def current_cylindrical_pose(self, positions):
        pose = self.kinematics.cylindrical_from_joint_degs(
            math.degrees(positions[0]),
            math.degrees(positions[1]),
            math.degrees(positions[2]),
        )

        radius = pose.radius
        theta_deg = pose.theta_deg
        z = self.debug_values.get("servo_z", pose.z)

        if abs(z) < 1e-6:
            z = pose.z

        return radius, theta_deg, z

    def apply_cylindrical_action(self, base_positions, action):
        positions = list(base_positions)
        radius, theta_deg, z = self.current_cylindrical_pose(positions)
        delta_theta_rad = self.clipped_delta(
            "action_delta_theta",
            action.get("action_delta_theta", 0.0),
        )
        radius += self.clipped_delta(
            "action_delta_radius",
            action.get("action_delta_radius", 0.0),
        )
        theta_deg += math.degrees(delta_theta_rad)

        solution = self.kinematics.cylindrical_ik(radius, theta_deg, z)

        if solution is None:
            self.warn_ik_throttled(
                f"fine-align policy cylindrical IK failed: {self.kinematics.last_error}"
            )
            positions[0] += delta_theta_rad
            return self.clamp_joint_positions(positions)

        positions[0] = math.radians(solution.base_deg)
        positions[1] = math.radians(solution.joint2_deg)
        positions[2] = math.radians(solution.joint3_deg)
        positions[3] = math.radians(solution.wrist_flex_deg)

        if "action_delta_gripper" in action and not self.freeze_gripper:
            positions[5] += self.clipped_delta(
                "action_delta_gripper",
                action["action_delta_gripper"],
            )

        return self.clamp_joint_positions(positions)

    def apply_joint_action(self, base_positions, action):
        positions = list(base_positions)
        target_map = {
            "action_delta_theta": 0,
            "action_delta_joint2": 1,
            "action_delta_joint3": 2,
            "action_delta_gripper": 5,
        }

        for name, value in action.items():
            index = target_map.get(name)

            if index is None:
                continue

            if name == "action_delta_gripper" and self.freeze_gripper:
                continue

            positions[index] += self.clipped_delta(name, value)

        if self.maintain_wrist_level:
            wrist_deg = self.kinematics.wrist_flex_deg(
                math.degrees(positions[1]),
                math.degrees(positions[2]),
            )
            positions[3] = math.radians(wrist_deg)

        return self.clamp_joint_positions(positions)

    def apply_action(self, base_positions, action):
        if "action_delta_radius" in action:
            return self.apply_cylindrical_action(base_positions, action)

        return self.apply_joint_action(base_positions, action)

    def publish_target(self, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = positions
        self.target_pub.publish(msg)

    def publish_debug(self, mode, action=None, reason=""):
        now = time.monotonic()

        if now - self.last_debug_publish_time < 0.25:
            return

        parts = [
            f"mode={mode}",
            f"status={self.cup_status}",
            f"reason={reason}",
            f"debug_fresh={self.debug_is_fresh()}",
            f"rule_fresh={self.rule_target_is_fresh()}",
        ]

        if action:
            parts.extend(
                f"{name}={value:.8f}"
                for name, value in action.items()
            )

        msg = String()
        msg.data = ", ".join(parts)
        self.debug_pub.publish(msg)
        self.last_debug_publish_time = now

    def forward_rule_based(self, reason):
        if self.latest_rule_positions is None or not self.rule_target_is_fresh():
            self.publish_debug("idle", reason=f"{reason}:no_fresh_rule_target")
            return

        self.policy_positions = None
        self.publish_target(self.latest_rule_positions)
        self.publish_debug("rule_based", reason=reason)

    def loop(self):
        if self.cup_status in self.forward_statuses:
            self.forward_rule_based("forward_status")
            return

        if self.cup_status not in self.active_statuses:
            self.policy_positions = None
            self.publish_debug("idle", reason="inactive_status")
            return

        if not self.policy_ready():
            if self.fallback_to_rule_based:
                self.forward_rule_based("policy_not_ready")
            else:
                self.publish_debug("idle", reason="policy_not_ready")
            return

        start_positions = self.source_positions(self.policy_start_source)

        if self.policy_positions is None or self.mode == "one_step":
            self.policy_positions = list(start_positions)

        feature_positions = self.source_positions(self.policy_feature_source)

        if feature_positions is None:
            feature_positions = list(self.policy_positions)

        action = self.predict_action(feature_positions)
        next_positions = self.apply_action(self.policy_positions, action)

        if self.mode == "integrated":
            self.policy_positions = list(next_positions)
        else:
            self.policy_positions = list(start_positions)

        self.publish_target(next_positions)
        self.publish_debug("policy", action=action, reason=self.mode)


def main(args=None):
    rclpy.init(args=args)
    node = FineAlignPolicyGate()

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
