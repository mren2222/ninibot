import math

import numpy as np

from ninibot.task_params import PRESS_BUTTON_PARAMS
from ninibot.skills.press_button.config import PressButtonConfig
from ninibot.skills.press_button.math_utils import gripper_percent_to_rad
from ninibot.skills.press_button.ros_params import load_press_button_config


class FakeParameter:
    def __init__(self, value):
        self.value = value


class FakeNode:
    PARAMETER_DEFAULTS = PRESS_BUTTON_PARAMS

    def __init__(self, values):
        self.values = values

    def get_parameter(self, name):
        return FakeParameter(self.values[name])


def normalize_camera_source(value):
    return ("camera", value)


def resolve_urdf_path(value):
    return value or "/resolved/so101.urdf"


def test_config_normalizes_ros_parameter_values():
    values = PRESS_BUTTON_PARAMS.copy()
    values.update(
        {
            "camera_device": "",
            "camera_index": 7,
            "search_pan_min_deg": 90.0,
            "search_pan_max_deg": 10.0,
            "search_pan_speed_deg_s": -15.0,
            "ik_joint4_min_deg": 60.0,
            "ik_joint4_max_deg": 30.0,
            "prep_trajectory_duration_sec": -1.0,
            "fine_align_max_joint_step_deg": -3.0,
            "fine_align_marker_samples": 0,
            "press_cycles": 0,
            "joint6_percent": 150.0,
        }
    )

    config = PressButtonConfig.from_values(
        values,
        normalize_camera_source,
        resolve_urdf_path,
    )

    assert config.camera_source == ("camera", 7)
    assert config.urdf_path == "/resolved/so101.urdf"
    assert config.search_pan_min_deg == 10.0
    assert config.search_pan_max_deg == 90.0
    assert config.search_pan_speed_deg_s == 15.0
    assert config.ik_joint4_min_deg == 30.0
    assert config.ik_joint4_max_deg == 60.0
    assert config.prep_trajectory_duration_sec == 0.0
    assert config.fine_align_max_joint_step_rad == 0.0
    assert config.fine_align_marker_samples == 1
    assert config.press_cycles == 1
    assert config.joint6_percent == 100.0
    assert math.isclose(config.joint6_rad, gripper_percent_to_rad(100.0))
    assert np.allclose(
        config.target_marker_offset_xyz,
        [
            PRESS_BUTTON_PARAMS["target_x_offset"],
            PRESS_BUTTON_PARAMS["target_y_offset"],
            PRESS_BUTTON_PARAMS["target_z_offset"],
        ],
    )


def test_load_press_button_config_reads_declared_parameter_names():
    values = PRESS_BUTTON_PARAMS.copy()
    values["camera_device"] = "/dev/video9"
    node = FakeNode(values)

    config = load_press_button_config(
        node,
        normalize_camera_source,
        resolve_urdf_path,
    )

    assert config.camera_source == ("camera", "/dev/video9")
    assert config.debug_image_topic == PRESS_BUTTON_PARAMS["debug_image_topic"]


def test_config_apply_to_sets_node_style_attributes():
    config = PressButtonConfig.from_values(
        PRESS_BUTTON_PARAMS.copy(),
        normalize_camera_source,
        resolve_urdf_path,
    )
    target = type("Target", (), {})()

    config.apply_to(target)

    assert target.target_topic == PRESS_BUTTON_PARAMS["target_topic"]
    assert target.urdf_path == "/resolved/so101.urdf"
