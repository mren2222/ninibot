import math

from ninibot.skills.move_cup.config import MoveCupConfig
from ninibot.task_params import MOVE_CUP_PARAMS
import numpy as np


class FakeParameter:
    def __init__(self, value):
        self.value = value


class FakeNode:
    def __init__(self, overrides=None):
        self.params = dict(MOVE_CUP_PARAMS)
        if overrides:
            self.params.update(overrides)

    def get_parameter(self, name):
        return FakeParameter(self.params[name])


def test_move_cup_config_preserves_runtime_param_transforms():
    config = MoveCupConfig.from_node(
        FakeNode(
            {
                'joint_goal_tolerance_deg': 12.0,
                'post_release_clearance_retract_m': -1.0,
                'post_release_clearance_duration_sec': -2.0,
                'target_check_joint_count': 99,
                'target_x_offset': 0.1,
                'target_y_offset': -0.2,
                'target_z_offset': 0.3,
                'target_lock_samples': 0,
                'target_lock_max_std_m': -0.1,
                'target_radius_margin_m': -0.2,
                'search_prep_min_duration_sec': 0.0,
                'search_prep_head_down_duration_sec': 0.0,
                'search_prep_duration_scale': 0.5,
                'search_prep_trajectory_samples': 4,
            }
        )
    )

    assert config.command_topic == MOVE_CUP_PARAMS['command_topic']
    assert math.isclose(config.joint_goal_tolerance, math.radians(12.0))
    assert config.post_release_clearance_retract_m == 0.0
    assert config.post_release_clearance_duration_sec == 0.0
    assert config.target_check_joint_count == 4
    np.testing.assert_allclose(
        config.target_marker_offset_xyz,
        np.asarray([0.1, -0.2, 0.3]),
    )
    assert config.target_lock_samples == 1
    assert config.target_lock_max_std_m == 0.0
    assert config.target_radius_margin_m == 0.0
    assert config.search_prep_min_duration_sec == 0.1
    assert config.search_prep_head_down_duration_sec == 0.1
    assert config.search_prep_duration_scale == 1.0
    assert config.search_prep_trajectory_samples == 16


def test_move_cup_config_apply_to_node_keeps_legacy_fields():
    config = MoveCupConfig.from_node(FakeNode())
    node = type('NodeFields', (), {})()

    config.apply_to_node(node)

    assert node.target_topic == config.target_topic
    assert node.real_joint_states_topic == config.real_joint_states_topic
    assert node.control_rate == config.control_rate
    np.testing.assert_allclose(
        node.target_marker_offset_xyz,
        config.target_marker_offset_xyz,
    )
