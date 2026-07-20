import numpy as np

from ninibot.skills.press_button.fine_align import FineAlignCommandAhead
from ninibot.skills.press_button.fine_align import average_samples
from ninibot.skills.press_button.fine_align import make_step_context


def test_average_samples_returns_mean_target():
    samples = [
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([3.0, 4.0, 5.0]),
    ]

    mean = average_samples(samples)

    assert np.allclose(mean, [2.0, 3.0, 4.0])


def test_command_ahead_accumulates_and_caps():
    controller = FineAlignCommandAhead(
        integral_enable=True,
        max_command_ahead=0.05,
    )

    first, ahead1 = controller.step_target(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        0.03,
    )
    second, ahead2 = controller.step_target(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        0.03,
    )

    assert np.allclose(first, [0.03, 0.0, 0.0])
    assert ahead1 == 0.03
    assert np.allclose(second, [0.05, 0.0, 0.0])
    assert ahead2 == 0.05


def test_command_ahead_resets_when_direction_flips():
    controller = FineAlignCommandAhead(
        integral_enable=True,
        max_command_ahead=0.10,
    )
    controller.step_target([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.03)

    target, ahead = controller.step_target(
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        0.02,
    )

    assert np.allclose(target, [-0.02, 0.0, 0.0])
    assert ahead == 0.02


def test_make_step_context_limits_visual_step_to_max_step():
    controller = FineAlignCommandAhead(
        integral_enable=False,
        max_command_ahead=0.0,
    )

    context = make_step_context(
        target_base_xyz=[0.10, 0.0, 0.0],
        tip_base_xyz=[0.0, 0.0, 0.0],
        fk_tip_xyz=[0.02, 0.0, 0.0],
        command_ahead=controller,
        max_step=0.03,
    )

    assert np.allclose(context.visual_step_target, [0.03, 0.0, 0.0])
    assert np.allclose(context.ik_step_target, [0.05, 0.0, 0.0])
    assert context.distance == 0.10
    assert context.max_step == 0.03

