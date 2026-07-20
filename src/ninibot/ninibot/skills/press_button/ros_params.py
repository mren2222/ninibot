from ninibot.skills.press_button.config import PressButtonConfig


def load_press_button_config(node, normalize_camera_source, resolve_urdf_path):
    values = {
        name: node.get_parameter(name).value
        for name in node.PARAMETER_DEFAULTS
    }
    return PressButtonConfig.from_values(
        values,
        normalize_camera_source,
        resolve_urdf_path,
    )

