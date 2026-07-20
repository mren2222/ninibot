from ninibot.skills.press_button.controller import PressButtonController


class PressButtonPhaseHandlers(PressButtonController):
    def __init__(self, node=None):
        self.node = node

    def loop(self):
        if self.node is None:
            return None
        observation = self.node.build_controller_observation()
        step = self.step(observation)
        self.node.execute_controller_step(step)
        return step

__all__ = ["PressButtonPhaseHandlers"]
