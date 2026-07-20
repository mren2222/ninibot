# Third-party notices

Ninibot includes or interoperates with the following third-party projects.
Their names and trademarks remain the property of their respective owners.

## SO-101 mechanical assets

The SO-101 mesh files under `src/arm_controller/urdf/assets/` and the arm
geometry represented by `src/arm_controller/urdf/so101.urdf` are based on the
open-source SO-ARM100 / SO-101 project by TheRobotStudio:

- Source: https://github.com/TheRobotStudio/SO-ARM100
- License: Apache License 2.0

The Ninibot URDF contains project-specific frame, camera-mount, limit, and
grippertip adjustments.

## Hugging Face LeRobot

The real-arm motor and calibration adapters import SO follower APIs from
Hugging Face LeRobot:

- Source: https://github.com/huggingface/lerobot
- License: Apache License 2.0

LeRobot is not vendored in this repository and must be installed separately for
real-hardware operation.

