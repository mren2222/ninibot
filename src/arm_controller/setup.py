# flake8: noqa
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'arm_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),

        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),

        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*.urdf')),

        (os.path.join('share', package_name, 'urdf', 'assets'),
            glob('urdf/assets/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mren2222',
    maintainer_email='182375090+mren2222@users.noreply.github.com',
    description='SO-101 kinematics, URDF, motor bridge, and RViz tools for Ninibot.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'keyboard_point = arm_controller.debug.keyboard_point:main',

            'read_arm = arm_controller.motor.read_arm:main',

            'point_ik = arm_controller.kinematics.point_ik:main',

            'grippertip_ik_service = arm_controller.kinematics.grippertip_ik_service:main',

            'gripper_pose_debug = arm_controller.debug.gripper_pose_debug:main',

            'static_joints = arm_controller.debug.static_joints:main',
            
            'motor_driver = arm_controller.motor.motor_driver:main',

            'target_to_rviz = arm_controller.motor.target_to_rviz:main',

        ],
    },
)
