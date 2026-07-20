from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'audio_controller'

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
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mren2222',
    maintainer_email='182375090+mren2222@users.noreply.github.com',
    description='Audio monitoring nodes for Ninibot water-flow feedback.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'water_flow_monitor = audio_controller.water_flow_monitor_node:main',
        ],
    },
)
