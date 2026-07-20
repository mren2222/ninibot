# flake8: noqa
from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'vision_controller'
aruco_assets = [path for path in glob('assets/aruco/*') if os.path.isfile(path)]

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/assets/aruco',
            aruco_assets),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mren2222',
    maintainer_email='182375090+mren2222@users.noreply.github.com',
    description='Vision perception helpers for Ninibot.',
    license='Apache-2.0',
)
