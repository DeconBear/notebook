from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'simple_robot'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*')),
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*')),
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*')),
        (os.path.join('share', package_name, 'models', 'simple_diff_robot'),
            glob('models/simple_diff_robot/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer="ROS2 Humble Notes",
    maintainer_email='ros2-humble-notes@users.noreply.github.com',
    description='Lessons 10-12: URDF, Gazebo, RViz',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [],
    },
)
