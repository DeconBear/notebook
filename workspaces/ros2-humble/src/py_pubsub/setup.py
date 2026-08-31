from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'py_pubsub'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer="ROS2 Humble Notes",
    maintainer_email='ros2-humble-notes@users.noreply.github.com',
    description='Python Topic / Service / Action / Param / Launch 示例包',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'talker = py_pubsub.talker_node:main',
            'listener = py_pubsub.listener_node:main',
            'add_two_ints_server = py_pubsub.add_two_ints_server:main',
            'add_two_ints_client = py_pubsub.add_two_ints_client:main',
            'fibonacci_action_server = py_pubsub.fibonacci_action_server:main',
            'fibonacci_action_client = py_pubsub.fibonacci_action_client:main',
            'param_talker = py_pubsub.param_talker:main',
        ],
    },
)
