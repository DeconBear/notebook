from setuptools import find_packages, setup

package_name = 'py_learning'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer="ROS2 Humble Notes",
    maintainer_email='ros2-humble-notes@users.noreply.github.com',
    description='Lessons 06-09 learning nodes',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'sensor_status_publisher = py_learning.sensor_status_publisher:main',
            'sensor_status_subscriber = py_learning.sensor_status_subscriber:main',
            'set_led_server = py_learning.set_led_server:main',
            'set_led_client = py_learning.set_led_client:main',
            'tf_broadcaster = py_learning.tf_broadcaster:main',
            'tf_listener = py_learning.tf_listener:main',
            'cmd_vel_publisher = py_learning.cmd_vel_publisher:main',
        ],
    },
)
