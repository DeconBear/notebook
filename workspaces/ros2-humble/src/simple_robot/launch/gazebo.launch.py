# Lesson 11 — Gazebo + ROS bridge（笔记: docs/11-Gazebo最小仿真.md）
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('simple_robot')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    world = os.path.join(pkg, 'worlds', 'simple_robot.sdf')
    models = os.path.join(pkg, 'models')

    # 让 Gazebo 能找到本包 models/
    set_resource = AppendEnvironmentVariable(
        'IGN_GAZEBO_RESOURCE_PATH',
        models,
    )
    # 兼容部分环境变量名
    set_gz_resource = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        models,
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': f'-r {world}',
        }.items(),
    )

    # ROS Twist <-> Gazebo Twist
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # 双向桥：ROS Twist <-> Gazebo Twist；ROS Odometry <-> Gazebo Odometry
            '/model/simple_diff_robot/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/model/simple_diff_robot/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry',
        ],
        remappings=[
            ('/model/simple_diff_robot/cmd_vel', '/cmd_vel'),
            ('/model/simple_diff_robot/odometry', '/odom'),
        ],
        output='screen',
    )

    return LaunchDescription([
        set_resource,
        set_gz_resource,
        gz_sim,
        bridge,
    ])
