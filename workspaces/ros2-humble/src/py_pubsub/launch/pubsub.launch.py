# Lesson 05 — Launch（笔记: docs/05-Launch一键启动.md）
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """一次启动 talker + listener。"""
    return LaunchDescription([
        Node(
            package='py_pubsub',
            executable='talker',
            name='talker',
            output='screen',
        ),
        Node(
            package='py_pubsub',
            executable='listener',
            name='listener',
            output='screen',
        ),
    ])
