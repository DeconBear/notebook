# Lesson 05 — Launch（笔记: docs/05-Launch一键启动.md）
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """启动带参数的 param_talker，可用命令行覆盖参数。"""
    return LaunchDescription([
        DeclareLaunchArgument(
            'message_prefix',
            default_value='启动参数',
            description='发布消息前缀',
        ),
        DeclareLaunchArgument(
            'publish_period',
            default_value='1.0',
            description='发布周期（秒）',
        ),
        Node(
            package='py_pubsub',
            executable='param_talker',
            name='param_talker',
            output='screen',
            parameters=[{
                'message_prefix': LaunchConfiguration('message_prefix'),
                'publish_period': ParameterValue(
                    LaunchConfiguration('publish_period'),
                    value_type=float,
                ),
            }],
        ),
        Node(
            package='py_pubsub',
            executable='listener',
            name='listener',
            output='screen',
        ),
    ])
