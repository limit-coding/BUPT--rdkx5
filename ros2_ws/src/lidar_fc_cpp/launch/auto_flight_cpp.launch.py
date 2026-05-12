"""
Livox MID360 -> FAST_LIO -> C++相对定位 -> C++飞控串口桥接
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    livox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('livox_ros_driver2'),
                'launch_ROS2',
                'msg_MID360_launch.py',
            )
        )
    )

    fast_lio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('fast_lio'),
                'launch',
                'mapping.launch.py',
            )
        ),
        launch_arguments={'rviz': 'false'}.items(),
    )

    mid360_xy_node = Node(
        package='lidar_fc_cpp',
        executable='mid360_xy_cpp',
        name='mid360_xy_node',
        output='screen',
    )

    relative_pose_node = Node(
        package='lidar_fc_cpp',
        executable='relative_pose_cpp',
        name='relative_pose_node',
        output='screen',
    )

    fc_bridge_node = Node(
        package='lidar_fc_cpp',
        executable='fc_bridge_cpp',
        name='fc_bridge_node',
        output='screen',
        parameters=[{
            'serial_port': '/dev/ttyFC',
            'baudrate': 115200,
            'send_freq': 20.0,
        }],
    )

    qr_detector_node = Node(
        package='lidar_fc_cpp',
        executable='qr_detector_cpp',
        name='qr_detector_cpp',
        output='screen',
        parameters=[{
            'image_topic': '/image',
            'confirm_frames': 3,
        }],
    )

    return LaunchDescription([
        livox_launch,
        TimerAction(period=2.0, actions=[fast_lio_launch]),
        TimerAction(period=4.0, actions=[
            mid360_xy_node,
            relative_pose_node,
            fc_bridge_node,
            qr_detector_node,
        ]),
    ])
