import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    #main_ctrl
    main_ctrl_node = Node(
        package='main',
        executable='main_stop'
    )

    camera_node= Node(
        package='camera',
        executable='animal_enable'
    )

    #tf
    tf_publisher_node = Node(
        package='tf',
        executable='tf_publisher'
    )
    #communication
    controller_node = Node(
        package='communication',
        executable='uart'
    )
    bluetooth_node = Node(
        package='communication',
        executable='bluetooth'
    )

    launch_fast_lio = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('fast_lio'),
                'launch',
                'mid.launch.py'
            ])
        ]),
    
    )

    launch_livox_ros_driver2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('livox_ros_driver2'),
                'launch_ROS2',
                'msg_MID360_launch.py'
            ])
        ]),

    )
        
    ld = LaunchDescription()
    #lidar
    ld.add_action(launch_fast_lio)
    ld.add_action(launch_livox_ros_driver2)


    #camera
    ld.add_action(camera_node)
    
    #communicator
    ld.add_action(controller_node)
    ld.add_action(bluetooth_node)

    #main
    ld.add_action(main_ctrl_node)
    
    return ld