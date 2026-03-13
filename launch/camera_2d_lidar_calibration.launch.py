import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory("camera_2d_lidar_calibration"), "config", "params.yaml"
    )

    return LaunchDescription([
        Node(
            package="camera_2d_lidar_calibration",
            executable="camera_2d_lidar_calibration",
            name="camera_2d_lidar_calibration",
            output="screen",
            parameters=[config_file]
        )
    ])