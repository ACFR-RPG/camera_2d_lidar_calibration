from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'camera_2d_lidar_calibration'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=[]),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        # Install config files
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=['setuptools', 'opencv-python>=4.11.0','tk', 'matplotlib>=3.5.1','scikit-learn>=1.6.1','numpy<2.0'],
    zip_safe=True,
    maintainer='Yiduo',
    maintainer_email='yiduo.wang@sydney.edu.au',
    description='ROS2 Bag to Camera and 2D LiDAR Calibration Package.',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_2d_lidar_calibration = camera_2d_lidar_calibration.camera_2d_lidar_calibration:main',
        ],
    },
)