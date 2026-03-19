# Camera 2D LiDAR Calibration Library

This is a camera and 2D LiDAR calibration library that computes the camera's intrinsic matrix and distortion coefficients as well as the SE(2) transformation from 2D LiDAR frame to the camera frame. 

A set of example ROS bags are made available in this [Google Drive folder](https://drive.google.com/drive/folders/1pZpSstLPTSJcQ-2KaaZFc8NeBMKuVD_V?usp=drive_link). 
We also provide a useful tool that extracts images and point clouds in [this package](https://github.com/ACFR-RPG/ros2_bag_to_image). 
This package will be used regularly in these calibratoin processes to obtain the sensor measurements. 

# Camera Intrinsic Calibration

The camera intrinsic calibration script is [here](https://github.com/ACFR-RPG/camera_2d_lidar_calibration/blob/main/camera_2d_lidar_calibration/cam_intrinsic.py). It reads in all the pictures in a user-designated folder and conducts camera intrinsic calibration, expecting all the input images to contain a fully visible checkerboard with a known size and dimension. 

## Dependencies

This camera intrinsic calibration script relies on very few packages, and all of them should have already been installed with python and [ROS2](https://docs.ros.org/en/jazzy/Installation.html) (assuming full installation). 
However, since this is not a ROS dependent package, we still explain the following required dependencies: 
- [OpenCV](https://pypi.org/project/opencv-python/) - You should not need to do this as OpenCV should have already been installed via the [`cv_bridge`](https://index.ros.org/p/cv_bridge/) package within ROS. 
- [numpy](https://numpy.org/install/)

## Experiment Procedure

This camera intrinsic calibration method follows the procedure detailed below:

1. A square checkerboard pattern is within full-view of the camera, and supported by a wall (a hard surface). 
A copy of this checkerboard is provided in `checkerboard.pdf`. You need to know the dimensions of this checkerboard, specifically the checkerboard square's size and the layout of the squares. You will need to adjust the script [`cam_intrinsic.py`](https://github.com/ACFR-RPG/camera_2d_lidar_calibration/blob/main/camera_2d_lidar_calibration/cam_intrinsic.py#L31) to reflect that. 
2. Record a ROS bag where you hold the checkerboard in the view of the camera and move the board around. You should pause at different poses to make sure the camera capture a clear image of the checkerboard, and make sure that the whole board is within the camera's field of view. An example bag is provided [here](https://drive.google.com/drive/folders/1pZpSstLPTSJcQ-2KaaZFc8NeBMKuVD_V?usp=drive_link). 
3. Use the [ROS bag image extraction tool](https://github.com/ACFR-RPG/ros2_bag_to_image) to obtain all the images in the bag, and select at minimum 10 images out of them. Put them in a separate folder. 
4. Run the [camera calibration script](https://github.com/ACFR-RPG/camera_2d_lidar_calibration/blob/main/camera_2d_lidar_calibration/cam_intrinsic.py) with the `image_dir` input parameter pointing at the folder where you placed your selected images. Make sure you use the correct checkerboard properties within the script. 
5. Obtain the camera extrinsics from the print out. 

## Running the script

Run at the top directory level of this package folder:
```
python camera_2d_lidar_calibration/cam_intrinsic.py <path/to/image/folder>
```

## TODOs

Within the [`cam_intrinsic.py`](https://github.com/ACFR-RPG/camera_2d_lidar_calibration/blob/main/camera_2d_lidar_calibration/cam_intrinsic.py#L31) script, you are expected to update the properties of the checkerboard you use in your calibration process, i.e. use the width and height of checkerboard vertices and size of the blocks that matches with the actual board. 


# Camera 2D LiDAR Calibration

The camera LiDAR extrinsic calibration script is [here](https://github.com/ACFR-RPG/camera_2d_lidar_calibration/blob/main/camera_2d_lidar_calibration/cam_lidar_2d_icp.py). It reads in all the images and point clouds from two user-designated folders to calibrate for the relative 2D/SE(2) transformation between the camera and the 2D LiDAR using ICP. It expects all the input images to contain a fully visible checkerboard with a known size and dimension, as well as the majority of the checkerboard wall in the camera view. 

## Installation and Dependencies

To install, run the following code:
```
mkdir git
cd git
git clone https://github.com/ACFR-RPG/camera_2d_lidar_calibration.git
cd camera_2d_lidar_calibration
pip install -e . # install python dependencies
```

This script depends on the following packages: 
- [OpenCV](https://pypi.org/project/opencv-python/) - You should not need to do this as OpenCV should have already been installed via the [`cv_bridge`](https://index.ros.org/p/cv_bridge/) package within ROS. 
- [Open3D](https://pypi.org/project/open3d/)
- [numpy](https://numpy.org/install/)
- [scipy](https://pypi.org/project/scipy/)
- [tkinter](https://docs.python.org/3/library/tkinter.html) - built into python
- [scikit-learn](https://scikit-learn.org/stable/install.html)
- [matplotlib](https://matplotlib.org/stable/install/index.html)

## Running

Run at the top directory level of this package folder:
```
python camera_2d_lidar_calibration/cam_lidar_2d_icp.py <path/to/image/folder> <path/to/cloud/folder>
```

## Assumptions and Environment Setup

This camera LiDAR extrinsic calibration method relies on the following environment setup and assumptions:

1. A square checkerboard pattern is within full-view of the camera, and supported by a wall with straight edges. A copy of this checkerboard is provided in `checkerboard.pdf`. The software does not necessarily require the dimensions of the wall, but it needs dimensions of the checkerboard; specifically the checkerboard square's sizes and the total number of squares.
2. Corners on the checkerboard will be used to determine the pose of the checkerboard, and in turn estimate a horizontal line that can represent the wall. Any line defined by a row of corners should be parallel with the ground plane and orthogonal to the edges of the wall.
4. The checkerboard is orthogonal to the ground plane.
5. The LiDAR is scanning the world in a plane that is parallel to the ground. The LiDAR can be placed at any height, so long as its rays intersect with the wall.
6. The left and right edges of the wall should be visible in the calibration image, and are discernible from the surroundings. However this is not a hard requirement, but a recommendation. 

To see a typical example that satisfies the assumptions above, refer to the following image corresponding to TurtleBot3 Burger. Its 2D LiDAR is located at the top of the robot and scans the world in a plane parallel to the ground.

<p align="center">
<img src="readme_pictures/thumbnail_IMG_5137.jpg" width="400">
</p>

In this example, both the checkerboard and the left/right edges of the wall are within full-view.

<p align="center">
<img src="readme_pictures/camera_pov.png" height="200">
<img src="readme_pictures/lidar_scans.png" height="200">
</p>

<p align="center">
Camera POV (left) and RViz2 Visualisation of LiDAR Scans (right). In the LiDAR scans, the bottom three sharp lines correspond to the boxes, and the top line corresponds to the wall.
</p>

## Calibration Routine

After setting up the environment, the following routine will produce the required ROS bags that will be fed into the library.

1. Obtain the intrinsic and distortion parameters of the camera through any standard camera calibration method, e.g. the camera intrinsic calibration process above. Ensure the units of these parameters agree with the units of the checkerboard. We recommend using meter (m) as the standard unit.
2. Setup the camera in an environment where you can recognise some features or structure in the LiDAR data (e.g. the boxes surrounding the TurtleBot in the picture above). This setup will assist in the calibration process later. 
3. Start recording a ROS bag that listens to both the camera image topic `/camera/image_raw` and LiDAR topic `/pointcloud2d`. 
4. Set up the checkerboard wall, making sure the whole checkerboard is visible in the camera view and the wall is vertical to the ground plane. Keep it stationary for a few seconds. 
5. Move the wall and checkerboard to a different position, making sure the whole checkerboard is visible in the camera view and the wall is vertical to the ground plane. Keep it stationary for a few seconds. 
6. Repeat step 5 about 5-10 times. 
7. Stop recording the ROS bag, and extract all the images and point clouds from the bag using scripts provided in [this package](https://github.com/ACFR-RPG/ros2_bag_to_image). For each wall position collected in the dataset, select a pair of corresponding camera view and the LiDAR scan (matched using timestamps, for example), and put the images and point clouds in two separate folders. 
8. Run the camera LiDAR calibration script, specifying the `image_dir` and `laser_dir` arguments, which correspond to `<path/to/image/folder>` and `<path/to/cloud/folder>` respectively. Make sure the camera intrinsic and distortion parameters [here](https://github.com/ACFR-RPG/camera_2d_lidar_calibration/blob/main/camera_2d_lidar_calibration/cam_lidar_2d_icp.py#L84) and checkerboard parameters [here](https://github.com/ACFR-RPG/camera_2d_lidar_calibration/blob/main/camera_2d_lidar_calibration/cam_lidar_2d_icp.py#L92) agree with your experiment setup.  
9. Use the interactive GUI to confirm the checkerboard detection and select corresponding LiDAR points for each image pair. 
10. Obtain the calibration results in program print out. 

Alternatively, you can also record a separate ROS bag for 10-20 seconds for each checkerboard pose, like what we did in the example dataset [here](https://drive.google.com/drive/folders/1pZpSstLPTSJcQ-2KaaZFc8NeBMKuVD_V?usp=drive_link). 
You will need to run the image and point cloud extraction script for each bag in this case, and each bag will yield you a pair of corresponding camera view and point cloud. 

## TODOs

Within the [`cam_lidar_2d_icp.py`](https://github.com/ACFR-RPG/camera_2d_lidar_calibration/blob/main/camera_2d_lidar_calibration/cam_lidar_2d_icp.py) script, you are expected to update the properties of the checkerboard you use in your calibration process [here](https://github.com/ACFR-RPG/camera_2d_lidar_calibration/blob/main/camera_2d_lidar_calibration/cam_lidar_2d_icp.py#L92), i.e. use the width and height of checkerboard vertices and size of the blocks that matches with the actual board, 
as well as use the correct camera intrinsic by adjusting the parameters [here](https://github.com/ACFR-RPG/camera_2d_lidar_calibration/blob/main/camera_2d_lidar_calibration/cam_lidar_2d_icp.py#L84). 

We further provide an initial transformation to assist ICP, if needed. 
The transformation can be edited [here](https://github.com/ACFR-RPG/camera_2d_lidar_calibration/blob/yw-vis-cleanup/camera_2d_lidar_calibration/cam_lidar_2d_icp.py#L180). 
This is not mandatory. You are welcome to keep the `initial_tf` as identity if your ICP result is good. 

e## Interactive Interfac

After running, an interface will appear with instructions on selecting 2D LiDAR points that represent the wall containing the checkerboard pattern. Note that the interface appears for one ROS bag at a time, and all 2D points of the LiDAR for each bag are accumulated for each interface. To select the points, first change the sliders that control the starting and ending indices of scans from the ROS bag to desired values. Then, use `matplotlib`'s zoom feature, enabled by default, to zoom into a particular region and click `Select Points`. Once finished, click `Done` and repeat this for the other ROS bags. To see this process, watch the video below.

<div align="center">
  <video src="readme_pictures/2026_mtrx5700_assignment2_cameralidarcalibration.mp4" width="400"></video>
</div>



<p align="center">
<img src="readme_pictures/lidar_2d_selection_menu.png" height="400">
<img src="readme_pictures/lidar_2d_selection.png" height="400">
</p>

<p align="center">
<img src="readme_pictures/lidar_2d_selection_red.png" height="400">
<img src="readme_pictures/lidar_2d_selection_zoom_out.png" height="400">
</p>

After selecting the 2D  scans, another interface will appear to match the 2D LiDAR points with the camera frame for each ROS bag. This interface will contain the relevant instructions on selecting vertical lines that correspond to the edges of the wall, and works similarly to the previous interface. In the current implementation, at least one endpoint of each vertical line must be in the zoomed window to be selected.

<p align="center">
<img src="readme_pictures/wall_edge_menu.png" height="400">
<img src="readme_pictures/wall_edge_selection.png" height="400">
</p>

After following these instructions, plots will appear for each bag that represent the fitted lines corresponding to the wall.

<p align="center">
<img src="readme_pictures/lidar_detected_wall.png" height="400">
<img src="readme_pictures/camera_detected_wall.png" height="400">
</p>

As a sanity check, a set of test points from the camera frame are projected into the LiDAR frame.

<p align="center">
<img src="readme_pictures/test_points.png" height="400">
<img src="readme_pictures/transformed_points.png" height="400">
</p>

The resulting calibration is saved in `~/ros2_ws/src/camera_2d_lidar_calibration/results/camera_2d_lidar_calibration_result_{timestamp}/camera_2d_lidar_calibration_transformation_result_{timestamp}.txt`, alongside any relevant data. 

Evidently, the transformed points are close albeit can differ from any individual ROS bag. Recording more ROS bags and selecting the LiDAR points more carefully will improve the accuracy.

<!-- ## How It Works -->

<!-- After selecting the 2D LiDAR points, the RANSAC algorithm from scikit-learn (https://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html) is used to robustly find the wall line when outliers can exist. This can fail if the gradient is infinite, so the axis with smallest range is used as the dependent variable.

The vertical edges are found using a Probabilistic Hough Transform, followed by a filtering by the line's gradient to select the candidate vertical lines. After selecting the lines, the first principal component is computed using singular value decomposition, which computes the line that minimises the L2-norm to all lines' endpoints. This new line is used as a single representative for a vertical edge of the wall.

After computing the vertical edges, the centre horizontal line of the checkerboard is extended in the camera frame until it intersects the two planes spanned by each vertical edge and the normal vector of the checkerboard. With this procedure, the camera frame positions of the edges of the wall are heuristically estimated using the vertical lines.

After finding these two positions of the wall's edges in the camera frame and the LiDAR frame, this forms two correspondences between the frames. Computing these correspondences across all ROS bags, the SE(3) transformation and scaling is computed using `cv2.estimateAffine3D(wall_edge_camera_frame_points, wall_edge_lidar_points, force_rotation=True)`.
 -->
