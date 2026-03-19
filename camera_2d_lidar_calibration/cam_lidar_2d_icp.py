
# import rclpy
# from rclpy.executors import ExternalShutdownException
# from rclpy.node import Node

import os
import argparse
# from datetime import datetime

import cv2
import cv_bridge
import open3d as o3d

import numpy as np
from numpy import linalg as la
from scipy.spatial.transform import Rotation

import tkinter as tk
from tkinter import ttk
from pathlib import Path

import icp_2d

import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg as tkagg 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model

from gui import SelectPointsInterface, ImageVisInterface

home = Path.home()

def load_images_from_folder(folder):
    images = []
    print("Reading images from directory: " + folder)    
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename))
        print(os.path.join(folder,filename)) # printing file names to verify the order of them in the list
        if img is not None:
            # image = cv2.rotate(img, cv2.ROTATE_180)
            # images.append(image)
            images.append(img)
    return images

def load_clouds_from_folder(folder):
    clouds = []
    print("Reading point clouds from directory: " + folder)    
    for filename in sorted(os.listdir(folder)):
        pcd = o3d.io.read_point_cloud(os.path.join(folder,filename))
        print(os.path.join(folder,filename)) # printing file names to verify the order of them in the list
        # print(pcd) 
        if len(pcd.points) > 0:
            clouds.append(pcd)
    return clouds

def draw(img, corners, imgpts):
    # Draw a 3D axis at the OpenCV origin of a checkerboard, for introspection
    corner = tuple(corners[0].ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0,0,255), 2)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 2)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (255,0,0), 2)
    return img

def main():
  parser = argparse.ArgumentParser(description="Calibrate image and laser extrinsics from a collection of checkerboard images and laser scans.")
  parser.add_argument("image_dir", help="Image directory.")
  parser.add_argument("laser_dir", help="Laser directory.")
  args = parser.parse_args()

  image_dir = args.image_dir
  laser_dir = args.laser_dir

  # Load images and pcb clouds from file, after they are extracted from a rosbag and selected for calibration
  # They have to have one to one correspondences - that is usually true when they are ordered in each folder correctly
  # The load functions will have print outs for order verification
  images = load_images_from_folder(image_dir)
  lasers = load_clouds_from_folder(laser_dir)

  assert len(images) == len(lasers), "Images and lasers length mismatch!"

  # Camera intrinsics, from camera calibration process
  # TODO: Change this into the camera intrinsic calibration results
  camera_k = np.array([(505.29124569, 0.0, 322.19927863), (0.0, 504.55350189, 236.33609581), (0.0, 0.0, 1.0)])
  camera_dist = np.array([(0.18421732, -0.31277364, -0.00041421, 0.00039006, -0.07993784)])

  # termination criteria, for aligning checkerboard corners onto an image
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  # Checkerboard shape, example: 7*10 in checkerboard blocks, 20 mm in checkerboard block size
  # TODO: Change the checkerboard parameters into those that match with the actual board you are using
  checkerboard_height = 8
  checkerboard_width = 11 
  checkerboard_size = 0.019
  checkerboard_points = np.zeros((checkerboard_width*checkerboard_height, 3), np.float32)
  checkerboard_points[:,:2] = np.mgrid[0:checkerboard_height,0:checkerboard_width].T.reshape(-1,2)*checkerboard_size

  # A predefined 3D axis for visualisation, axis length 10 cm
  axis = np.float32([[0.1,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)

  # Collected/extracted camera and LiDAR points in 2D - supposed to be aligned with each other
  camera_points = []
  laser_points = []

  # Now knowing that list 'images' and 'lasers' are of the same length, loop through them at the same time
  # to extract the corresponding 2D line points for alignment
  for i, (this_image, this_laser) in enumerate(zip(images, lasers)):
    # print(i) 
  
    # Extract the checkerboard from the image and extract a line that represent the checkerboard in 2D
    # But first! Undistortion - without undistortion, the projection of linear structure into the 3D space will be distorted too
    # Distortion effect means that a straight line in 3D is not a straight line in the camera view
    # The attempt to extract a straight line on the checkerboard from the camera view will therefore be affected
    h, w = this_image.shape[:2]
    new_camera_k, _ = cv2.getOptimalNewCameraMatrix(camera_k, camera_dist, (w,h), 1, (w,h))
    new_camera_dist = np.zeros((1, 5)) # the distortion coefficients will be zeroes after undistortion
    undistorted_image = cv2.undistort(this_image, camera_k, camera_dist, None, new_camera_k)
    # Now we actually extract the checkerboard using the undistorted image and camera parameters
    gray = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (checkerboard_height, checkerboard_width), None)
    if ret == True:
      corners2 = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), criteria)
      # Find the rotation and translation vectors (pose) between the board and the camera.
      ret_pnp, rvecs, tvecs = cv2.solvePnP(checkerboard_points, corners2, new_camera_k, new_camera_dist)
      # draw checkerboard corners for introspection
      cv2.drawChessboardCorners(undistorted_image, (checkerboard_width, checkerboard_height), corners2, ret)
      # project 3D axis to image plane for introspection
      imgpts, jacobian = cv2.projectPoints(axis, rvecs, tvecs, new_camera_k, new_camera_dist)
      undistorted_image = draw(undistorted_image, corners2, imgpts)

      # # Visualisation - OpenCV
      # cv2.imshow('Image with checkerboard axis', undistorted_image) # Note the direction of the z axis
      # cv2.waitKey(500)

      # Visualisation - Interactive and extract horizontal line
      # This is where a horizontal line is actually estimated based on the checkerboard pose
      visualise_camera_interface = ImageVisInterface(rvecs, tvecs, undistorted_image, camera_points)
      confirmed, camera_points = visualise_camera_interface.run()
      if confirmed == False:
        print("Something wrong with this image?")
        return
      # else:
      #   print(camera_points) # For introspection only
    else: 
      print("Unable to extract a checkerboard - are you using the correct checkerboard parameters?")
      return

    # Extract the line that correspond to the checkerboard from the LiDAR scan
    # Using an interactive interface
    this_laser_points = np.asarray(this_laser.points)
    select_points_interface = SelectPointsInterface(this_laser_points, laser_points)
    laser_points = select_points_interface.run()
    # print(laser_points)

  # Concatenate the list of arrays into long arrays
  all_lidar_points = np.vstack(laser_points)
  all_camera_points = np.vstack(camera_points)

  # Introspection - are the two sets of points, camera_points and laser_points, looking reasonable with each other?
  fig = plt.figure()
  man = plt.get_current_fig_manager()
  man.set_window_title(f"Checkerboard in Image (Green) and in LiDAR (Blue)")
  ax = fig.add_subplot()

  ax.scatter(all_lidar_points[:,0], all_lidar_points[:,1], c='blue', label='All 2D LiDAR Points')
  ax.scatter(all_camera_points[:,0], all_camera_points[:,1], c='green', label='All 2D Camera Points')

  ax.legend()
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_title('Detected Wall - Topdown 2D View of points from Camera and LiDAR')

  plt.show(block=False)
  plt.pause(0.01)

  # Initial transformation to help with the alignment
  # This can be obtained from an initial measurement. 
  # A good ICP system should not need this initialisation, but sometimes it helps. 
  # TODO: If you need to, you may put your own desired/measured transformation into the initial_tf 
  initial_tf = np.identity(3)
  # initial_tf[0, 2] = -0.1
  laser_points_initial_correction = []
  for this_laser_line in laser_points: 
      ones = np.ones((this_laser_line.shape[0], 1))
      lidar_points_h = np.hstack((this_laser_line, ones))
      transformed_h = (initial_tf @ lidar_points_h.T).T
      laser_points_initial_correction.append(transformed_h[:, :2])

  # Concatenate the list of arrays into long arrays
  all_lidar_points_initial_corrected = np.vstack(laser_points_initial_correction)

  # Alignment using ICP
  # ICP will compute a pair of rotation and translation to align two point clouds
  # Though the point-to-point distance model in the most common ICP methods assumes point-to-point correspondence
  # in practice it is not always necessary - if you have a sufficient density. 

  # Here, we are aligning 2 point clouds of mostly line structures
  # However, these lines don't necessarily have the same length - 
  # LiDAR lines measure the wall from one edge to another
  # Camera lines are estimated based on checkerboard pose, with no knowledge on the dimension of the wall. 
  # We are therefore only able to align the orientations of these line structure and the distances to them
  # Therefore we need at least 2 pairs of camera/LIDAR lines for the alignment process

  # ICP using Open3D
  # Put into 3D first because of the built-in ICP in Open3D is for 3D point cloud
  zeroes = np.ones((all_lidar_points_initial_corrected.shape[0], 1))
  lidar_points_3d = np.hstack((all_lidar_points_initial_corrected, zeroes))
  zeroes = np.ones((all_camera_points.shape[0], 1))
  camera_points_3d = np.hstack((all_camera_points, zeroes))
  lidar_cloud = o3d.geometry.PointCloud()
  lidar_cloud.points = o3d.utility.Vector3dVector(lidar_points_3d)
  camera_cloud = o3d.geometry.PointCloud()
  camera_cloud.points = o3d.utility.Vector3dVector(camera_points_3d)

  # Define the type of registration:
  type = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)  # "False" means rigid transformation, scale = 1
  # Define the number of iterations
  iterations = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 300)

  result = o3d.pipelines.registration.registration_icp(lidar_cloud, camera_cloud, 0.1, np.identity(4), type, iterations)
  # print(result.transformation)
  icp_result = np.array([
      [result.transformation[0,0], result.transformation[0,1], result.transformation[0,3]],
      [result.transformation[1,0], result.transformation[1,1], result.transformation[1,3]],
      [0,         0,         1]
    ])
  icp_result = icp_result @ initial_tf
  print("\n\nThe final results of 2D ICP using Open3D") 
  print(icp_result)

  # 2D ICP using a custom function
  # We provide 2 2D ICP options in this separate icp_2d.py file
  # Option 1 icp_2d.icp(): combine all points together and run ICP on all of them
  # Option 2 icp_2d.icp_per_line(): the point-to-point matching are conducted per line, instead of combined
  # The benefit of using option 2 is that there will be no confused correspondences across lines, 
  # as these inter-line correspondences should not be physically present
  transformation_history, points = icp_2d.icp_per_line(camera_points, laser_points_initial_correction, 300, 0.1, 1e-7, 1e-7, 50, False)
  # Compound the transformations because of the output (transformation_history) is for each iteration
  tf_total = np.eye(3)
  for tf in transformation_history: 
    tf_homogeneous = np.eye(3)
    tf_homogeneous[:2, :] = tf
    tf_total = tf_total @ tf_homogeneous
  tf_total = tf_total @ initial_tf
  print("\n\nThe final results of 2D ICP using custom function") 
  print(tf_total) 

  # Introspection - the end result of the alignment
  fig = plt.figure()
  man = plt.get_current_fig_manager()
  man.set_window_title(f"Aligned point clouds")
  ax = fig.add_subplot()

  ones = np.ones((all_lidar_points.shape[0], 1))
  lidar_points_h = np.hstack((all_lidar_points, ones))
  transformed_h = (icp_result @ lidar_points_h.T).T
  transformed = transformed_h[:, :2]

  ax.scatter(all_camera_points[:,0], all_camera_points[:,1], c='green', label='All 2D Camera Points')
  ax.scatter(transformed[:,0], transformed[:,1], c='red', label='All 2D LiDAR Points, Open3D')
  ax.scatter(points[:,0], points[:,1], c='blue', label='All 2D LiDAR Points, custom function')

  ax.legend()
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_title('Detected Wall - Aligned')

  plt.show()

if __name__ == '__main__':
    main()