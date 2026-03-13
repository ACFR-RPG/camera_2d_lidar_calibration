import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
import rosbag2_py
from sensor_msgs.msg import Image, LaserScan
import os
from rclpy.serialization import deserialize_message
import cv2
import cv_bridge
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg as tkagg 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from datetime import datetime
import laser_geometry
home = Path.home()


class ImageAndScans:
    def __init__(self, image:np.ndarray, scans:list[LaserScan], bag_name:str):
        assert isinstance(image, np.ndarray), "Error: image must be of the form np.ndarray"
        assert isinstance(scans , list), "Error: scans must be a list."
        for scan in scans:
            assert isinstance(scan, LaserScan), "Error: scans must contain LaserScan objects."
        self.bag_name_ = bag_name
        self.image_ = image
        self.scans_ = scans
        self.selected_points_bool_ = False
        self.selected_lidar_pc2_points_ = None
        self.selected_left_lines_bool_ = False
        self.selected_left_lines_ = None
        self.selected_right_lines_bool_ = False
        self.selected_right_lines_ = None
        self.undistorted_image_bool_ = False
        self.undistorted_image_ = None

    def get_bag_name(self):
        return self.bag_name_

    def concatenate_scans_to_points(self, indices=[0]) -> np.ndarray:
        scans = self.get_scans()
        laser_projection = laser_geometry.LaserProjection()
        selected_scans = [laser_projection.projectLaser(scans[i]) for i in indices]
        
        scans_numpy = [pc2.read_points(scan) for scan in selected_scans]
        concatenated_scans = np.concatenate(scans_numpy)
        return concatenated_scans # x,y,z,intensity,index
    
    def set_undistorted_image(self, undistorted_image:np.ndarray):
        self.undistorted_image_ = undistorted_image.copy()
        self.undistorted_image_bool_ = True
 
    def set_selected_lidar_points(self, selected_points:np.ndarray):
        self.selected_lidar_pc2_points_ = selected_points.copy()
        self.selected_points_bool_ = True
    
    def set_selected_left_lines(self, selected_lines:np.ndarray):
        self.selected_left_lines_ = selected_lines.copy()
        self.selected_left_lines_bool_ = True

    def set_selected_right_lines(self, selected_lines:np.ndarray):
        self.selected_right_lines_ = selected_lines.copy()
        self.selected_right_lines_bool_ = True

    def get_image(self) -> np.ndarray:
        return self.image_.copy()
    def get_scans(self) -> list[LaserScan]:
        return self.scans_.copy()
    
    def has_undistored_image(self) -> bool:
        return self.undistorted_image_bool_
    
    def has_selected_points(self)-> bool:
        return self.selected_points_bool_
    
    def has_selected_left_lines(self)-> bool:
        return self.selected_left_lines_bool_
    
    def has_selected_right_lines(self)-> bool:
        return self.selected_right_lines_bool_
    
    def get_undistorted_image(self) -> np.ndarray:
        if self.undistorted_image_bool_:
            return self.undistorted_image_.copy()
        else:
            return None
    
    def get_selected_lidar_points(self) -> np.ndarray:
        if self.selected_points_bool_:
            return self.selected_lidar_pc2_points_.copy()
        else:
            return None
    
    def get_selected_left_lines(self) -> np.ndarray:
        if self.selected_left_lines_bool_:
            return self.selected_left_lines_.copy()
        else:
            return None
        
    def get_selected_right_lines(self) -> np.ndarray:
        if self.selected_right_lines_bool_:
            return self.selected_right_lines_.copy()
        else:
            return None

    def copy(self):
        image_and_scans_copy = ImageAndScans(self.get_image(), self.get_scans(), self.get_bag_name())
        if self.has_undistored_image():
            image_and_scans_copy.set_undistorted_image(self.get_undistorted_image())
        if self.has_selected_points():
            image_and_scans_copy.set_selected_lidar_points(self.get_selected_lidar_points())
        if self.has_selected_left_lines():
            image_and_scans_copy.set_selected_left_lines(self.get_selected_left_lines())
        if self.has_selected_right_lines():
            image_and_scans_copy.set_selected_right_lines(self.get_selected_right_lines())
        return image_and_scans_copy

class SelectLinesInterface:
    """
    Class for selecting lines corresponding to edge of wall.
    """
    def __init__(self, image_and_scans:ImageAndScans):
        assert isinstance(image_and_scans, ImageAndScans), "Error: image_and_scans must be of type ImageAndScans."
        self.root = tk.Tk()
        self.root.title("Camera 2D LiDAR Calibration Menu - Wall Edge Selection")
        self.root.geometry("700x700")
        self.menubar = tk.Menu(self.root)        
        self.app = tk.Frame(self.root)

        self.image_and_scans = image_and_scans.copy()
        
        window_text = tk.Label(self.root,
                            text=f"ROS Bag: {image_and_scans.get_bag_name()}\nSelect lines corresponding to the left and right edges of the perpendicular surface \nby using the Zoom feature followed by clicking 'Select Left Edge' and 'Select Right Edge'.\nOnce finished, click 'Done'.")
        window_text.pack(padx=5, pady=5)

                
        self.button_frame = ttk.Frame(self.root)    
        self.button_frame.pack(side="top", pady=(20,0))

        self.select_left_edge_button = ttk.Button(self.button_frame, text="Select Left Edge Lines")
        self.select_left_edge_button.pack(side="left", padx=25,pady=(0,10), ipadx=20, ipady=20)
        self.select_left_edge_button.bind("<ButtonPress>", self.select_left_edge_lines)

        self.select_right_edge_button = ttk.Button(self.button_frame, text="Select Right Edge Lines")
        self.select_right_edge_button.pack(side="left", padx=25,pady=(0,10), ipadx=20, ipady=20)
        self.select_right_edge_button.bind("<ButtonPress>", self.select_right_edge_lines)

        self.done_button = ttk.Button(self.button_frame, text="Done")
        self.done_button.pack(side="left", padx=25,pady=(0,10), ipadx=20, ipady=20)
        self.done_button.bind("<ButtonPress>", self.done_callback)


        self.image_with_lines, self.vertical_lines, horizontal_lines = get_vertical_and_horizontal_lines(self.image_and_scans.get_undistorted_image())
        self.ax_selected_left_edge_lines = []
        self.ax_selected_right_edge_lines = []
        self.selected_left_lines_indices = None
        self.selected_right_lines_indices = None

        self.add_figure()

        self.add_detected_lines(self.image_and_scans.get_undistorted_image())


    def add_figure(self) -> None:

        self.figure = plt.Figure(figsize=(7, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.chart_type = FigureCanvasTkAgg(self.figure, self.root)
        self.navigation_tool_bar = tkagg.NavigationToolbar2Tk(self.chart_type, self.root)
        self.navigation_tool_bar.zoom()
        self.chart_type.get_tk_widget().pack()
        self.ax.set_title('Detected Vertical Lines')
        self.ax.callbacks.connect('xlim_changed', self.on_xlims_change)
        self.ax.callbacks.connect('ylim_changed', self.on_ylims_change)

    def add_detected_lines(self, img) -> None:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # changes image encoding from RGB to BGR for cv2.imwrite to work correctly
        self.ax.imshow(img)

        for line in self.vertical_lines:
            self.ax.plot((line[0,0], line[1,0]), (line[0,1], line[1,1]),c='blue', linewidth=2.0)
        

    def on_xlims_change(self, event_ax) -> None:
        self.xlims = event_ax.get_xlim()

    def on_ylims_change(self, event_ax) -> None:
        self.ylims = event_ax.get_ylim()

    def clear_selected_left_edge_lines(self) -> None:
        if self.ax_selected_left_edge_lines:
            for ax_line in self.ax_selected_left_edge_lines:
                self.ax.lines.remove(ax_line[0])
        self.ax_selected_left_edge_lines = None
        self.selected_left_lines_indices = None

    def clear_selected_right_edge_lines(self) -> None:
        if self.ax_selected_right_edge_lines:
            for ax_line in self.ax_selected_right_edge_lines:
                self.ax.lines.remove(ax_line[0])
        self.ax_selected_right_edge_lines = None
        self.selected_right_lines_indices = None

    

    def select_left_edge_lines(self, event) -> None:
        self.clear_selected_left_edge_lines()

        bounding_box_points = np.array(np.round([(self.xlims[0], self.ylims[0]), (self.xlims[0], self.ylims[1]), (self.xlims[1], self.ylims[1]), (self.xlims[1],self.ylims[0])]), np.int32)
        
        selected_left_lines_indices = []
        for i, vertical_line in enumerate(self.vertical_lines):
            for vertical_line_boundary_pt in vertical_line:
                dist = cv2.pointPolygonTest(bounding_box_points, (int(vertical_line_boundary_pt[0]), int(vertical_line_boundary_pt[1])), False)
                if dist >= 0:
                    selected_left_lines_indices.append(i)
                    break
        self.selected_left_lines_indices = selected_left_lines_indices
        selected_left_lines = self.vertical_lines[self.selected_left_lines_indices]
        
        self.ax_selected_left_edge_lines = []
        for line in selected_left_lines:
            self.ax_selected_left_edge_lines.append(self.ax.plot((line[0,0], line[1,0]), (line[0,1], line[1,1]),c='red', linewidth=2.0))
        
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    
    def select_right_edge_lines(self, event) -> None:
        self.clear_selected_right_edge_lines()

        bounding_box_points = np.array(np.round([(self.xlims[0], self.ylims[0]), (self.xlims[0], self.ylims[1]), (self.xlims[1], self.ylims[1]), (self.xlims[1],self.ylims[0])]), np.int32)
        
        selected_right_lines_indices = []
        for i, vertical_line in enumerate(self.vertical_lines):
            for vertical_line_boundary_pt in vertical_line:
                dist = cv2.pointPolygonTest(bounding_box_points, (int(vertical_line_boundary_pt[0]), int(vertical_line_boundary_pt[1])), False)
                if dist >= 0:
                    selected_right_lines_indices.append(i)
                    break
        self.selected_right_lines_indices = selected_right_lines_indices
        selected_right_lines = self.vertical_lines[self.selected_right_lines_indices]
        
        self.ax_selected_right_edge_lines = []
        for line in selected_right_lines:
            self.ax_selected_right_edge_lines.append(self.ax.plot((line[0,0], line[1,0]), (line[0,1], line[1,1]),c='green', linewidth=2.0))
        
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def done_callback(self, event) -> None:
        if self.selected_left_lines_indices and self.selected_right_lines_indices:
            self.image_and_scans.set_selected_left_lines(self.vertical_lines[self.selected_left_lines_indices])
            self.image_and_scans.set_selected_right_lines(self.vertical_lines[self.selected_right_lines_indices])
            self.root.destroy()


    def run(self) -> ImageAndScans:
        self.app.mainloop()
        assert self.image_and_scans.has_selected_left_lines() and self.image_and_scans.has_selected_right_lines(), "Error: You must select left and right edges to proceed."
        return self.image_and_scans
            
 

class SelectPointsInterface:
    """
    Class for selecting lidar points corresponding to back wall.
    """

    def __init__(self, image_and_scans:ImageAndScans):
        assert isinstance(image_and_scans, ImageAndScans), "Error: image_and_scans must be of type ImageAndScans."
        
        self.root = tk.Tk()
        self.root.title("Camera 2D LiDAR Calibration Menu - LiDAR 2D Point Selection")
        self.root.geometry("600x700")
        self.menubar = tk.Menu(self.root)        
        self.app = tk.Frame(self.root)

        self.image_and_scans = image_and_scans.copy()
        
        window_text = tk.Label(self.root,
                                            text=f"ROS Bag: {image_and_scans.get_bag_name()}\nUse the slides to choose the starting and ending indices of 2D LiDAR scans respectively. \nSelect 2D LiDAR points corresponding to the perpendicular surface \nby using the Zoom feature followed by clicking 'Select Points'.\nOnce finished, click 'Done'.")
        window_text.pack(padx=5, pady=5)
        
        self.scan_start_slider = tk.Scale(self.root, from_=0, to=len(image_and_scans.get_scans())-1, 
                                     tickinterval=int(len(image_and_scans.get_scans())/5), orient=tk.HORIZONTAL, 
                                     command=self.scan_slider_callback)
        self.scan_start_slider.set(0)
        self.scan_start_slider.pack(ipadx=100)

        self.scan_end_slider = tk.Scale(self.root, from_=0, to=len(image_and_scans.get_scans())-1, 
                                     tickinterval=int(len(image_and_scans.get_scans())/5), orient=tk.HORIZONTAL, 
                                     command=self.scan_slider_callback)
        self.scan_end_slider.set(len(image_and_scans.get_scans())-1)
        self.scan_end_slider.pack(ipadx=100)
        
        self.button_frame = ttk.Frame(self.root)    
        self.button_frame.pack(side="top", pady=(20,0))

        self.select_points_button = ttk.Button(self.button_frame, text="Select Points")
        self.select_points_button.pack(side="left", padx=25,pady=(0,10), ipadx=20, ipady=20)
        self.select_points_button.bind("<ButtonPress>", self.select_points)


        self.done_button = ttk.Button(self.button_frame, text="Done")
        self.done_button.pack(side="left", padx=25,pady=(0,10), ipadx=20, ipady=20)
        self.done_button.bind("<ButtonPress>", self.done_callback)

        self.ax_lidar_points = None
        self.ax_selected_lidar_points = None
        self.scan_indices = np.arange(0,len(image_and_scans.get_scans()))
        self.selected_points_indices = None

        self.add_figure()

        self.reset_and_add_2d_lidar_points()

    
    def reset_and_add_2d_lidar_points(self):
        self.clear_2d_lidar_points()
        self.clear_selected_2d_lidar_points()
        self.all_2d_lidar_points = self.image_and_scans.concatenate_scans_to_points(self.scan_indices)
        self.add_2d_lidar_points(self.all_2d_lidar_points.copy())
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def scan_slider_callback(self,event):
        if self.scan_end_slider.get() < self.scan_start_slider.get():
            self.scan_end_slider.set(self.scan_start_slider.get())

        self.scan_indices = np.arange(self.scan_start_slider.get(),self.scan_end_slider.get()+1)
        
        self.reset_and_add_2d_lidar_points()

    

    def select_points(self, event):
        self.clear_selected_2d_lidar_points()

        points_xy = np.array([[point[0], point[1]] for point in self.all_2d_lidar_points.copy()])

        selected_x_indices = np.logical_and(points_xy[:,0]>= self.xlims[0], points_xy[:,0] <= self.xlims[1])
        selected_y_indices = np.logical_and(points_xy[:,1] >= self.ylims[0], points_xy[:,1] <= self.ylims[1])

        self.selected_points_indices = list(np.logical_and(selected_x_indices, selected_y_indices))
        points_xy_selected = points_xy[np.array(self.selected_points_indices)]


        self.ax_selected_lidar_points = self.ax.scatter(points_xy_selected[:,0], points_xy_selected[:,1], c='red')

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
    
    def done_callback(self, event):
        if self.selected_points_indices:
            pc2_points = self.image_and_scans.concatenate_scans_to_points(self.scan_indices)
            selected_pc2_points = pc2_points[self.selected_points_indices]
            self.image_and_scans.set_selected_lidar_points(selected_pc2_points)
            self.root.destroy()

    def on_xlims_change(self, event_ax):
        self.xlims = event_ax.get_xlim()

    def on_ylims_change(self, event_ax):
        self.ylims = event_ax.get_ylim()

    def add_figure(self):
        
        self.figure = plt.Figure(figsize=(7, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.chart_type = FigureCanvasTkAgg(self.figure, self.root)
        self.navigation_tool_bar = tkagg.NavigationToolbar2Tk(self.chart_type, self.root)
        self.navigation_tool_bar.zoom()
        self.chart_type.get_tk_widget().pack()
        self.ax.set_title('2D LiDAR Scanned Points')
        self.ax.callbacks.connect('xlim_changed', self.on_xlims_change)
        self.ax.callbacks.connect('ylim_changed', self.on_ylims_change)
        # self.ax.set_axis_off()
        pass

    def clear_selected_2d_lidar_points(self):
        if self.ax_selected_lidar_points:
            self.ax_selected_lidar_points.remove()
        self.ax_selected_lidar_points = None
        self.selected_points_indices = None

    def clear_2d_lidar_points(self):
        if self.ax_lidar_points:
            self.ax_lidar_points.remove()
        self.ax_lidar_points = None
    
    def add_2d_lidar_points(self, points:np.ndarray):
        assert isinstance(points, np.ndarray), "Error: points must be a numpy array."
        points_xy = np.array([[point[0], point[1]] for point in points])
        self.ax_lidar_points = self.ax.scatter(points_xy[:,0], points_xy[:,1], c='blue')

    def run(self) -> ImageAndScans:
        self.app.mainloop()
        assert self.image_and_scans.has_selected_points(), "Error: You must select 2D LiDAR points to proceed."
        return self.image_and_scans
    
class CameraParameters:
    def __init__(self, camera_intrinsic_matrix, k1,k2,p1,p2,k3):
        assert isinstance(camera_intrinsic_matrix, np.ndarray), "Error: camera_intrinsic_matrix must be a matrix."
        assert camera_intrinsic_matrix.shape == (3,3), "Error: camera_intrinsic_matrix must be a 3x3 matrix."
        self.camera_intrinsic_matrix = camera_intrinsic_matrix
        self.distortion_coeffs = np.array([k1,k2,p1,p2,k3])
        
    def get_camera_parameters(self) -> tuple[np.ndarray, list]:
        return (self.camera_intrinsic_matrix.copy(), self.distortion_coeffs.copy())
   
class ChessboardParameters:
    def __init__(self, chessboard_square_size, chessboard_inner_width, chessboard_inner_height):
        self.chessboard_square_size = chessboard_square_size
        self.chessboard_inner_width = chessboard_inner_width
        self.chessboard_inner_height = chessboard_inner_height
        
    def get_chessboard_parameters(self) -> tuple[np.ndarray, list]:
        return (self.chessboard_square_size, self.chessboard_inner_width, self.chessboard_inner_height)
    

class BagToImageAndScans(Node):

    def __init__(self):
        super().__init__('camera_2d_lidar_calibration')
        
        # Declare parameters with default values
        self.declare_parameter("camera_intrinsic_matrix", [1.0,0.0,0.0,
                                                            0.0,1.0,0.0,
                                                            0.0,0.0,1.0])
        self.declare_parameter("distortion_coefficients.k1", 0.0)
        self.declare_parameter("distortion_coefficients.k2", 0.0)
        self.declare_parameter("distortion_coefficients.p1", 0.0)
        self.declare_parameter("distortion_coefficients.p2", 0.0)
        self.declare_parameter("distortion_coefficients.k3", 0.0)
        self.declare_parameter("chessboard_square_size", 2.0)
        self.declare_parameter("chessboard_inner_width", 11)
        self.declare_parameter("chessboard_inner_height", 8)

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("scan_topic", "/scan")


        # Retrieve parameters
        camera_intrinsic_matrix = self.get_parameter("camera_intrinsic_matrix").value
        k1 = self.get_parameter("distortion_coefficients.k1").value
        k2 = self.get_parameter("distortion_coefficients.k2").value
        p1 = self.get_parameter("distortion_coefficients.p1").value
        p2 = self.get_parameter("distortion_coefficients.p2").value
        k3 = self.get_parameter("distortion_coefficients.k3").value
        chessboard_square_size = self.get_parameter("chessboard_square_size").value
        chessboard_inner_width = self.get_parameter("chessboard_inner_width").value
        chessboard_inner_height = self.get_parameter("chessboard_inner_height").value

        self.image_topic = self.get_parameter("image_topic").value
        self.scan_topic = self.get_parameter("scan_topic").value

        self.camera_params = CameraParameters(np.array(camera_intrinsic_matrix).reshape(3,3), k1,k2,p1,p2,k3)
        self.chessboard_params = ChessboardParameters(chessboard_square_size, chessboard_inner_width, chessboard_inner_height)

        self.image_and_scan_list = []

        self.image_publisher = self.create_publisher(Image, '/image', 10) # publisher for real-time image monitoring if necessary
        self.lidar_publisher = self.create_publisher(LaserScan, '/scan', 10) # publisher for real-time lidar monitoring if necessary

        bags_location = f'{home}/ros2_ws/src/camera_2d_lidar_calibration/bags/'

        self.bags = os.listdir(bags_location)
        
        for bag_file_name in self.bags:
            self.get_logger().info('Processing ' + bag_file_name + 'bag file.')
            self.reader = rosbag2_py.SequentialReader()
            storage_options = rosbag2_py.StorageOptions(
                uri=f"{bags_location}{bag_file_name}/",
                storage_id='sqlite3')

            converter_options = rosbag2_py.ConverterOptions('', '')
            self.reader.open(storage_options, converter_options)

            self.image_and_scan_list.append(self.extract_image_and_scans(bag_file_name))

    def extract_image_and_scans(self,bag_file_name) -> ImageAndScans:
        selected_image = False
        image = None
        scans = []
        while self.reader.has_next():
            msg = self.reader.read_next()
            if msg[0] == self.image_topic and selected_image == False: # ignore other ROS topics in this bag
                decoded_data = deserialize_message(msg[1], Image) # get serialized version of message and decode it
                cvbridge = cv_bridge.CvBridge()
                cv_image = cvbridge.imgmsg_to_cv2(decoded_data, desired_encoding='passthrough') # change ROS2 Image to cv2 image
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR) # changes image encoding from RGB to BGR for cv2.imwrite to work correctly
                # cv2.imwrite(file_location, cv_image)
                image = cv_image
                selected_image = True
                self.image_publisher.publish(msg[1])
                self.get_logger().info(f'Published serialized data to /image')
            
            if msg[0] == self.scan_topic:
                decoded_data = deserialize_message(msg[1], LaserScan) # get serialized version of message and decode it
                self.lidar_publisher.publish(msg[1])
                scans.append(decoded_data)      
        return ImageAndScans(image, scans, bag_file_name)

    def get_camera_params(self) -> CameraParameters:
        return self.camera_params
    
    def get_image_and_laser_scans(self) -> list[ImageAndScans]:
        return self.image_and_scan_list.copy()
    
    def get_chessboard_params(self) -> ChessboardParameters:
        return self.chessboard_params


def get_vertical_and_horizontal_lines(img:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Thresholds large and small gradients to detect vertical and horizontal lines respectively. 
    This will fail if the camera is rotated at an angle, so thresholds may require fine-tuning.
    """
    dst = cv2.Canny(img, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    
    vertical_gradient_min = 10 # minimum gradient to be considered a vertical line
    horizontal_gradient_max = 5 # maximum gradient to be considered a horizontal line

    vertical_lines = []
    horizontal_lines = []
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            m = (l[3] - l[1])/(l[2]-l[0]) if l[2]!=l[0] else np.inf
            if np.abs(m) >= vertical_gradient_min:
                vertical_lines.append([(l[0], l[1]), (l[2], l[3])])
                cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
            if np.abs(m) <= horizontal_gradient_max:
                horizontal_lines.append([(l[0], l[1]), (l[2], l[3])])
    return (cdstP, np.array(vertical_lines), np.array(horizontal_lines))

def get_detected_chessboard(img:np.ndarray, camera_params:CameraParameters, chessboard_params:ChessboardParameters):
    """
    Uses real-world properties of chessboard to define a chessboard frame where Z points perpendicular from the chessboard towards camera, X points width-wise and Y points height-wise.
    Computes the extrinsic matrix to the camera frame relative to this chessboard frame.
    """
    square_size, chessboard_inner_width, chesboard_inner_height = chessboard_params.get_chessboard_parameters()
    chessboard_size = (int(chessboard_inner_width), int(chesboard_inner_height))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = (gray.shape[1], gray.shape[0])
    
    # Prepare object points, such as (0,0,0), (1,0,0), (2,0,0) ....,(10,7,0), plus any offsets
    
    # This offsets points so that the chessboard's origin is set to middle point instead of the bottom. If perpendicularity assumption is met, this centre point does not matter. 
    # However, we can heuristically help the estimation process by raising the height corresponding to the plane that the 2D LiDAR is scanning over
    centre_offset = np.min(chessboard_size)*square_size/2 

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size - centre_offset

    # Arrays to store object points and image points from all the images
    # objp = []  # 3D point in real world space
    corners2 = []  # 2D points in image plane


    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    # If corners found, refine and save them
    if ret == True:
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        ret, rvec, tvec = cv2.solvePnP(objp, corners2, *camera_params.get_camera_parameters())

        R, _ = cv2.Rodrigues(rvec)

        # Full extrinsic matrix [R | t]
        extrinsic_matrix = np.vstack([np.hstack((R, tvec)),[0,0,0,1]])

        # Save object and image points
        
        # Draw and display corners
        img_with_corners = img.copy()
        cv2.drawChessboardCorners(img_with_corners, chessboard_size, corners2, ret)
        
        # cv2.imshow("Image with Corners",img_with_corners) # optional, must follow by cv2.waitKey() to show
        return corners2.reshape(len(corners2),-1), objp, extrinsic_matrix
    else:
        return corners2, objp, extrinsic_matrix

def chessboard_pixels_to_camera_frame(image_points, camera_params:CameraParameters, extrinsic_matrix):
    """
    Assumes the n chessboard image points are given in units of pixels, with shape n x 2.
    Example: 
    image_points = np.array([[10, 20], [30,40]], dtype=np.float32)
    """
    K_inv = np.linalg.inv(camera_params.get_camera_parameters()[0])

    # Convert 2D image points to normalized camera coordinates
    image_points_h = np.hstack((image_points, np.ones((image_points.shape[0], 1))))  # (u, v, 1)
    camera_rays = (K_inv @ image_points_h.T).T  # Normalized direction vectors

    chessboard_position = (extrinsic_matrix @ (np.array([0,0,0,1]).reshape(4,1)))[:3,:]
    chessboard_x_vec = (extrinsic_matrix @ (np.array([1,0,0,1]).reshape(4,1)))[:3,:] - chessboard_position
    chessboard_y_vec = (extrinsic_matrix @ (np.array([0,1,0,1]).reshape(4,1)))[:3,:] - chessboard_position
    
    projected_rays = []
    for ray in camera_rays:
        projected_ray = compute_ray_plane_intersection(np.zeros(3), ray, chessboard_position, chessboard_x_vec, chessboard_y_vec)
        projected_rays.append(projected_ray)
    return np.array(projected_rays)


def undistort_image(img:np.ndarray, camera_params:CameraParameters):
    h, w = img.shape[:2]
    newcameramatrix, _ = cv2.getOptimalNewCameraMatrix(
    *camera_params.get_camera_parameters(), (w,h), 1, (w,h)
    )
    undistorted_image = cv2.undistort(
    img, *camera_params.get_camera_parameters(), None, newcameramatrix
    )
    return undistorted_image
    # cv2.imshow("undistorted", undistorted_image)

def compute_ray_plane_intersection(ray_origin, ray, point_on_plane, basis_v1, basis_v2):
    ray_origin = ray_origin.reshape(3,1)
    ray = ray.reshape(3,1)
    point_on_plane = point_on_plane.reshape(3,1)
    basis_v1 = basis_v1.reshape(3,1)
    basis_v2 = basis_v2.reshape(3,1)
    origin_difference = point_on_plane - ray_origin
    ray_trace_matrix = np.hstack([ray, basis_v1, basis_v2])
    coeffs = np.linalg.inv(ray_trace_matrix) @ origin_difference
    projected_ray = coeffs[0] * ray + ray_origin
    return projected_ray.reshape(3)

def first_principal_component(points:np.ndarray):
    """
    Compute first principal component, assuming points are in the form [p1,p2,p3,p4,...] as row vectors
    """
    B = np.array(points)
    mean = np.mean(B, axis=0) 
    B1 = B-mean
    U, S, Vt = np.linalg.svd(B1, full_matrices=True)
    first_component = Vt[0,:]
    std_first_component = S[0]

    return (mean, first_component, std_first_component)

def ransac(points:np.ndarray, plotting=True, ransac_plot_title='RANSAC Regression', ransac_window_title='RANSAC Regression'):
    """
    Applies RANSAC. Chooses dependent variable to be the one with smallest range to avoid potentially infinite gradients. This can fail if the variables have different units.
    """
    ranges = points.max(axis=0) - points.min(axis=0)
    smallest_range_axis = np.argmin(ranges)
    
    x_indices = []
    for i in range(len(ranges)):
        if i != smallest_range_axis:
            x_indices.append(i)
    x_indices = np.array(x_indices)
    X = points[:,x_indices].reshape(len(points),points.shape[1]-1)
    y = points[:,smallest_range_axis]

    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    line_X = np.linspace(X[inlier_mask].min(), X[inlier_mask].max(),2)[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)


    reconstructed_line = np.zeros(shape=(len(line_X),len(ranges)))
    reconstructed_line[:,x_indices] = line_X
    reconstructed_line[:, smallest_range_axis] = line_y_ransac

    # Compare estimated coefficients

    if plotting:
        plt.figure()
        man = plt.get_current_fig_manager()
        man.set_window_title(ransac_window_title)    
        lw = 2
        plt.scatter(
            points[inlier_mask,0], points[inlier_mask,1], color="gold", marker=".", label="Inliers"
        )
        plt.scatter(
            points[outlier_mask,0], points[outlier_mask,1], color="red", marker=".", label="Outliers"
        )
        plt.plot(
            reconstructed_line[:,0],
            reconstructed_line[:,1],
            color="cornflowerblue",
            linewidth=lw,
            label="RANSAC regressor",
        )
        plt.legend(loc="lower right")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(ransac_plot_title)
        plt.show(block=False)
        plt.pause(0.01)
        
    return reconstructed_line


def get_line_end_points(points_on_line) -> tuple[np.ndarray, np.ndarray]:
    mean, first_component, std_first_component = first_principal_component(points_on_line)
    return (mean-std_first_component*first_component, mean+std_first_component*first_component)

def save_camera_lidar_calibration_results(image_and_scan_list:list[ImageAndScans], transformation, scaling_factor):
    """Save camera 2d lidar calibration results"""
    save_location = f'{home}/ros2_ws/src/camera_2d_lidar_calibration/results/'
    if not os.path.exists(save_location):
        os.mkdir(save_location)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = save_location + f'camera_2d_lidar_calibration_result_{timestamp}/'
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    filename = folder_name + f"camera_2d_lidar_calibration_transformation_result_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write("# Camera 2D LiDAR Calibration Results\n\n")
        f.write(f"# ROS Bags Used: {[image_and_scan.get_bag_name() for image_and_scan in image_and_scan_list]}\n\n")
        f.write(f"# Camera-to-LiDAR Scaling Factor (Prior to Transformation): {scaling_factor}\n\n")
        f.write("# Camera-to-LiDAR Transformation Matrix:\n")
        for row in transformation:
            f.write(f"{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}, {row[3]:.6f}\n")

    for image_and_scan in image_and_scan_list:
        cv2.imwrite(folder_name + image_and_scan.get_bag_name() + ".png",image_and_scan.get_undistorted_image())
        selected_lidar_points = image_and_scan.get_selected_lidar_points()
        selected_lidar_points_filename = folder_name +  image_and_scan.get_bag_name() + f"_selected_2d_lidar_points_{timestamp}.txt"
        with open(selected_lidar_points_filename, 'w') as f:
            f.write("('x','y','z','intensity','index')\n")
            for point in selected_lidar_points:
                f.write(str(point) + "\n")

    print(f"Camera 2D LiDAR calibration results saved to {filename}")


def camera_lidar_calibration(camera_params:CameraParameters, chessboard_params:ChessboardParameters, image_and_scan_list:list[ImageAndScans], plotting=True):
    """
    Performs Camera and 2D LiDAR calibration by assuming that the edges of wall correspond to edges of the selected points in scan.
    """

    wall_edge_camera_frame_points = []
    wall_edge_lidar_points = []

    for image_and_scan in image_and_scan_list:

        img = image_and_scan.get_undistorted_image()

        vertical_left_lines = image_and_scan.get_selected_left_lines()
        vertical_right_lines = image_and_scan.get_selected_right_lines()

        vertical_left_line = np.array(get_line_end_points(vertical_left_lines.reshape(-1,2)))
        vertical_right_line =  np.array(get_line_end_points(vertical_right_lines.reshape(-1,2)))

        corners, corners_3d_world, extrinsic_matrix = get_detected_chessboard(img, camera_params, chessboard_params)
        corners_camera_frame = np.array([((extrinsic_matrix @ np.vstack([corner_3d.reshape(3,1),[1]]))[:3,:]).reshape(3) for corner_3d in corners_3d_world])
        vertical_left_line_camera_frame = chessboard_pixels_to_camera_frame(vertical_left_line, camera_params, extrinsic_matrix)
        vertical_right_line_camera_frame = chessboard_pixels_to_camera_frame(vertical_right_line, camera_params, extrinsic_matrix)
        
        chessboard_position = (extrinsic_matrix @ (np.array([0,0,0,1]).reshape(4,1)))[:3,:]
        chessboard_x_vec = (extrinsic_matrix @ (np.array([1,0,0,1]).reshape(4,1)))[:3,:] - chessboard_position
        chessboard_y_vec = (extrinsic_matrix @ (np.array([0,1,0,1]).reshape(4,1)))[:3,:] - chessboard_position
        chessboard_normal = np.cross(chessboard_x_vec.reshape(3), chessboard_y_vec.reshape(3))
        
        line_direction = vertical_left_line_camera_frame[1] - vertical_left_line_camera_frame[0]
        projected_left_ray = compute_ray_plane_intersection(chessboard_position, chessboard_x_vec, vertical_left_line_camera_frame[0], line_direction, chessboard_normal)
        
        line_direction = vertical_right_line_camera_frame[1] - vertical_right_line_camera_frame[0]
        projected_right_ray = compute_ray_plane_intersection(chessboard_position, chessboard_x_vec, vertical_right_line_camera_frame[0], line_direction, chessboard_normal)
        
        selected_lidar_points = image_and_scan.get_selected_lidar_points()
        selected_lidar_points_xy = np.array([[point[0],point[1]] for point in selected_lidar_points])
        lidar_wall_line = ransac(selected_lidar_points_xy, plotting=plotting, ransac_plot_title='RANSAC Detection',ransac_window_title=f"RANSAC Detection - Bag: {image_and_scan.get_bag_name()}")

        lidar_wall_left = lidar_wall_line[0,:]
        lidar_wall_right = lidar_wall_line[1,:]
        # swap lidar points so that they are oriented correctly relative to origin of laser scan
        angle1 = np.arctan2(lidar_wall_left[1], lidar_wall_left[0])
        angle2 = np.arctan2(lidar_wall_right[1], lidar_wall_right[0])
        if angle1 < angle2:
            lidar_wall_left, lidar_wall_right = lidar_wall_right, lidar_wall_left
        

        fig = plt.figure()
        man = plt.get_current_fig_manager()
        man.set_window_title(f"Detected 2D LiDAR Wall - Bag: {image_and_scan.get_bag_name()}")
        ax = fig.add_subplot()
        ax.scatter(selected_lidar_points_xy[:,0],selected_lidar_points_xy[:,1], c='blue', label='Selected Points')
        ax.plot(lidar_wall_line[:,0], lidar_wall_line[:,1], c='cyan', label='Detected Wall')
        ax.scatter(lidar_wall_left[0], lidar_wall_left[1], c='purple', label='Left Edge')
        ax.scatter(lidar_wall_right[0], lidar_wall_right[1], c='orange', label='Right Edge')
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Detected 2D LiDAR Wall')
        if plotting:
            plt.show(block=False)
            plt.pause(0.01)
        # cv2.waitKey()


        fig = plt.figure()
        man = plt.get_current_fig_manager()
        man.set_window_title(f"Detected Points on Wall Edge - Bag: {image_and_scan.get_bag_name()}")
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=-90, azim=-90)
        ax.scatter(corners_camera_frame[:,0],corners_camera_frame[:,1], corners_camera_frame[:,2], c='black', label='Detected Chessboard Corners')
        ax.plot(corners_camera_frame[:,0],corners_camera_frame[:,1], corners_camera_frame[:,2], c='black', label='Chessboard Corner Ordering')
        line = vertical_left_line_camera_frame
        ax.plot(line[:,0],line[:,1], line[:,2], c='red', label='Wall Left Edge')
        ax.scatter(projected_left_ray[0],projected_left_ray[1],projected_left_ray[2], c='purple', label='Projected Left Point')
        line = vertical_right_line_camera_frame
        ax.plot(line[:,0],line[:,1], line[:,2],c='green', label='Wall Right Edge')
        ax.scatter(projected_right_ray[0],projected_right_ray[1],projected_right_ray[2], c='orange',label='Projected Right Point')

        wall_line = np.vstack([projected_left_ray, projected_right_ray])
        ax.plot(wall_line[:,0], wall_line[:,1], wall_line[:,2], c='cyan', label='Detected Wall')

        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title("Detected Points on Wall Edge (Camera Frame)")
        ax.legend()
        # plt.plot()
        if plotting:
            plt.show(block=False)
            plt.pause(0.01)
        
        wall_edge_camera_frame_points.append(projected_left_ray)
        wall_edge_lidar_points.append(lidar_wall_left)

        wall_edge_camera_frame_points.append(projected_right_ray)
        wall_edge_lidar_points.append(lidar_wall_right)

    wall_edge_camera_frame_points = np.array(wall_edge_camera_frame_points)
    wall_edge_lidar_points = np.array(wall_edge_lidar_points)

    wall_edge_lidar_points = np.array([[point[0],point[1], 0.0] for point in wall_edge_lidar_points]) # assume Z = 0 for lidar frame

    rigid_transformation, scale = cv2.estimateAffine3D(wall_edge_camera_frame_points, wall_edge_lidar_points, force_rotation=True)
    transformation = np.vstack([rigid_transformation, np.array([0,0,0,1])])


    # test projection of points from camera frame into lidar frame for last bag

    # test_points = np.array([1/2*(projected_left_ray+projected_right_ray)])
    test_points = np.linspace(projected_left_ray-1*(projected_right_ray-projected_left_ray), projected_right_ray+1*(projected_right_ray-projected_left_ray), 10)
    
    fig = plt.figure()
    man = plt.get_current_fig_manager()
    man.set_window_title(f"Detected Points on Wall Edge - Bag: {image_and_scan.get_bag_name()}")
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=-90, azim=-90)
    ax.scatter(corners_camera_frame[:,0],corners_camera_frame[:,1], corners_camera_frame[:,2], c='black', label='Detected Chessboard Corners')
    ax.plot(corners_camera_frame[:,0],corners_camera_frame[:,1], corners_camera_frame[:,2], c='black', label='Chessboard Corner Ordering')
    line = vertical_left_line_camera_frame
    ax.plot(line[:,0],line[:,1], line[:,2], c='red', label='Wall Left Edge')
    ax.scatter(projected_left_ray[0],projected_left_ray[1],projected_left_ray[2], c='purple', label='Projected Left Point')
    line = vertical_right_line_camera_frame
    ax.plot(line[:,0],line[:,1], line[:,2],c='green', label='Wall Right Edge')
    ax.scatter(projected_right_ray[0],projected_right_ray[1],projected_right_ray[2], c='orange',label='Projected Right Point')
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    ax.scatter(test_points[:,0], test_points[:,1], test_points[:,2], c='lime', label='Test Points')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title("Detected Points on Wall Edge (Camera Frame) - Reprojection Test Points")
    ax.legend()
    # plt.plot()
    if plotting:
        plt.show(block=False)
        plt.pause(0.01)

    transformed_points = []
    for test_point in test_points:
        transformed_point = (transformation @ np.vstack([(scale*test_point).reshape(3,1),[1]])).reshape(4)[:3]
        transformed_points.append(transformed_point)
    transformed_points=np.array(transformed_points)
    
    fig = plt.figure()
    man = plt.get_current_fig_manager()
    man.set_window_title(f"Detected 2D LiDAR Wall - Bag: {image_and_scan.get_bag_name()}")
    ax = fig.add_subplot()
    all_lidar_points = image_and_scan.concatenate_scans_to_points(indices=np.arange(1,len(image_and_scan.get_scans())))
    all_lidar_points_xy = np.array([[point[0],point[1]] for point in all_lidar_points])
    ax.scatter(all_lidar_points_xy[:,0],all_lidar_points_xy[:,1], c='blue', label='All 2D LiDAR Points')
    ax.scatter(selected_lidar_points_xy[:,0],selected_lidar_points_xy[:,1], c='red', label='Selected Points')
    ax.plot(lidar_wall_line[:,0], lidar_wall_line[:,1], c='cyan', label='Detected Wall')
    ax.scatter(lidar_wall_left[0], lidar_wall_left[1], c='purple', label='Left Edge')
    ax.scatter(lidar_wall_right[0], lidar_wall_right[1], c='orange', label='Right Edge')
    ax.scatter(transformed_points[:,0], transformed_points[:,1], c='lime',label='Transformed Test Points') # ignore Z value
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Detected 2D LiDAR Wall - Reprojection Transformed Test Points')
    

    save_camera_lidar_calibration_results(image_and_scan_list, transformation, scale)

    if plotting:
        plt.show()


def main(args=None):
    try:
        rclpy.init(args=args)
        bag_to_image_and_scans = BagToImageAndScans()
        
        camera_params = bag_to_image_and_scans.get_camera_params()
        chessboard_params = bag_to_image_and_scans.get_chessboard_params()
        image_and_scan_list = bag_to_image_and_scans.get_image_and_laser_scans()
        
        for image_and_scan in image_and_scan_list:
            image_and_scan.set_undistorted_image(undistort_image(image_and_scan.get_image(), camera_params))

        updated_image_and_scan_list = []

        for image_and_scan in image_and_scan_list:
            select_points_interface = SelectPointsInterface(image_and_scan)
            image_and_scan = select_points_interface.run()
            updated_image_and_scan_list.append(image_and_scan)

        image_and_scan_list = updated_image_and_scan_list

        updated_image_and_scan_list = []

        for image_and_scan in image_and_scan_list:
            select_lines_interface = SelectLinesInterface(image_and_scan)
            image_and_scan = select_lines_interface.run()
            updated_image_and_scan_list.append(image_and_scan)

        camera_lidar_calibration(camera_params,chessboard_params,updated_image_and_scan_list)

    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()