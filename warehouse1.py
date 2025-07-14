# Copyright 2025 NXP

# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
from rclpy.action import ActionClient
from rclpy.parameter import Parameter

import math
import time
import numpy as np
import cv2
from typing import Optional, Tuple
import asyncio
import threading

import tensorflow as tf
import torch

from sensor_msgs.msg import Joy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage

from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped

from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import BehaviorTreeLog
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from synapse_msgs.msg import Status
from synapse_msgs.msg import WarehouseShelf

from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA

import tkinter as tk
from tkinter import ttk

QOS_PROFILE_DEFAULT = 10
SERVER_WAIT_TIMEOUT_SEC = 5.0

PROGRESS_TABLE_GUI = True


class WindowProgressTable:
	def __init__(self, root, shelf_count):
		self.root = root
		self.root.title("Shelf Objects & QR Link")
		self.root.attributes("-topmost", True)

		self.row_count = 2
		self.col_count = shelf_count

		self.boxes = []
		for row in range(self.row_count):
			row_boxes = []
			for col in range(self.col_count):
				box = tk.Text(root, width=10, height=3, wrap=tk.WORD, borderwidth=1,
					      relief="solid", font=("Helvetica", 14))
				box.insert(tk.END, "NULL")
				box.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
				row_boxes.append(box)
			self.boxes.append(row_boxes)

		# Make the grid layout responsive.
		for row in range(self.row_count):
			self.root.grid_rowconfigure(row, weight=1)
		for col in range(self.col_count):
			self.root.grid_columnconfigure(col, weight=1)

	def change_box_color(self, row, col, color):
		self.boxes[row][col].config(bg=color)

	def change_box_text(self, row, col, text):
		self.boxes[row][col].delete(1.0, tk.END)
		self.boxes[row][col].insert(tk.END, text)

box_app = None
def run_gui(shelf_count):
	global box_app
	root = tk.Tk()
	box_app = WindowProgressTable(root, shelf_count)
	root.mainloop()


class WarehouseExplore(Node):
	""" Initializes warehouse explorer node with the required publishers and subscriptions.

		Returns:
			None
	"""
	def __init__(self):
		super().__init__('warehouse_explore')

		self.action_client = ActionClient(
			self,
			NavigateToPose,
			'/navigate_to_pose')

		self.subscription_pose = self.create_subscription(
			PoseWithCovarianceStamped,
			'/pose',
			self.pose_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_global_map = self.create_subscription(
			OccupancyGrid,
			'/global_costmap/costmap',
			self.global_map_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_simple_map = self.create_subscription(
			OccupancyGrid,
			'/map',
			self.simple_map_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_status = self.create_subscription(
			Status,
			'/cerebri/out/status',
			self.cerebri_status_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_behavior = self.create_subscription(
			BehaviorTreeLog,
			'/behavior_tree_log',
			self.behavior_tree_log_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_shelf_objects = self.create_subscription(
			WarehouseShelf,
			'/shelf_objects',
			self.shelf_objects_callback,
			QOS_PROFILE_DEFAULT)

		# Subscription for camera images.
		self.subscription_camera = self.create_subscription(
			CompressedImage,
			'/camera/image_raw/compressed',
			self.camera_image_callback,
			QOS_PROFILE_DEFAULT)

		self.publisher_joy = self.create_publisher(
			Joy,
			'/cerebri/in/joy',
			QOS_PROFILE_DEFAULT)

		# Publisher for output image (for debug purposes).
		self.publisher_qr_decode = self.create_publisher(
			CompressedImage,
			"/debug_images/qr_code",
			QOS_PROFILE_DEFAULT)

		self.publisher_shelf_data = self.create_publisher(
			WarehouseShelf,
			"/shelf_data",
			QOS_PROFILE_DEFAULT)

		self.declare_parameter('shelf_count', 1)
		self.declare_parameter('initial_angle', 0.0)

		self.shelf_count = \
			self.get_parameter('shelf_count').get_parameter_value().integer_value
		self.initial_angle = \
			self.get_parameter('initial_angle').get_parameter_value().double_value

		# --- Robot State ---
		self.armed = False
		self.logger = self.get_logger()

		# --- Robot Pose ---
		self.pose_curr = PoseWithCovarianceStamped()
		self.buggy_pose_x = 0.0
		self.buggy_pose_y = 0.0
		self.buggy_center = (0.0, 0.0)
		self.world_center = (0.0, 0.0)

		# --- Map Data ---
		self.simple_map_curr = None
		self.global_map_curr = None

		# --- Goal Management ---
		self.xy_goal_tolerance = 0.5
		self.goal_completed = True  # No goal is currently in-progress.
		self.goal_handle_curr = None
		self.cancelling_goal = False
		self.recovery_threshold = 10

		# --- Goal Creation ---
		self._frame_id = "map"

		# --- Exploration Parameters ---
		self.max_step_dist_world_meters = 7.0
		self.min_step_dist_world_meters = 4.0
		self.full_map_explored_count = 0

		# --- QR Code Data ---
		self.qr_code_str = "Empty"
		if PROGRESS_TABLE_GUI:
			self.table_row_count = 0
			self.table_col_count = 0

		# --- Shelf Data ---
		self.shelf_objects_curr = WarehouseShelf()

		# --- Shelf Detection Data ---
		self.shelf_detected_poses = []

		# --- YOLOv5 TFLite Initialization ---
		self.interpreter = tf.lite.Interpreter(model_path="yolov5s_f16.tflite")
		self.interpreter.allocate_tensors()
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()
		self.input_size = self.input_details[0]['shape'][1]
		self.int8 = self.input_details[0]["dtype"] == np.uint8

		self.label_names = [
		    "person", "bicycle", "car", "motorcycle",
		    "airplane", "bus", "train", "truck",
		    "boat", "traffic light", "fire hydrant", "stop sign","zebra"
		]

	def preprocess_image(self, image):
		image = cv2.resize(image, (self.input_size, self.input_size))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
		return np.expand_dims(image, axis=0)

	def xywh2xyxy(self, x):
		y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
		y[:, 0] = x[:, 0] - x[:, 2] / 2
		y[:, 1] = x[:, 1] - x[:, 3] / 2
		y[:, 2] = x[:, 0] + x[:, 2] / 2
		y[:, 3] = x[:, 1] + x[:, 3] / 2
		return y

	def box_iou(self, box1, box2):
		def box_area(box):
		    return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
		area1 = box_area(box1)
		area2 = box_area(box2)
		inter = (
		    torch.min(box1[:, None, 2:], box2[:, 2:]) -
		    torch.max(box1[:, None, :2], box2[:, :2])
		).clamp(0).prod(2)
		return inter / (area1[:, None] + area2 - inter)

	def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
		if isinstance(prediction, (list, tuple)):
		    prediction = prediction[0]
		bs = prediction.shape[0]
		nc = prediction.shape[2] - 5
		xc = prediction[..., 4] > conf_thres
		max_wh = 7680
		max_nms = 30000
		output = []
		for xi, x in enumerate(prediction):
		    x = x[xc[xi]]
		    if not x.shape[0]:
		        output.append(torch.zeros((0, 6)))
		        continue
		    x[:, 5:] *= x[:, 4:5]
		    box = self.xywh2xyxy(x[:, :4])
		    conf, j = x[:, 5:].max(1, keepdim=True)
		    x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
		    x = x[x[:, 4].argsort(descending=True)[:max_nms]]
		    c = x[:, 5:6] * max_wh
		    boxes, scores = x[:, :4] + c, x[:, 4]
		    i = []
		    while boxes.size(0):
		        max_idx = scores.argmax()
		        i.append(max_idx.item())
		        if len(i) >= max_det or boxes.size(0) == 1:
		            break
		        iou = self.box_iou(boxes[max_idx].unsqueeze(0), boxes)[0]
		        keep = iou <= iou_thres
		        boxes = boxes[keep]
		        scores = scores[keep]
		        x = x[keep]
		    if i:
		        output.append(x[torch.tensor(i)])
		    else:
		        output.append(torch.zeros((0, 6)))
		return output
	def pose_callback(self, message):
		"""Callback function to handle pose updates.

		Args:
			message: ROS2 message containing the current pose of the rover.

		Returns:
			None
		"""
		self.pose_curr = message
		self.buggy_pose_x = message.pose.pose.position.x
		self.buggy_pose_y = message.pose.pose.position.y
		self.buggy_center = (self.buggy_pose_x, self.buggy_pose_y)

	def simple_map_callback(self, message):
		"""Callback function to handle simple map updates.

		Args:
			message: ROS2 message containing the simple map data.

		Returns:
			None
		"""
		self.simple_map_curr = message
		map_info = self.simple_map_curr.info
		self.world_center = self.get_world_coord_from_map_coord(
			map_info.width / 2, map_info.height / 2, map_info
		)
		
		self.detect_shelves_from_map()
		
	def detect_shelves_from_map(self):
		"""
		Detect shelves from the occupancy grid in self.simple_map_curr.
		Stores detected shelves into self.shelf_detected_poses.
		"""
		map_msg = self.simple_map_curr
		if map_msg is None:
			return
	
		# Extract map info
		height = map_msg.info.height
		width = map_msg.info.width
		res = map_msg.info.resolution
		origin_x = map_msg.info.origin.position.x
		origin_y = map_msg.info.origin.position.y
	
		# Convert map data to numpy
		map_data = np.array(map_msg.data, dtype=np.int8).reshape((height, width))
	
		# Create binary map for occupied cells
		binary_map = (map_data == 100).astype(np.uint8)
	
		# Label connected components
		labeled_map, num_features = label(binary_map)
	
		shelf_poses = []
	
		for label_id in range(1, num_features + 1):
			# Find pixels belonging to this label
			mask = labeled_map == label_id
			points = np.argwhere(mask)
	
			# Filter out small noise clusters
			if len(points) < 50:
				continue
	
			# Compute center of mass in map coords
			cy, cx = points.mean(axis=0)
	
			# PCA to measure length, width, and orientation
			pca = PCA(n_components=2)
			pca.fit(points)
			lengths = 2 * np.sqrt(pca.explained_variance_)
	
			# Convert to meters
			length_x_m = lengths[0] * res
			length_y_m = lengths[1] * res
	
			# Check if dimensions match shelf size
			if (0.4 <= min(length_x_m, length_y_m) <= 0.6 and 1.2 <= max(length_x_m, length_y_m) <= 1.5):
				# Shelf detected!
				orientation = np.arctan2(pca.components_[0][1], pca.components_[0][0])
	
				# Convert center of mass to world coordinates
				world_x = cx * res + origin_x
				world_y = cy * res + origin_y
	
				shelf_poses.append((world_x, world_y, orientation))
	
		self.shelf_detected_poses = shelf_poses
	
		# Log the results
		self.logger.info(f"Detected {len(shelf_poses)} shelf-like objects.")
		# Compute and store goal poses
		self.shelf_goal_poses = self.compute_shelf_viewing_poses()
		for idx, (x, y, theta) in enumerate(shelf_poses):
			self.logger.info(f"Shelf {idx+1}: x={x:.2f}, y={y:.2f}, orientation={theta:.2f} rad")
		
		# Optional debug visualization
		debug_image = cv2.cvtColor(binary_map * 255, cv2.COLOR_GRAY2BGR)
		
		for world_x, world_y, orientation in shelf_poses:
			map_x, map_y = self.get_map_coord_from_world_coord(world_x, world_y, map_msg.info)
			cv2.circle(debug_image, (map_x, map_y), 5, (0, 0, 255), -1)
		
		self.publish_debug_image(self.publisher_qr_decode, debug_image)

	def compute_shelf_viewing_poses(self, distance_qr=1.0, distance_obj=1.0):
	    """
	    Compute robot poses for viewing QR and objects for each detected shelf.
	
	    Args:
	        distance_qr (float): Distance to stand from shelf for QR scanning (meters)
	        distance_obj (float): Distance to stand from shelf for object scanning (meters)
	
	    Returns:
	        List[Tuple[PoseStamped, PoseStamped]]:
	            For each shelf, a tuple (qr_pose, object_pose)
	    """
	    poses = []
	
	    for idx, (shelf_x, shelf_y, shelf_theta) in enumerate(self.shelf_detected_poses):
	        # ---- QR Pose (parallel) ----
	        qr_x = shelf_x + distance_qr * np.cos(shelf_theta)
	        qr_y = shelf_y + distance_qr * np.sin(shelf_theta)
	        qr_yaw = shelf_theta  # parallel
	
	        qr_pose = self.create_goal_from_world_coord(qr_x, qr_y, qr_yaw)
	
	        # ---- Object Pose (perpendicular) ----
	        # Rotate shelf_theta by +90 degrees for perpendicular view
	        object_theta = shelf_theta + np.pi/2
	
	        obj_x = shelf_x + distance_obj * np.cos(object_theta)
	        obj_y = shelf_y + distance_obj * np.sin(object_theta)
	        obj_yaw = object_theta  # facing perpendicular to shelf
	
	        obj_pose = self.create_goal_from_world_coord(obj_x, obj_y, obj_yaw)
	
	        # Save pair
	        poses.append((qr_pose, obj_pose))
	
	        self.logger.info(
	            f"Shelf {idx+1} poses:\n"
	            f"  QR Pose: x={qr_x:.2f}, y={qr_y:.2f}, yaw={qr_yaw:.2f}\n"
	            f"  Obj Pose: x={obj_x:.2f}, y={obj_y:.2f}, yaw={obj_yaw:.2f}"
	        )
	
	    return poses


	def global_map_callback(self, message):
		"""Callback function to handle global map updates.

		Args:
			message: ROS2 message containing the global map data.

		Returns:
			None
		"""
		self.global_map_curr = message
		return
		if not self.goal_completed:
			return

		height, width = self.global_map_curr.info.height, self.global_map_curr.info.width
		map_array = np.array(self.global_map_curr.data).reshape((height, width))

		frontiers = self.get_frontiers_for_space_exploration(map_array)

		map_info = self.global_map_curr.info
		if frontiers:
			closest_frontier = None
			min_distance_curr = float('inf')

			for fy, fx in frontiers:
				fx_world, fy_world = self.get_world_coord_from_map_coord(fx, fy,
											 map_info)
				distance = euclidean((fx_world, fy_world), self.buggy_center)
				if (distance < min_distance_curr and
				    distance <= self.max_step_dist_world_meters and
				    distance >= self.min_step_dist_world_meters):
					min_distance_curr = distance
					closest_frontier = (fy, fx)

			if closest_frontier:
				fy, fx = closest_frontier
				goal = self.create_goal_from_map_coord(fx, fy, map_info)
				self.send_goal_from_world_pose(goal)
				print("Sending goal for space exploration.")
				return
			else:
				self.max_step_dist_world_meters += 2.0
				new_min_step_dist = self.min_step_dist_world_meters - 1.0
				self.min_step_dist_world_meters = max(0.25, new_min_step_dist)

			self.full_map_explored_count = 0
		else:
			self.full_map_explored_count += 1
			print(f"Nothing found in frontiers; count = {self.full_map_explored_count}")

	def get_frontiers_for_space_exploration(self, map_array):
		"""Identifies frontiers for space exploration.

		Args:
			map_array: 2D numpy array representing the map.

		Returns:
			frontiers: List of tuples representing frontier coordinates.
		"""
		frontiers = []
		for y in range(1, map_array.shape[0] - 1):
			for x in range(1, map_array.shape[1] - 1):
				if map_array[y, x] == -1:  # Unknown space and not visited.
					neighbors_complete = [
						(y, x - 1),
						(y, x + 1),
						(y - 1, x),
						(y + 1, x),
						(y - 1, x - 1),
						(y + 1, x - 1),
						(y - 1, x + 1),
						(y + 1, x + 1)
					]

					near_obstacle = False
					for ny, nx in neighbors_complete:
						if map_array[ny, nx] > 0:  # Obstacles.
							near_obstacle = True
							break
					if near_obstacle:
						continue

					neighbors_cardinal = [
						(y, x - 1),
						(y, x + 1),
						(y - 1, x),
						(y + 1, x),
					]

					for ny, nx in neighbors_cardinal:
						if map_array[ny, nx] == 0:  # Free space.
							frontiers.append((ny, nx))
							break

		return frontiers



	def publish_debug_image(self, publisher, image):
		"""Publishes images for debugging purposes.

		Args:
			publisher: ROS2 publisher of the type sensor_msgs.msg.CompressedImage.
			image: Image given by an n-dimensional numpy array.

		Returns:
			None
		"""
		if image.size:
			message = CompressedImage()
			_, encoded_data = cv2.imencode('.jpg', image)
			message.format = "jpeg"
			message.data = encoded_data.tobytes()
			publisher.publish(message)

	def camera_image_callback(self, message):
	    """Callback function to handle incoming camera images and run YOLO inference."""
	    np_arr = np.frombuffer(message.data, np.uint8)
	    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
	
	    # Detect and decode QR code first
	    qr_detector = cv2.QRCodeDetector()
	    data, points, _ = qr_detector.detectAndDecode(image)
	
	    if data:
	        self.qr_code_str = data
	        self.get_logger().info(f"QR Code Data: {data}")
	
	        if points is not None:
	            points = points.astype(int)
	            for i in range(len(points[0])):
	                pt1 = tuple(points[0][i])
	                pt2 = tuple(points[0][(i + 1) % len(points[0])])
	                cv2.line(image, pt1, pt2, color=(0, 255, 0), thickness=3)
	
	    input_img = self.preprocess_image(image)
	
	    if self.int8:
	        scale, zero_point = self.input_details[0]["quantization"]
	        input_img = (input_img / scale + zero_point).astype(np.uint8)
	
	    self.interpreter.set_tensor(self.input_details[0]["index"], input_img)
	    self.interpreter.invoke()
	
	    output_data = []
	    for out in self.output_details:
	        out_tensor = self.interpreter.get_tensor(out["index"])
	        if self.int8:
	            scale, zero_point = out["quantization"]
	            out_tensor = (out_tensor.astype(np.float32) - zero_point) * scale
	        output_data.append(out_tensor)
	
	    detected_objects = {}
	    for pred in output_data:
	        pred[0][..., :4] *= [self.input_size, self.input_size, self.input_size, self.input_size]
	        torch_pred = torch.tensor(pred)
	        detections = self.non_max_suppression(torch_pred, 0.25, 0.45, max_det=1000)
	        for det in detections:
	            if det is None or not len(det):
	                continue
	            for *xyxy, conf, cls in det:
	                label = self.label_names[int(cls)] if int(cls) < len(self.label_names) else str(int(cls))
	                detected_objects[label] = detected_objects.get(label, 0) + 1
	
	                start_point = (int(xyxy[0]), int(xyxy[1]))
	                end_point = (int(xyxy[2]), int(xyxy[3]))
	                cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
	                cv2.putText(image, label, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
	
	    if detected_objects:
	        shelf_data_message = WarehouseShelf()
	        shelf_data_message.object_name = list(detected_objects.keys())
	        shelf_data_message.object_count = list(detected_objects.values())
	        shelf_data_message.qr_decoded = self.qr_code_str
	
	        self.publisher_shelf_data.publish(shelf_data_message)
	
	    # OPTIONAL debug image
	    self.publish_debug_image(self.publisher_qr_decode, image)
	
	    # OPTIONAL GUI update
	    if PROGRESS_TABLE_GUI:
	        obj_str = ""
	        for name, count in detected_objects.items():
	            obj_str += f"{name}: {count}\n"
	        box_app.change_box_text(self.table_row_count, self.table_col_count, obj_str)
	        box_app.change_box_color(self.table_row_count, self.table_col_count, "cyan")
	
	        box_app.change_box_text(self.table_row_count + 1, self.table_col_count, self.qr_code_str)
	        box_app.change_box_color(self.table_row_count + 1, self.table_col_count, "yellow")
	
	        self.table_col_count += 1


	def cerebri_status_callback(self, message):
		"""Callback function to handle cerebri status updates.

		Args:
			message: ROS2 message containing cerebri status.

		Returns:
			None
		"""
		if message.mode == 3 and message.arming == 2:
			self.armed = True
		else:
			# Initialize and arm the CMD_VEL mode.
			msg = Joy()
			msg.buttons = [0, 1, 0, 0, 0, 0, 0, 1]
			msg.axes = [0.0, 0.0, 0.0, 0.0]
			# self.publisher_joy.publish(msg)

	def behavior_tree_log_callback(self, message):
		"""Alternative method for checking goal status.

		Args:
			message: ROS2 message containing behavior tree log.

		Returns:
			None
		"""
		for event in message.event_log:
			if (event.node_name == "FollowPath" and
				event.previous_status == "SUCCESS" and
				event.current_status == "IDLE"):
				# self.goal_completed = True
				# self.goal_handle_curr = None
				pass

	def shelf_objects_callback(self, message):
		"""Callback function to handle shelf objects updates.

		Args:
			message: ROS2 message containing shelf objects data.

		Returns:
			None
		"""
		self.shelf_objects_curr = message
		# Process the shelf objects as needed.

		# How to send WarehouseShelf messages for evaluation.
		
		#Example for sending WarehouseShelf messages for evaluation.
	     '''shelf_data_message = WarehouseShelf()

		shelf_data_message.object_name = ["car", "clock"]
		shelf_data_message.object_count = [1, 2]
		shelf_data_message.qr_decoded = "test qr string"

		self.publisher_shelf_data.publish(shelf_data_message)'''

		"""* Alternatively, you may store the QR for current shelf as self.qr_code_str.
			Then, add it as self.shelf_objects_curr.qr_decoded = self.qr_code_str
			Then, publish as self.publisher_shelf_data.publish(self.shelf_objects_curr)
			This, will publish the current detected objects with the last QR decoded.
		"""

		# Optional code for populating TABLE GUI with detected objects and QR data.
		
		if PROGRESS_TABLE_GUI:
			shelf = self.shelf_objects_curr
			obj_str = ""
			for name, count in zip(shelf.object_name, shelf.object_count):
				obj_str += f"{name}: {count}\n"

			box_app.change_box_text(self.table_row_count, self.table_col_count, obj_str)
			box_app.change_box_color(self.table_row_count, self.table_col_count, "cyan")
			self.table_row_count += 1

			box_app.change_box_text(self.table_row_count, self.table_col_count, self.qr_code_str)
			box_app.change_box_color(self.table_row_count, self.table_col_count, "yellow")
			self.table_row_count = 0
			self.table_col_count += 1
		
		
	def rover_move_manual_mode(self, speed, turn):
		"""Operates the rover in manual mode by publishing on /cerebri/in/joy.

		Args:
			speed: The speed of the car in float. Range = [-1.0, +1.0];
				   Direction: forward for positive, reverse for negative.
			turn: Steer value of the car in float. Range = [-1.0, +1.0];
				  Direction: left turn for positive, right turn for negative.

		Returns:
			None
		"""
		msg = Joy()
		msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
		msg.axes = [0.0, speed, 0.0, turn]
		self.publisher_joy.publish(msg)



	def cancel_goal_callback(self, future):
		"""
		Callback function executed after a cancellation request is processed.

		Args:
			future (rclpy.Future): The future is the result of the cancellation request.
		"""
		cancel_result = future.result()
		if cancel_result:
			self.logger.info("Goal cancellation successful.")
			self.cancelling_goal = False  # Mark cancellation as completed (success).
			return True
		else:
			self.logger.error("Goal cancellation failed.")
			self.cancelling_goal = False  # Mark cancellation as completed (failed).
			return False

	def cancel_current_goal(self):
		"""Requests cancellation of the currently active navigation goal."""
		if self.goal_handle_curr is not None and not self.cancelling_goal:
			self.cancelling_goal = True  # Mark cancellation in-progress.
			self.logger.info("Requesting cancellation of current goal...")
			cancel_future = self.action_client._cancel_goal_async(self.goal_handle_curr)
			cancel_future.add_done_callback(self.cancel_goal_callback)

	def goal_result_callback(self, future):
		"""
		Callback function executed when the navigation goal reaches a final result.

		Args:
			future (rclpy.Future): The future that is result of the navigation action.
		"""
		status = future.result().status
		# NOTE: Refer https://docs.ros2.org/foxy/api/action_msgs/msg/GoalStatus.html.

		if status == GoalStatus.STATUS_SUCCEEDED:
			self.logger.info("Goal completed successfully!")

			# --- NEW: Publish shelf data after goal completion ---
			shelf_data_message = WarehouseShelf()
			shelf_data_message.qr_decoded = self.qr_code_str
			shelf_data_message.object_name = self.shelf_objects_curr.object_name
			shelf_data_message.object_count = self.shelf_objects_curr.object_count

			self.publisher_shelf_data.publish(shelf_data_message)
			self.logger.info("Published shelf data after completing goal.")

		else:
			self.logger.warn(f"Goal failed with status: {status}")

		self.goal_completed = True  # Mark goal as completed.
		self.goal_handle_curr = None  # Clear goal handle.


	

	def goal_response_callback(self, future):
		"""
		Callback function executed after the goal is sent to the action server.

		Args:
			future (rclpy.Future): The future that is server's response to goal request.
		"""
		goal_handle = future.result()
		if not goal_handle.accepted:
			self.logger.warn('Goal rejected :(')
			self.goal_completed = True  # Mark goal as completed (rejected).
			self.goal_handle_curr = None  # Clear goal handle.
		else:
			self.logger.info('Goal accepted :)')
			self.goal_completed = False  # Mark goal as in progress.
			self.goal_handle_curr = goal_handle  # Store goal handle.

			get_result_future = goal_handle.get_result_async()
			get_result_future.add_done_callback(self.goal_result_callback)

	def goal_feedback_callback(self, msg):
		"""
		Callback function to receive feedback from the navigation action.

		Args:
			msg (nav2_msgs.action.NavigateToPose.Feedback): The feedback message.
		"""
		distance_remaining = msg.feedback.distance_remaining
		number_of_recoveries = msg.feedback.number_of_recoveries
		navigation_time = msg.feedback.navigation_time.sec
		estimated_time_remaining = msg.feedback.estimated_time_remaining.sec

		self.logger.debug(f"Recoveries: {number_of_recoveries}, "
				  f"Navigation time: {navigation_time}s, "
				  f"Distance remaining: {distance_remaining:.2f}, "
				  f"Estimated time remaining: {estimated_time_remaining}s")

		if number_of_recoveries > self.recovery_threshold and not self.cancelling_goal:
			self.logger.warn(f"Cancelling. Recoveries = {number_of_recoveries}.")
			self.cancel_current_goal()  # Unblock by discarding the current goal.

	def send_goal_from_world_pose(self, goal_pose):
		"""
		Sends a navigation goal to the Nav2 action server.

		Args:
			goal_pose (geometry_msgs.msg.PoseStamped): The goal pose in the world frame.

		Returns:
			bool: True if the goal was successfully sent, False otherwise.
		"""
		if not self.goal_completed or self.goal_handle_curr is not None:
			return False

		self.goal_completed = False  # Starting a new goal.

		goal = NavigateToPose.Goal()
		goal.pose = goal_pose

		if not self.action_client.wait_for_server(timeout_sec=SERVER_WAIT_TIMEOUT_SEC):
			self.logger.error('NavigateToPose action server not available!')
			return False

		# Send goal asynchronously (non-blocking).
		goal_future = self.action_client.send_goal_async(goal, self.goal_feedback_callback)
		goal_future.add_done_callback(self.goal_response_callback)

		return True



	def _get_map_conversion_info(self, map_info) -> Optional[Tuple[float, float]]:
		"""Helper function to get map origin and resolution."""
		if map_info:
			origin = map_info.origin
			resolution = map_info.resolution
			return resolution, origin.position.x, origin.position.y
		else:
			return None

	def get_world_coord_from_map_coord(self, map_x: int, map_y: int, map_info) \
					   -> Tuple[float, float]:
		"""Converts map coordinates to world coordinates."""
		if map_info:
			resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
			world_x = (map_x + 0.5) * resolution + origin_x
			world_y = (map_y + 0.5) * resolution + origin_y
			return (world_x, world_y)
		else:
			return (0.0, 0.0)

	def get_map_coord_from_world_coord(self, world_x: float, world_y: float, map_info) \
					   -> Tuple[int, int]:
		"""Converts world coordinates to map coordinates."""
		if map_info:
			resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
			map_x = int((world_x - origin_x) / resolution)
			map_y = int((world_y - origin_y) / resolution)
			return (map_x, map_y)
		else:
			return (0, 0)

	def _create_quaternion_from_yaw(self, yaw: float) -> Quaternion:
		"""Helper function to create a Quaternion from a yaw angle."""
		cy = math.cos(yaw * 0.5)
		sy = math.sin(yaw * 0.5)
		q = Quaternion()
		q.x = 0.0
		q.y = 0.0
		q.z = sy
		q.w = cy
		return q

	def create_yaw_from_vector(self, dest_x: float, dest_y: float,
				   source_x: float, source_y: float) -> float:
		"""Calculates the yaw angle from a source to a destination point.
			NOTE: This function is independent of the type of map used.

			Input: World coordinates for destination and source.
			Output: Angle (in radians) with respect to x-axis.
		"""
		delta_x = dest_x - source_x
		delta_y = dest_y - source_y
		yaw = math.atan2(delta_y, delta_x)

		return yaw

	def create_goal_from_world_coord(self, world_x: float, world_y: float,
					 yaw: Optional[float] = None) -> PoseStamped:
		"""Creates a goal PoseStamped from world coordinates.
			NOTE: This function is independent of the type of map used.
		"""
		goal_pose = PoseStamped()
		goal_pose.header.stamp = self.get_clock().now().to_msg()
		goal_pose.header.frame_id = self._frame_id

		goal_pose.pose.position.x = world_x
		goal_pose.pose.position.y = world_y

		if yaw is None and self.pose_curr is not None:
			# Calculate yaw from current position to goal position.
			source_x = self.pose_curr.pose.pose.position.x
			source_y = self.pose_curr.pose.pose.position.y
			yaw = self.create_yaw_from_vector(world_x, world_y, source_x, source_y)
		elif yaw is None:
			yaw = 0.0
		else:  # No processing needed; yaw is supplied by the user.
			pass

		goal_pose.pose.orientation = self._create_quaternion_from_yaw(yaw)

		pose = goal_pose.pose.position
		print(f"Goal created: ({pose.x:.2f}, {pose.y:.2f}, yaw={yaw:.2f})")
		return goal_pose

	def create_goal_from_map_coord(self, map_x: int, map_y: int, map_info,
				       yaw: Optional[float] = None) -> PoseStamped:
		"""Creates a goal PoseStamped from map coordinates."""
		world_x, world_y = self.get_world_coord_from_map_coord(map_x, map_y, map_info)

		return self.create_goal_from_world_coord(world_x, world_y, yaw)


def main(args=None):
	rclpy.init(args=args)

	warehouse_explore = WarehouseExplore()

	if PROGRESS_TABLE_GUI:
		gui_thread = threading.Thread(target=run_gui, args=(warehouse_explore.shelf_count,))
		gui_thread.start()

	rclpy.spin(warehouse_explore)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	warehouse_explore.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
