import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Point
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan 
from .utils import LineTrajectory, PathProcessor, MapProcessor
import cv2
import os
from PIL import Image

import numpy as np
from tf_transformations import euler_from_quaternion

import math

from queue import PriorityQueue

class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")
        self.declare_parameter('path_topic', "/planned_path")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value
        self.path_topic = self.get_parameter('path_topic').get_parameter_value().string_value
        
        self.cur_pose = None
        self.goal_pose = None

        self.map = None
        self.map_set = False

        self.traversal_rate = 0.125
        self.obstacle_threshold = 0.5
        self.car_buffer = 0.7
        
        # Create path processor with collision checker
        self.path_processor = PathProcessor(
            collision_checker=self.is_collision_free,
            max_smoothing_iterations=100,
            max_attempts=10,
        )

        # Create map processor to dilate map
        self.map_processor = MapProcessor(
            dilation_radius = self.car_buffer
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.traj_pub = self.create_publisher(
            PoseArray,
            self.path_topic,
            10
        )

        self.path_req_sub = self.create_subscription(
            PoseArray,
            "/path_request",
            self.path_req_cb,
            1
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace=self.path_topic)# viz_namespace="/planned_trajectory")
    
    def path_req_cb(self, pts_msg):
        # Clear prexisting trajectory
        self.trajectory.clear()

        target_points = pts_msg.poses
        self.cur_pose = (self.custom_round(target_points[0].position.x, self.traversal_rate), self.custom_round(target_points[0].position.y, self.traversal_rate))
        self.goal_pose = (self.custom_round(target_points[1].position.x, self.traversal_rate), self.custom_round(target_points[1].position.y, self.traversal_rate))

        self.plan_path(self.cur_pose, self.goal_pose)

    def map_cb(self, map_msg):
        
        self.resolution = map_msg.info.resolution  # number pixels per meter

        raw_map = np.array(map_msg.data, np.double).reshape((map_msg.info.height, map_msg.info.width))
        self.map = raw_map

        self.map_width = map_msg.info.width
        self.map_height = map_msg.info.height

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        self.origin_o = map_msg.info.origin.orientation
        origin_o = euler_from_quaternion(
            (self.origin_o.x, self.origin_o.y, self.origin_o.z, self.origin_o.w)
        )
        self.origin = (origin_p.x, origin_p.y, origin_o[2])

        # Make the map set
        self.map_set = True
        self.get_logger().info("Map initialized")
        self.plan_path(self.cur_pose, self.goal_pose)

    def travel_cost(self, cur_pos, next_pos):
        return abs(cur_pos[0] - next_pos[0]) + abs(cur_pos[1] - next_pos[1])

    def plan_path(self, start_point, end_point):
        #Check if map, and at least one set of start/end points exist before running
        if not (self.map_set and self.cur_pose and self.goal_pose):
            if not self.map_set:
                self.get_logger().info("Map is not set")
            elif not self.cur_pose:
                self.get_logger().info("Initial pose is not set")
            else:
                self.get_logger().info("Goal pose is not set")
            return
        
        path = self.run_astar(start_point, end_point)

        smoothed_path = self.path_processor.smooth_path(path)
        # smoothed_path = path
        for point in smoothed_path:
                self.trajectory.addPoint(point)

        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

    def run_astar(self, start_point, end_point):
        path = []

        # Initiate A*
        priority_queue = PriorityQueue() 
        priority_queue.put((0, start_point))
        came_from = dict()
        cost_so_far = dict()
        came_from[start_point] = None
        cost_so_far[start_point] = 0
        
        while not priority_queue.empty():
            cur_priority, cur_pos = priority_queue.get()

            if cur_pos == end_point:
                break

            for neighbor in self.get_neighbors(cur_pos, self.traversal_rate):
                new_cost = cost_so_far[cur_pos] + self.travel_cost(cur_pos, neighbor)
                if (neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]) and self.is_collision_free(cur_pos, neighbor):
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.travel_cost(neighbor, end_point)
                    priority_queue.put((priority, neighbor))
                    came_from[neighbor] = cur_pos
        
        # constructs points of path, starting from the end
        if end_point in came_from:
            cur_pos = end_point
            while (cur_pos is not None):
                path.append((float(cur_pos[0]), float(cur_pos[1])))
                cur_pos = came_from[cur_pos]
        
        path.reverse()
        return path

    def get_neighbors(self, cur_pos, traversal_rate):
        # 4 neighbors
        # return [(cur_pos[0], cur_pos[1] + traversal_rate), # up
        #         (cur_pos[0], cur_pos[1] - traversal_rate), # down
        #         (cur_pos[0] + traversal_rate, cur_pos[1]), # left
        #         (cur_pos[0] - traversal_rate, cur_pos[1] ), # right
        #         ]

        # 8 neighbors
        return [(cur_pos[0], cur_pos[1] + traversal_rate), # up
                (cur_pos[0], cur_pos[1] - traversal_rate), # down
                (cur_pos[0 ] + traversal_rate, cur_pos[1]), # left
                (cur_pos[0] - traversal_rate, cur_pos[1] ), # right
                (cur_pos[0] + traversal_rate, cur_pos[1] + traversal_rate), # up-left
                (cur_pos[0] - traversal_rate, cur_pos[1] + traversal_rate), # up-right
                (cur_pos[0] + traversal_rate, cur_pos[1] + traversal_rate), # down-left
                (cur_pos[0] - traversal_rate, cur_pos[1] + traversal_rate), # down right 
                ]
        
    def world_to_map(self, pos):
         # First, translate to origin
        dx = pos[0] - self.origin[0]
        dy = pos[1] - self.origin[1]

        # Then rotate by the map's orientation
        # Convert quaternion to yaw
        q = self.origin_o
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        # Rotate the translated point (using positive yaw since we're transforming to map frame)
        rotated_x = dx * math.cos(yaw) - dy * math.sin(yaw)
        rotated_y = dx * math.sin(yaw) + dy * math.cos(yaw)

        # Finally, scale by resolution
        u = int(rotated_x / self.resolution)
        v = int(rotated_y / self.resolution)

         # Check if within map bounds
        if 0 <= u < self.map.shape[1] and 0 <= v < self.map.shape[0]:
            return (v, u)
        else:
            self.get_logger().warn(
                f"Point ({v}, {u}) is outside map bounds ({self.map.shape[1]}, {self.map.shape[0]})"
            )
            return None
        
    def custom_round(self, val, nearest_mod):
        """Custom round for more accurate path finding"""
        dif = val%nearest_mod
        return_val = val - dif
        if dif >= nearest_mod/2:
            return return_val
        else :
            return return_val + nearest_mod


    def is_collision_free(self, point1, point2):
        """Check if the path between two points is collision-free"""
        if self.map is None:
            self.get_logger().warn("Map not available for collision checking")
            return False
        
        # if point2 == self.goal_pose:
        #     return True


        # Convert points to map coordinates
        p1 = self.world_to_map(point1)
        p2 = self.world_to_map(point2)

        # If either point is outside map bounds, path is not valid
        if p1 is None or p2 is None:
            return False


        # Simple line sampling approach with higher sampling rate
        # Calculate number of steps based on distance
        dy = p2[0] - p1[0]
        dx = p2[1] - p1[1]
        distance = math.sqrt(dx * dx + dy * dy)
        # Use more samples to ensure we don't miss any cells
        num_steps = max(int(distance * 2), 10)

        for i in range(1, num_steps):  # Skip endpoints which we already checked
            # Interpolate between points
            t = i / num_steps
            y = int(p1[0] + t * dy)
            x = int(p1[1] + t * dx)

            # Check if point is in collision
            if (
                0 <= x < self.map.shape[1]
                and 0 <= y < self.map.shape[0]
                and self.map[y, x] > self.obstacle_threshold
            ):
                return False

        return True


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()