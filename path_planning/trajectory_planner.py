import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Point
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory

import numpy as np
from tf_transformations import euler_from_quaternion

import cv2
import math


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.cur_start_pose = None
        self.pose_set = False

        self.cur_goal = None
        self.goal_set = False

        self.map = None
        self.map_set = False

        self.traversal_rate = 0.25
        self.obstacle_threshold = self.obstacle_threshold
        self.car_buffer = 0.25

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        # self.map_processor = MapProcessor(dilation_radius=10, erosion_radius=0)


    def map_cb(self, map_msg):
        #Updates Map

        # Convert the map to a numpy array
        # self.map = np.array(map_msg.data, np.double).reshape((map_msg.info.height, map_msg.info.width)) / 100.0
       
        self.resolution = map_msg.info.resolution  # number pixels per meter

        raw_map = np.array(map_msg.data, np.double).reshape((map_msg.info.height, map_msg.info.width))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (math.ceil(self.car_buffer / self.resolution), 
                                                               math.ceil(self.car_buffer / self.resolution)))
        clipped_map = np.clip(raw_map, 0, 1)
        self.map = cv2.morphologyEx(clipped_map, cv2.MORPH_DILATE, kernel)

        # self.map = np.clip(self.map, 0, 1)
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
        print("Map initialized")
        self.plan_path(self.cur_start_pose, self.cur_goal, self.map)


    def pose_cb(self, pose_msg):
        #Reinitializes pose
        self.cur_start_pose = (round(pose_msg.pose.pose.position.x), round(pose_msg.pose.pose.position.y))
        self.pose_set = True
        print("Current Position Located")
        self.plan_path(self.cur_start_pose, self.cur_goal, self.map)

    def goal_cb(self, goal_msg):
        #Reinitializes Goal
        self.cur_goal = (round(goal_msg.pose.position.x) , round(goal_msg.pose.position.y))
        self.goal_set = True
        print("Goal Located")
        self.plan_path(self.cur_start_pose, self.cur_goal, self.map)

    def plan_path(self, start_point, end_point, map):
        #Check if map, pose, and goal exist before running
        if not (self.pose_set and self.goal_set and self.map_set):
            self.get_logger().info("One of starting position, goal, or map is not set")
            return

        #Initiate BFS
        queue = [start_point]
        came_from = dict()
        came_from[start_point] = None
        while len(queue) > 0:
            cur_pos = queue.pop(0)
            if cur_pos == end_point:
                break

            #Consider neighbor(s) if not already considered and not 
            for neighbor in self.get_neighbors(cur_pos, self.traversal_rate):
                if neighbor not in came_from and self.not_wall(neighbor) and self.is_collision_free(cur_pos, neighbor): 
                    came_from[neighbor] = cur_pos
                    queue.append(neighbor)
            
        if end_point in came_from:
            cur_pos = end_point
            while (cur_pos is not None):
                self.trajectory.addPoint((float(cur_pos[0]), float(cur_pos[1])))
                cur_pos = came_from[cur_pos]
        
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

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

    def not_wall(self, pos):
        new_pos = self.world_to_map(pos)
        return self.map[new_pos[0]][new_pos[1]] < .8 if new_pos is not None else False
        
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


    def is_collision_free(self, point1, point2):
        """Check if the path between two points is collision-free"""
        if self.map is None:
            self.get_logger().warn("Map not available for collision checking")
            return False

        # Convert points to map coordinates
        p1 = self.world_to_map(point1)
        p2 = self.world_to_map(point2)

        # If either point is outside map bounds, path is not valid
        if p1 is None or p2 is None:
            return False

        # Check if either point is in collision
        if self.map[p1[0], p1[1]] > self.obstacle_threshold:
            return False
        if self.map[p2[0], p2[1]] > self.obstacle_threshold:
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