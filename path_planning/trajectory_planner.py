import rclpy
from rclpy.node import Node
import numpy as np
import random
import math
from scipy.spatial import KDTree

assert rclpy
from geometry_msgs.msg import (
    PoseWithCovarianceStamped,
    PoseStamped,
    PoseArray,
    Point,
    Pose,
)
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory, MapProcessor, PathProcessor


class PathPlan(Node):
    """Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter("odom_topic", "default")
        self.declare_parameter("map_topic", "default")
        self.declare_parameter("initial_pose_topic", "default")

        self.odom_topic = (
            self.get_parameter("odom_topic").get_parameter_value().string_value
        )
        self.map_topic = (
            self.get_parameter("map_topic").get_parameter_value().string_value
        )
        self.initial_pose_topic = (
            self.get_parameter("initial_pose_topic").get_parameter_value().string_value
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid, self.map_topic, self.map_cb, 1
        )

        self.goal_sub = self.create_subscription(
            PoseStamped, "/goal_pose", self.goal_cb, 10
        )

        self.traj_pub = self.create_publisher(PoseArray, "/trajectory/current", 10)

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, self.initial_pose_topic, self.pose_cb, 10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        # RRT* parameters
        self.max_iterations = int(1e4)
        self.step_size = 0.5  # meters
        self.goal_threshold = 0.5  # meters
        self.search_radius = 1.0  # meters
        self.goal_sampling_rate = 0.1  # 10% chance to sample goal
        self.obstacle_threshold = (
            0  # occupancy grid threshold - 0 means any non-free cell is an obstacle
        )

        # State variables
        self.map = None
        self.map_resolution = None
        self.map_origin = None
        self.current_pose = None
        self.goal_pose = None

        # Create map processor with desired parameters
        # Example: 10 pixel dilation radius, no erosion
        self.map_processor = MapProcessor(dilation_radius=10, erosion_radius=0)

        # Create path processor with collision checker
        self.path_processor = PathProcessor(
            collision_checker=self.is_collision_free,
            max_smoothing_iterations=100,
            max_attempts=10,
        )

    def map_cb(self, msg):
        """Store the occupancy grid map"""
        raw_map = np.array(msg.data).reshape((msg.info.height, msg.info.width))

        # Process the map with morphological operations
        self.map = self.map_processor.process_map(raw_map)

        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin

        # Convert quaternion to yaw for debugging
        q = self.map_origin.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        self.get_logger().info(
            f"Map received: shape={self.map.shape}, resolution={self.map_resolution}\n"
            f"Map origin: position=({self.map_origin.position.x}, {self.map_origin.position.y}, {self.map_origin.position.z})\n"
            f"Map orientation: quaternion=({q.x}, {q.y}, {q.z}, {q.w}), yaw={math.degrees(yaw)}°"
        )

    def pose_cb(self, pose):
        """Store the current pose"""
        self.current_pose = pose.pose.pose

    def goal_cb(self, msg):
        """Store the goal pose and trigger path planning"""
        self.goal_pose = msg.pose
        if self.map is None:
            self.get_logger().warn("No map available for planning")
            return

        if self.current_pose is None:
            self.get_logger().warn("No current pose available for planning")
            return

        start_point = (self.current_pose.position.x, self.current_pose.position.y)
        end_point = (self.goal_pose.position.x, self.goal_pose.position.y)

        # Validate start and end points
        if not self.is_point_in_free_space(start_point[0], start_point[1]):
            self.get_logger().warn(f"Start point {start_point} is not in free space")
            return

        if not self.is_point_in_free_space(end_point[0], end_point[1]):
            self.get_logger().warn(f"End point {end_point} is not in free space")
            return

        self.plan_path(start_point, end_point, self.map)

    def world_to_map(self, x, y):
        """Convert world coordinates to map coordinates"""
        if self.map_resolution is None or self.map_origin is None:
            self.get_logger().warn("Map resolution or origin not set")
            return None

        # First, translate to origin
        dx = x - self.map_origin.position.x
        dy = y - self.map_origin.position.y

        # Then rotate by the map's orientation
        # Convert quaternion to yaw
        q = self.map_origin.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        # Rotate the translated point (using positive yaw since we're transforming to map frame)
        rotated_x = dx * math.cos(yaw) - dy * math.sin(yaw)
        rotated_y = dx * math.sin(yaw) + dy * math.cos(yaw)

        # Finally, scale by resolution
        u = int(rotated_x / self.map_resolution)
        v = int(rotated_y / self.map_resolution)

        # Check if within map bounds
        if 0 <= u < self.map.shape[1] and 0 <= v < self.map.shape[0]:
            return (u, v)
        else:
            return None

    def is_point_in_free_space(self, x, y):
        """Check if a world point is in free space"""
        map_point = self.world_to_map(x, y)
        if map_point is None:
            return False

        u, v = map_point
        return self.map[v, u] <= self.obstacle_threshold

    def is_collision_free(self, point1, point2):
        """Check if the path between two points is collision-free"""
        if self.map is None:
            self.get_logger().warn("Map not available for collision checking")
            return False

        # Convert points to map coordinates
        p1 = self.world_to_map(point1[0], point1[1])
        p2 = self.world_to_map(point2[0], point2[1])

        # If either point is outside map bounds, path is not valid
        if p1 is None or p2 is None:
            return False

        # Check if either point is in collision
        if self.map[p1[1], p1[0]] > self.obstacle_threshold:
            return False
        if self.map[p2[1], p2[0]] > self.obstacle_threshold:
            return False

        # Simple line sampling approach with higher sampling rate
        # Calculate number of steps based on distance
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        distance = math.sqrt(dx * dx + dy * dy)
        # Use more samples to ensure we don't miss any cells
        num_steps = max(int(distance * 2), 10)

        for i in range(1, num_steps):  # Skip endpoints which we already checked
            # Interpolate between points
            t = i / num_steps
            x = int(p1[0] + t * dx)
            y = int(p1[1] + t * dy)

            # Check if point is in collision
            if (
                0 <= x < self.map.shape[1]
                and 0 <= y < self.map.shape[0]
                and self.map[y, x] > self.obstacle_threshold
            ):
                return False

        return True

    def get_random_point(self):
        """Generate a random point in the map bounds"""
        if self.map is None:
            return None

        # Get map bounds in world coordinates
        q = self.map_origin.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        # Calculate map dimensions in world coordinates
        map_width = self.map.shape[1] * self.map_resolution
        map_height = self.map.shape[0] * self.map_resolution

        # Generate point in map frame
        map_x = random.uniform(0, map_width)
        map_y = random.uniform(0, map_height)

        # Rotate back to world frame
        world_x = (
            map_x * math.cos(-yaw) - map_y * math.sin(-yaw) + self.map_origin.position.x
        )
        world_y = (
            map_x * math.sin(-yaw) + map_y * math.cos(-yaw) + self.map_origin.position.y
        )

        return (world_x, world_y)

    def get_nearest_node(self, tree, point):
        """Find the nearest node in the tree to the given point"""
        nodes = list(tree.keys())
        if not nodes:
            return None
        kdtree = KDTree(nodes)
        dist, idx = kdtree.query(point)
        return nodes[idx]

    def steer(self, from_point, to_point):
        """Steer from from_point towards to_point with step_size"""
        dist = math.sqrt(
            (to_point[0] - from_point[0]) ** 2 + (to_point[1] - from_point[1]) ** 2
        )
        if dist < self.step_size:
            return to_point
        else:
            theta = math.atan2(to_point[1] - from_point[1], to_point[0] - from_point[0])
            return (
                from_point[0] + self.step_size * math.cos(theta),
                from_point[1] + self.step_size * math.sin(theta),
            )

    def get_near_nodes(self, tree, point):
        """Get all nodes within search_radius of point"""
        nodes = list(tree.keys())
        if not nodes:
            return []
        kdtree = KDTree(nodes)
        indices = kdtree.query_ball_point(point, self.search_radius)
        return [nodes[i] for i in indices]

    def toPoseArray(self, path):
        """Convert path to PoseArray with correct orientations"""
        traj = PoseArray()
        traj.header = self.trajectory.make_header("/map")

        # Calculate orientations based on direction of travel
        for i in range(len(path)):
            pose = Pose()
            pose.position.x = path[i][0]
            pose.position.y = path[i][1]

            # Calculate orientation
            if i < len(path) - 1:
                # Point towards next point
                dx = path[i + 1][0] - path[i][0]
                dy = path[i + 1][1] - path[i][1]
            else:
                # Last point points same as previous
                dx = path[i][0] - path[i - 1][0]
                dy = path[i][1] - path[i - 1][1]

            yaw = math.atan2(dy, dx)
            pose.orientation.z = math.sin(yaw / 2)
            pose.orientation.w = math.cos(yaw / 2)

            traj.poses.append(pose)
        return traj

    def plan_path(self, start_point, end_point, map):
        """Implement RRT* algorithm to find a path from start to end"""
        if self.map is None:
            return

        # Initialize tree with start point
        tree = {start_point: {"parent": None, "cost": 0}}
        iteration = 0

        for _ in range(self.max_iterations):
            iteration += 1

            # Sample random point
            if random.random() < self.goal_sampling_rate:  # 10% chance to sample goal
                rand_point = end_point
            else:
                rand_point = self.get_random_point()
                if rand_point is None:
                    continue

            # Find nearest node
            nearest = self.get_nearest_node(tree, rand_point)
            if nearest is None:
                continue

            # Steer towards random point
            new_point = self.steer(nearest, rand_point)

            # Check if path is collision-free
            if not self.is_collision_free(nearest, new_point):
                continue

            # Find near nodes
            near_nodes = self.get_near_nodes(tree, new_point)

            # Find best parent
            min_cost = tree[nearest]["cost"] + math.sqrt(
                (new_point[0] - nearest[0]) ** 2 + (new_point[1] - nearest[1]) ** 2
            )
            best_parent = nearest

            # Cache collision-free checks
            collision_free_cache = {}
            for near in near_nodes:
                collision_free_cache[near] = self.is_collision_free(near, new_point)

            # Find best parent
            for near in near_nodes:
                if collision_free_cache[near]:
                    cost = tree[near]["cost"] + math.sqrt(
                        (new_point[0] - near[0]) ** 2 + (new_point[1] - near[1]) ** 2
                    )
                    if cost < min_cost:
                        min_cost = cost
                        best_parent = near

            # Add new node to tree
            tree[new_point] = {"parent": best_parent, "cost": min_cost}

            # Rewire tree
            for near in near_nodes:
                if near != best_parent and collision_free_cache[near]:
                    cost = min_cost + math.sqrt(
                        (new_point[0] - near[0]) ** 2 + (new_point[1] - near[1]) ** 2
                    )
                    if cost < tree[near]["cost"]:
                        tree[near]["parent"] = new_point
                        tree[near]["cost"] = cost

            # Check if goal is reached
            if (
                math.sqrt(
                    (new_point[0] - end_point[0]) ** 2
                    + (new_point[1] - end_point[1]) ** 2
                )
                < self.goal_threshold
            ):
                self.get_logger().info(f"Path found after {iteration} iterations!")
                # Reconstruct path
                path = []
                current = new_point
                while current is not None:
                    path.append(current)
                    current = tree[current]["parent"]
                path.reverse()

                # Post-process the path to make it smoother
                smoothed_path = self.path_processor.smooth_path(path)
                self.get_logger().info("Path reconstructed and smoothed.")
                # Convert smoothed path to trajectory
                self.trajectory.clear()
                for point in smoothed_path:
                    self.trajectory.addPoint((point[0], point[1]))

                # Publish trajectory with correct orientations
                self.traj_pub.publish(self.toPoseArray(smoothed_path))
                self.trajectory.publish_viz()
                return

        # If no path found after max iterations
        self.get_logger().warn(
            f"Failed to find path after {self.max_iterations} iterations"
        )


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
