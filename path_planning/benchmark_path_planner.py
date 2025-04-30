#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import time
import math
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
import threading
import sys
import random


class PathPlannerBenchmark(Node):
    def __init__(self):
        super().__init__("path_planner_benchmark")

        # Constants
        self.TRAJECTORY_TOPIC = "/planned_path"

        # Publishers
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )
        self.goal_pose_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

        self.path_pub = self.create_publisher(
            PoseArray,
            "path_request",
            10
        )

        # Visualization publishers
        self.start_marker_pub = self.create_publisher(
            Marker, "/benchmark/start_pose", 10
        )
        self.goal_marker_pub = self.create_publisher(Marker, "/benchmark/goal_pose", 10)

        # Subscriber for the planned path
        self.get_logger().info(
            f"Subscribing to trajectory topic: {self.TRAJECTORY_TOPIC}"
        )
        self.path_sub = self.create_subscription(
            PoseArray, self.TRAJECTORY_TOPIC, self.path_callback, 10
        )

        # Added subscription for /map data
        self.map = None
        self.map_origin = None
        self.map_resolution = None
        self.map_received = False
        self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)

        # State variables
        self.path_received = False
        self.path = None
        self.planning_start_time = None
        self.planning_end_time = None
        self.metrics = None
        self.path_event = threading.Event()  # New event for path synchronization
        self.successful_trials = 0
        self.timeout_failures = 0
        self.planning_failures = 0

        self.aggregate_metrics = {
        "planning_time": 0.0,
        "path_length": 0.0,
        "total_turning": 0.0,
        "avg_turning_per_meter": 0.0,
        "num_waypoints": 0,
    }

        time.sleep(2.0)

    def publish_sequential_req(self):

        if len(self.possible_points) < 2:
            self.get_logger.info("Not enough points to form a path")
            return

        for i in range(len(self.possible_points) - 1):
            self.planning_start_time = time.time()

            cur_pos = self.possible_points[i]
            goal_pos = self.possible_points[i+1]
            self.publish_path_req(cur_pos, goal_pos)

            # wait some random amount before continuing to simulate real life conditions
            time.sleep(random.uniform(0.0, 10.0))
            

    def publish_path_req(self, curr, goal):
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 0.0

        if len(self.possible_points) < 2:
            self.get_logger.info("Not enough points to form a path")
            return

        poses = []

        cur_pos = Pose()
        cur_pos.position.x = curr[0]
        cur_pos.position.y = curr[1]
        cur_pos.position.z = 0.0
        cur_pos.orientation.x = qx
        cur_pos.orientation.y = qy
        cur_pos.orientation.z = qz
        cur_pos.orientation.w = qw
        poses.append(cur_pos)

        goal_pos = Pose()
        goal_pos.position.x = goal[0]
        goal_pos.position.y = goal[1]
        goal_pos.position.z = 0.0
        goal_pos.orientation.x = qx
        goal_pos.orientation.y = qy
        goal_pos.orientation.z = qz
        goal_pos.orientation.w = qw
        poses.append(goal_pos)

        msg = PoseArray()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.poses = poses

        self.path_pub.publish(msg)

        # Publish visualization marker for the initial pose
        self.publish_start_marker(curr[0], curr[1])
        
        # Publish visualization marker for the goal pose
        self.publish_goal_marker(goal[0], goal[1])

    def publish_start_marker(self, x, y, duration=0.0):
        """Publish a marker for the start pose"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "benchmark"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Set position
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0

        # Set scale
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5

        # Set color (green for start)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Set duration (0 = forever)
        if duration > 0:
            marker.lifetime.sec = int(duration)
            marker.lifetime.nanosec = int((duration - int(duration)) * 1e9)

        self.start_marker_pub.publish(marker)

    def publish_goal_marker(self, x, y, duration=0.0):
        """Publish a marker for the goal pose"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "benchmark"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Set position
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0

        # Set scale
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5

        # Set color (red for goal)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Set duration (0 = forever)
        if duration > 0:
            marker.lifetime.sec = int(duration)
            marker.lifetime.nanosec = int((duration - int(duration)) * 1e9)

        self.goal_marker_pub.publish(marker)

    def path_callback(self, msg):
        """Callback for receiving the planned path"""
        self.planning_end_time = time.time()

        # Check if this is an empty path (failure signal)
        if len(msg.poses) == 0:
            self.get_logger().info("Received empty path (planning failure signal)")
            self.path = []
            self.path_received = True
            self.metrics = None
            self.path_event.set()  # Signal failure
            return

        # Normal path processing for successful plans
        self.get_logger().info(f"Received planned path with {len(msg.poses)} poses")
        self.path = [(pose.position.x, pose.position.y) for pose in msg.poses]
        self.path_received = True

        # Calculate metrics immediately when path is received
        metrics = self.calculate_path_metrics()
        if metrics is not None:
            metrics["planning_time"] = self.planning_end_time - self.planning_start_time
            self.metrics = metrics
            self.path_event.set()  # Signal that path is ready
            self.get_logger().info("==============================================")
            self.get_logger().info(f"Path metrics calculated: {metrics}")
            self.get_logger().info(f"      Planning time: {metrics['planning_time']:.3f}s")
            self.get_logger().info(f"      Path length: {metrics['path_length']:.3f}m")
            self.get_logger().info(f"      Total turning: {metrics['total_turning']:.3f}rad")
            self.get_logger().info(
                f"      Avg turning per meter: {metrics['avg_turning_per_meter']:.3f}rad/m"
            )
            self.get_logger().info(f"      Number of waypoints: {metrics['num_waypoints']}")

            self.aggregate_metrics["planning_time"] += metrics["planning_time"]
            self.aggregate_metrics["path_length"] += metrics["path_length"]
            self.aggregate_metrics["total_turning"] += metrics["total_turning"]
            self.aggregate_metrics["avg_turning_per_meter"] += metrics[
                "avg_turning_per_meter"
            ]
            self.aggregate_metrics["num_waypoints"] += metrics["num_waypoints"]
            self.successful_trials += 1
        elif self.path_received and len(self.path) == 0:
            # Empty path received - explicit planning failure
            self.get_logger().error("Received empty path")
            self.planning_failures += 1
        else:
            # No response received - likely a timeout or communication issue
            self.timeout_failures += 1
            self.get_logger().error("Failed to calculate path metrics")
            self.path_event.set()  # Signal even on failure to prevent hanging

    def calculate_path_metrics(self):
        """Calculate various metrics for the planned path"""

        if self.path is None or len(self.path) < 2:
            return None

        # Calculate path length
        path_length = 0.0
        for i in range(len(self.path) - 1):
            dx = self.path[i + 1][0] - self.path[i][0]
            dy = self.path[i + 1][1] - self.path[i][1]
            path_length += math.sqrt(dx * dx + dy * dy)

        # Calculate total turning
        total_turning = 0.0
        for i in range(1, len(self.path) - 1):
            # Calculate vectors
            v1 = (
                self.path[i][0] - self.path[i - 1][0],
                self.path[i][1] - self.path[i - 1][1],
            )
            v2 = (
                self.path[i + 1][0] - self.path[i][0],
                self.path[i + 1][1] - self.path[i][1],
            )

            # Calculate angle between vectors
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            v1_mag = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1])
            v2_mag = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1])

            if v1_mag > 0 and v2_mag > 0:
                cos_angle = dot_product / (v1_mag * v2_mag)
                # Clamp to avoid numerical errors
                cos_angle = max(-1.0, min(1.0, cos_angle))
                angle = math.acos(cos_angle)
                total_turning += angle

        # Calculate average turning per meter
        avg_turning_per_meter = total_turning / path_length if path_length > 0 else 0

        return {
            "path_length": path_length,
            "total_turning": total_turning,
            "avg_turning_per_meter": avg_turning_per_meter,
            "num_waypoints": len(self.path),
        }

    def map_callback(self, msg):
        self.get_logger().info("Map received")
        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin
        self.map = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_received = True

    def get_random_point(self):
        """Generate a random point in the map bounds by sampling only unoccupied cells"""
        if self.map is None or self.map_origin is None or self.map_resolution is None:
            self.get_logger().warn("Map data not available for random point generation")
            return None
        q = self.map_origin.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        # Find all unoccupied cells (value == 0)
        unoccupied_indices = np.argwhere(self.map == 0)
        if len(unoccupied_indices) == 0:
            return None  # No free space

        # Randomly select one unoccupied cell
        idx = random.randint(0, len(unoccupied_indices) - 1)
        map_y, map_x = unoccupied_indices[idx]

        # Convert map cell to world coordinates
        map_x = float(map_x) * self.map_resolution
        map_y = float(map_y) * self.map_resolution

        world_x = (
            map_x * math.cos(-yaw) - map_y * math.sin(-yaw) + self.map_origin.position.x
        )
        world_y = (
            map_x * math.sin(-yaw) + map_y * math.cos(-yaw) + self.map_origin.position.y
        )
        return (world_x, world_y)

    def run_benchmark(self):
        """Run a single benchmark test"""
        # Reset state
        self.path_received = False
        self.path = None
        self.metrics = None
        self.path_event.clear()  # Clear the event before starting

        # Process any pending callbacks to ensure clean state
        while rclpy.ok() and rclpy.spin_once(self, timeout_sec=0.01):
            pass

        self.possible_points = [
        (0.0, 0.0),
        (-18.206375122070312, -0.38484418392181396),
        (-13.63247013092041, 11.490372657775879), 
        (-20.36231231689453, 25.50994873046875),
        (-21.126, 34.1647),
        ]

        # self.possible_points = [
        # (20.0, 0.0),
        # (-18.206375122070312, -0.38484418392181396),
        # (-13.63247013092041, 11.490372657775879), 
        # (-54.36231231689453, 25.50994873046875),
        # ]
        
        self.get_logger().info("Publishing goal pose")

        self.publish_sequential_req()

        # Spin a few times to ensure the goal is published and initial callbacks are processed
        timeout_start = time.time()
        while rclpy.ok() and time.time() - timeout_start < 0.5:
            rclpy.spin_once(self, timeout_sec=0.1)

        # Wait for path response (success or failure)
        self.get_logger().info("Waiting for planner to complete (success or failure)")

        # Use a much longer safety timeout - this should only trigger if something is broken
        # The planner should always send either a valid path or an empty path to signal completion
        safety_timeout = 60.0  # 1 minute safety timeout

        timeout_start = time.time()
        while (
            rclpy.ok()
            and not self.path_event.is_set()
            and time.time() - timeout_start < safety_timeout
        ):
            # Keep spinning to process callbacks while waiting
            rclpy.spin_once(self, timeout_sec=0.1)

            # Optional: Print a progress message every 10 seconds
            elapsed = time.time() - timeout_start
            if elapsed > 0 and elapsed % 10 < 0.1:
                self.get_logger().info(
                    f"Still waiting for planner after {int(elapsed)}s..."
                )

        if not self.path_event.is_set():
            self.get_logger().error(
                f"Benchmark failed: Safety timeout reached after {safety_timeout} seconds - this indicates a communication issue"
            )
            return None

        # Check if we got a successful path or a failure signal
        if self.metrics is None:
            self.get_logger().info("Planner explicitly reported failure to find a path")
            return None

        self.get_logger().info("Path successfully received and processed")
        return self.metrics

    def is_point_in_free_space(self, x, y):
        """Check if a point is in free space in the map"""
        if self.map is None or self.map_origin is None or self.map_resolution is None:
            self.get_logger().warn("Map data not available for point validation")
            return False

        # Convert world coordinates to map coordinates using proper transformation
        map_coords = self.world_to_map(x, y)
        if map_coords is None:
            return False  # Point is outside map bounds

        map_x, map_y = map_coords
        return self.map[map_y, map_x] == 0  # 0 means free space

    def world_to_map(self, x, y):
        """Convert world coordinates to map coordinates with rotation"""
        if self.map is None or self.map_origin is None or self.map_resolution is None:
            return None

        # First, translate to origin
        dx = x - self.map_origin.position.x
        dy = y - self.map_origin.position.y

        # Then rotate by the map's orientation (using quaternion)
        q = self.map_origin.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        # Rotate the translated point
        rotated_x = dx * math.cos(yaw) - dy * math.sin(yaw)
        rotated_y = dx * math.sin(yaw) + dy * math.cos(yaw)

        # Scale by resolution and convert to integer coordinates
        map_x = int(rotated_x / self.map_resolution)
        map_y = int(rotated_y / self.map_resolution)

        # Check if within map bounds
        if 0 <= map_x < self.map.shape[1] and 0 <= map_y < self.map.shape[0]:
            return (map_x, map_y)
        else:
            return None


def main():
    rclpy.init()
    benchmark = PathPlannerBenchmark()

    # Wait for /map to be received
    print("Waiting for map data...")
    start_wait = time.time()
    while not benchmark.map_received and time.time() - start_wait < 10:
        rclpy.spin_once(benchmark, timeout_sec=0.1)
    if not benchmark.map_received:
        print("Map data not received. Exiting.")
        benchmark.destroy_node()
        rclpy.shutdown()
        return

    print("\n=== Path Planner Benchmark Results ===")
        
    # Track start time for trial duration
    trial_start = time.time()

    metrics = benchmark.run_benchmark()

    # Calculate total trial duration
    trial_duration = time.time() - trial_start

    time.sleep(1.0)  # Cool-down between trials

    print("\n=== Benchmark Summary ===")
    total_completed = benchmark.successful_trials + benchmark.planning_failures + benchmark.timeout_failures
    print(f"Trial duration: {trial_duration}")
    if total_completed > 0:
        print(
            f"Success rate: {benchmark.successful_trials}/{total_completed} trials ({benchmark.successful_trials/total_completed*100:.1f}%)"
        )
        print(
            f"Planning failures: {benchmark.planning_failures}/{total_completed} trials ({benchmark.planning_failures/total_completed*100:.1f}%)"
        )
        print(
            f"Timeout failures: {benchmark.timeout_failures}/{total_completed} trials ({benchmark.timeout_failures/total_completed*100:.1f}%)"
        )
    else:
        print("No trials were completed.")

    benchmark.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()