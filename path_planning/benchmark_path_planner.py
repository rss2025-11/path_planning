#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import time
import math
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
import threading
import sys
import random


class PathPlannerBenchmark(Node):
    def __init__(self):
        super().__init__("path_planner_benchmark")

        # Publishers
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )
        self.goal_pose_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

        # Subscriber for the planned path
        self.path_sub = self.create_subscription(
            PoseArray, "/trajectory/current", self.path_callback, 10
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

        time.sleep(2.0)

    def publish_initial_pose(self, x, y, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        """Publish the initial pose of the robot"""
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        self.initial_pose_pub.publish(msg)
        self.get_logger().info(
            f"Published initial pose: ({x}, {y}) with quaternion ({qx}, {qy}, {qz}, {qw})"
        )

    def publish_goal_pose(self, x, y, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        """Publish the goal pose"""
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        self.planning_start_time = time.time()
        self.goal_pose_pub.publish(msg)
        self.get_logger().info(
            f"Published goal pose: ({x}, {y}) with quaternion ({qx}, {qy}, {qz}, {qw})"
        )

    def path_callback(self, msg):
        """Callback for receiving the planned path"""
        self.get_logger().info("Received planned path")
        self.planning_end_time = time.time()
        self.path = [(pose.position.x, pose.position.y) for pose in msg.poses]
        self.path_received = True

        # Calculate metrics immediately when path is received
        metrics = self.calculate_path_metrics()
        if metrics is not None:
            metrics["planning_time"] = self.planning_end_time - self.planning_start_time
            self.metrics = metrics
        else:
            self.get_logger().error("Failed to calculate path metrics")

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
        if self.map is None:
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

    def run_benchmark(
        self,
        start_x,
        start_y,
        start_qx,
        start_qy,
        start_qz,
        start_qw,
        goal_x,
        goal_y,
        goal_qx,
        goal_qy,
        goal_qz,
        goal_qw,
    ):
        """Run a single benchmark test"""
        # Reset state
        self.path_received = False
        self.path = None
        self.metrics = None

        # Publish poses
        self.publish_initial_pose(
            start_x, start_y, start_qx, start_qy, start_qz, start_qw
        )
        self.publish_goal_pose(goal_x, goal_y, goal_qx, goal_qy, goal_qz, goal_qw)

        start_time = time.time()
        while self.metrics is None and time.time() - start_time < 10:
            rclpy.spin_once(self, timeout_sec=10)
        if self.metrics is None:
            self.get_logger().error("Benchmark failed: Timeout reached")
            return None

        return self.metrics


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

    # Set random seed for reproducibility
    random.seed(13)

    num_trials = 5
    successful_trials = 0
    aggregate_metrics = {
        "planning_time": 0.0,
        "path_length": 0.0,
        "total_turning": 0.0,
        "avg_turning_per_meter": 0.0,
        "num_waypoints": 0,
    }

    print("\n=== Path Planner Benchmark Results ===")
    for i in range(num_trials):
        start_point = benchmark.get_random_point()
        goal_point = benchmark.get_random_point()
        if start_point is None or goal_point is None:
            print(f"Trial {i+1}: Map data unavailable.")
            continue

        print(f"Trial {i+1}: Start {start_point}, Goal {goal_point}")
        metrics = benchmark.run_benchmark(
            start_point[0],
            start_point[1],
            0.0,
            0.0,
            0.0,
            1.0,
            goal_point[0],
            goal_point[1],
            0.0,
            0.0,
            0.0,
            1.0,
        )

        if metrics is not None:
            print(f"    Trial {i+1} Metrics:")
            print(f"      Planning time: {metrics['planning_time']:.3f}s")
            print(f"      Path length: {metrics['path_length']:.3f}m")
            print(f"      Total turning: {metrics['total_turning']:.3f}rad")
            print(
                f"      Avg turning per meter: {metrics['avg_turning_per_meter']:.3f}rad/m"
            )
            print(f"      Number of waypoints: {metrics['num_waypoints']}")

            aggregate_metrics["planning_time"] += metrics["planning_time"]
            aggregate_metrics["path_length"] += metrics["path_length"]
            aggregate_metrics["total_turning"] += metrics["total_turning"]
            aggregate_metrics["avg_turning_per_meter"] += metrics[
                "avg_turning_per_meter"
            ]
            aggregate_metrics["num_waypoints"] += metrics["num_waypoints"]
            successful_trials += 1
        else:
            print(f"    Trial {i+1} failed (timeout or error).")

    print("\n=== Benchmark Summary ===")
    print(f"Success rate: {successful_trials}/{num_trials} trials")
    if successful_trials > 0:
        print("Averages over successful trials:")
        print(
            "  Planning time: {:.3f}s".format(
                aggregate_metrics["planning_time"] / successful_trials
            )
        )
        print(
            "  Path length: {:.3f}m".format(
                aggregate_metrics["path_length"] / successful_trials
            )
        )
        print(
            "  Total turning: {:.3f}rad".format(
                aggregate_metrics["total_turning"] / successful_trials
            )
        )
        print(
            "  Avg turning per meter: {:.3f}rad/m".format(
                aggregate_metrics["avg_turning_per_meter"] / successful_trials
            )
        )
        print(
            "  Number of waypoints: {:.1f}".format(
                aggregate_metrics["num_waypoints"] / successful_trials
            )
        )
    else:
        print("No successful trials.")

    benchmark.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
