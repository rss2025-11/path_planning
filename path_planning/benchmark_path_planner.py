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

        # Spin until we get the metrics
        while self.metrics is None:
            rclpy.spin_once(self)

        return self.metrics


def main():
    rclpy.init()

    # Create benchmark node
    benchmark = PathPlannerBenchmark()

    # Define test scenario
    start_x = -5.355334281921387
    start_y = -1.5823718309402466
    start_qx = 0.0
    start_qy = 0.0
    start_qz = 0.9873531850164253
    start_qw = 0.15853607803247938

    goal_x = -20.891857147216797
    goal_y = 34.2781867980957
    goal_qx = 0.0
    goal_qy = 0.0
    goal_qz = -0.9964057465521651
    goal_qw = 0.08470884391740079

    # Run benchmark
    print("\n=== Path Planner Benchmark Results ===")
    print(
        "Metrics: planning_time(s), path_length(m), total_turning(rad), avg_turning_per_meter(rad/m), num_waypoints"
    )
    print("-" * 80)

    print(
        f"\nStart: ({start_x}, {start_y}) with quaternion ({start_qx}, {start_qy}, {start_qz}, {start_qw})"
    )
    print(
        f"Goal:  ({goal_x}, {goal_y}) with quaternion ({goal_qx}, {goal_qy}, {goal_qz}, {goal_qw})"
    )

    metrics = benchmark.run_benchmark(
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
    )

    if metrics:
        print(f"Planning time: {metrics['planning_time']:.3f}s")
        print(f"Path length: {metrics['path_length']:.3f}m")
        print(f"Total turning: {metrics['total_turning']:.3f}rad")
        print(f"Avg turning per meter: {metrics['avg_turning_per_meter']:.3f}rad/m")
        print(f"Number of waypoints: {metrics['num_waypoints']}")
    else:
        print("Benchmark failed")

    benchmark.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
