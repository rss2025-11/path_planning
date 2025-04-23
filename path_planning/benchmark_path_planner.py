#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import time
import math
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
import threading
import sys
import random


class PathPlannerBenchmark(Node):
    def __init__(self):
        super().__init__("path_planner_benchmark")

        # Constants
        self.TRAJECTORY_TOPIC = "/trajectory/current"

        # Publishers
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )
        self.goal_pose_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

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

        # Publish visualization marker for the initial pose
        self.publish_start_marker(x, y)

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

        # Publish visualization marker for the goal pose
        self.publish_goal_marker(x, y)

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
            self.get_logger().info(f"Path metrics calculated: {metrics}")
        else:
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
        self.path_event.clear()  # Clear the event before starting

        # Process any pending callbacks to ensure clean state
        while rclpy.ok() and rclpy.spin_once(self, timeout_sec=0.01):
            pass

        # Publish poses
        self.get_logger().info("Publishing initial pose")
        self.publish_initial_pose(
            start_x, start_y, start_qx, start_qy, start_qz, start_qw
        )
        time.sleep(1.0)

        # Process any callbacks that arrived after publishing initial pose
        while rclpy.ok() and rclpy.spin_once(self, timeout_sec=0.01):
            pass

        self.get_logger().info("Publishing goal pose")
        self.publish_goal_pose(goal_x, goal_y, goal_qx, goal_qy, goal_qz, goal_qw)

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

    # Define hardcoded start and end points from user's RViz clicked points
    # Format: (x, y) coordinates in the map frame
    start_points = [
        (-17.70425033569336, -1.57904052734375),  # Start point 1
        (11.074249267578125, -0.5683326721191406),  # Start point 2
        (-34.887779235839844, -0.8166084289550781),  # Start point 3
        (-54.17620086669922, 23.236248016357422),  # Start point 4
        (7.909730911254883, -0.5269126892089844),  # Start point 5
    ]

    end_points = [
        (-55.24238586425781, 17.946285247802734),  # End point 1
        (-14.402801513671875, 12.136001586914062),  # End point 2
        (-16.267887115478516, 25.600914001464844),  # End point 3
        (-15.41672134399414, 11.337997436523438),  # End point 4
        (-23.418113708496094, -1.0195960998535156),  # End point 5
    ]

    # Consistent orientation for all points (quaternion x,y,z,w)
    orientation = (0.0, 0.0, 0.0, 1.0)  # Identity quaternion (no rotation)

    num_trials = len(start_points)  # Number of trials equals number of point pairs
    successful_trials = 0
    planning_failures = 0  # Count explicit planning failures
    timeout_failures = 0  # Count timeout failures
    invalid_points = 0  # Count invalid point pairs

    aggregate_metrics = {
        "planning_time": 0.0,
        "path_length": 0.0,
        "total_turning": 0.0,
        "avg_turning_per_meter": 0.0,
        "num_waypoints": 0,
    }

    print("\n=== Path Planner Benchmark Results ===")
    for i in range(num_trials):
        start_point = start_points[i]
        end_point = end_points[i]

        # Validate points before running benchmark
        if not benchmark.is_point_in_free_space(start_point[0], start_point[1]):
            print(
                f"Trial {i+1}: Start point {start_point} is not in free space. Skipping trial."
            )
            invalid_points += 1
            continue

        if not benchmark.is_point_in_free_space(end_point[0], end_point[1]):
            print(
                f"Trial {i+1}: End point {end_point} is not in free space. Skipping trial."
            )
            invalid_points += 1
            continue

        print(f"Trial {i+1}: Start {start_point}, Goal {end_point}")

        # Track start time for trial duration
        trial_start = time.time()

        metrics = benchmark.run_benchmark(
            start_point[0],  # start_x
            start_point[1],  # start_y
            orientation[0],  # start_qx
            orientation[1],  # start_qy
            orientation[2],  # start_qz
            orientation[3],  # start_qw
            end_point[0],  # end_x
            end_point[1],  # end_y
            orientation[0],  # end_qx
            orientation[1],  # end_qy
            orientation[2],  # end_qz
            orientation[3],  # end_qw
        )

        # Calculate total trial duration
        trial_duration = time.time() - trial_start

        time.sleep(1.0)  # Cool-down between trials

        if metrics is not None:
            # Success case
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
        elif benchmark.path_received and len(benchmark.path) == 0:
            # Empty path received - explicit planning failure
            print(
                f"    Trial {i+1} failed: Planner couldn't find a path (iterations exhausted)"
            )
            print(f"    Total trial duration: {trial_duration:.3f}s")
            planning_failures += 1
        else:
            # No response received - likely a timeout or communication issue
            print(f"    Trial {i+1} failed: No response from planner (timeout)")
            print(f"    Total trial duration: {trial_duration:.3f}s")
            timeout_failures += 1

    print("\n=== Benchmark Summary ===")
    total_completed = successful_trials + planning_failures + timeout_failures
    if total_completed > 0:
        print(
            f"Success rate: {successful_trials}/{total_completed} trials ({successful_trials/total_completed*100:.1f}%)"
        )
        print(
            f"Planning failures: {planning_failures}/{total_completed} trials ({planning_failures/total_completed*100:.1f}%)"
        )
        print(
            f"Timeout failures: {timeout_failures}/{total_completed} trials ({timeout_failures/total_completed*100:.1f}%)"
        )
    else:
        print("No trials were completed.")

    if invalid_points > 0:
        print(f"Invalid point pairs (not in free space): {invalid_points}/{num_trials}")

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
