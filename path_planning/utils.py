import rclpy

import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, PoseArray, Point
from std_msgs.msg import Header
import os
from typing import List, Tuple
import json
from skimage.morphology import disk, dilation, erosion
import math

EPSILON = 0.00000000001


class MapProcessor:
    """A class to handle morphological operations on occupancy grids."""

    def __init__(self, dilation_radius=0, erosion_radius=0):
        """
        Initialize the map processor with optional dilation and erosion parameters.

        Args:
            dilation_radius (int): Radius in pixels for obstacle dilation. 0 means no dilation.
            erosion_radius (int): Radius in pixels for obstacle erosion. 0 means no erosion.
        """
        self.dilation_radius = dilation_radius
        self.erosion_radius = erosion_radius

    def process_map(self, occupancy_grid):
        """
        Process the occupancy grid with the configured morphological operations.

        Args:
            occupancy_grid (numpy.ndarray): 2D array representing the occupancy grid
                                          (0 = free, 100 = occupied)

        Returns:
            numpy.ndarray: Processed occupancy grid
        """
        # Convert to binary (0 or 1) for morphological operations
        binary_map = (occupancy_grid > 0).astype(np.uint8)

        # Apply erosion if configured
        if self.erosion_radius > 0:
            selem = disk(self.erosion_radius)
            binary_map = erosion(binary_map, selem)

        # Apply dilation if configured
        if self.dilation_radius > 0:
            selem = disk(self.dilation_radius)
            binary_map = dilation(binary_map, selem)

        # Convert back to original scale (0 or 100)
        return binary_map * 100

    def get_processed_map(self, occupancy_grid):
        """
        Get the processed map with morphological operations applied.
        This is a convenience method that processes the map and returns it.

        Args:
            occupancy_grid (numpy.ndarray): 2D array representing the occupancy grid

        Returns:
            numpy.ndarray: Processed occupancy grid
        """
        return self.process_map(occupancy_grid)


""" These data structures can be used in the search function
"""


class LineTrajectory:
    """A class to wrap and work with piecewise linear trajectories."""

    def __init__(self, node, viz_namespace=None):
        self.points: List[Tuple[float, float]] = []
        self.distances = []
        self.has_acceleration = False
        self.visualize = False
        self.viz_namespace = viz_namespace
        self.node = node

        if viz_namespace:
            self.visualize = True
            self.start_pub = self.node.create_publisher(
                Marker, viz_namespace + "/start_point", 1
            )
            self.traj_pub = self.node.create_publisher(
                Marker, viz_namespace + "/path", 1
            )
            self.end_pub = self.node.create_publisher(
                Marker, viz_namespace + "/end_pose", 1
            )

    # compute the distances along the path for all path segments beyond those already computed
    def update_distances(self):
        num_distances = len(self.distances)
        num_points = len(self.points)

        for i in range(num_distances, num_points):
            if i == 0:
                self.distances.append(0)
            else:
                p0 = self.points[i - 1]
                p1 = self.points[i]
                delta = np.array([p0[0] - p1[0], p0[1] - p1[1]])
                self.distances.append(self.distances[i - 1] + np.linalg.norm(delta))

    def distance_to_end(self, t):
        if not len(self.points) == len(self.distances):
            print(
                "WARNING: Different number of distances and points, this should never happen! Expect incorrect results. See LineTrajectory class."
            )
        dat = self.distance_along_trajectory(t)
        if dat == None:
            return None
        else:
            return self.distances[-1] - dat

    def distance_along_trajectory(self, t):
        # compute distance along path
        # ensure path boundaries are respected
        if t < 0 or t > len(self.points) - 1.0:
            return None
        i = int(t)  # which segment
        t = t % 1.0  # how far along segment
        if t < EPSILON:
            return self.distances[i]
        else:
            return (1.0 - t) * self.distances[i] + t * self.distances[i + 1]

    def addPoint(self, point: Tuple[float, float]) -> None:
        print("adding point to trajectory:", point)
        self.points.append(point)
        self.update_distances()
        self.mark_dirty()

    def clear(self):
        self.points = []
        self.distances = []
        self.mark_dirty()

    def empty(self):
        return len(self.points) == 0

    def save(self, path):
        print("Saving trajectory to:", path)
        data = {}
        data["points"] = []
        for p in self.points:
            data["points"].append({"x": p[0], "y": p[1]})
        with open(path, "w") as outfile:
            json.dump(data, outfile)

    def mark_dirty(self):
        self.has_acceleration = False

    def dirty(self):
        return not self.has_acceleration

    def load(self, path):
        print("Loading trajectory:", path)

        # resolve all env variables in path
        path = os.path.expandvars(path)

        with open(path) as json_file:
            json_data = json.load(json_file)
            for p in json_data["points"]:
                self.points.append((p["x"], p["y"]))
        self.update_distances()
        print("Loaded:", len(self.points), "points")
        self.mark_dirty()

    # build a trajectory class instance from a trajectory message
    def fromPoseArray(self, trajMsg):
        for p in trajMsg.poses:
            self.points.append((p.position.x, p.position.y))
        self.update_distances()
        self.mark_dirty()
        print("Loaded new trajectory with:", len(self.points), "points")

    def toPoseArray(self):
        traj = PoseArray()
        traj.header = self.make_header("/map")
        for i in range(len(self.points)):
            p = self.points[i]
            pose = Pose()
            pose.position.x = p[0]
            pose.position.y = p[1]
            traj.poses.append(pose)
        return traj

    def publish_start_point(self, duration=0.0, scale=0.1):
        should_publish = len(self.points) > 0
        self.node.get_logger().info("Before Publishing start point")
        if self.visualize and self.start_pub.get_subscription_count() > 0:
            self.node.get_logger().info("Publishing start point")
            marker = Marker()
            marker.header = self.make_header("/map")
            marker.ns = self.viz_namespace + "/trajectory"
            marker.id = 0
            marker.type = 2  # sphere
            marker.lifetime = rclpy.duration.Duration(seconds=duration).to_msg()
            if should_publish:
                marker.action = 0
                marker.pose.position.x = self.points[0][0]
                marker.pose.position.y = self.points[0][1]
                marker.pose.orientation.w = 1.0
                marker.scale.x = 1.0
                marker.scale.y = 1.0
                marker.scale.z = 1.0
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
            else:
                # delete marker
                marker.action = 2

            self.start_pub.publish(marker)
        elif self.start_pub.get_subscription_count() == 0:
            self.node.get_logger().info("Not publishing start point, no subscribers")

    def publish_end_point(self, duration=0.0):
        should_publish = len(self.points) > 1
        if self.visualize and self.end_pub.get_subscription_count() > 0:
            marker = Marker()
            marker.header = self.make_header("/map")
            marker.ns = self.viz_namespace + "/trajectory"
            marker.id = 1
            marker.type = 2  # sphere
            marker.lifetime = rclpy.duration.Duration(seconds=duration).to_msg()
            if should_publish:
                marker.action = 0
                marker.pose.position.x = self.points[-1][0]
                marker.pose.position.y = self.points[-1][1]
                marker.pose.orientation.w = 1.0
                marker.scale.x = 1.0
                marker.scale.y = 1.0
                marker.scale.z = 1.0
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
            else:
                # delete marker
                marker.action = 2

            self.end_pub.publish(marker)
        elif self.end_pub.get_subscription_count() == 0:
            print("Not publishing end point, no subscribers")

    def publish_trajectory(self, duration=0.0):
        should_publish = len(self.points) > 1
        if self.visualize and self.traj_pub.get_subscription_count() > 0:
            self.node.get_logger().info("Publishing trajectory")
            marker = Marker()
            marker.header = self.make_header("/map")
            marker.ns = self.viz_namespace + "/trajectory"
            marker.id = 2
            marker.type = marker.LINE_STRIP  # line strip
            marker.lifetime = rclpy.duration.Duration(seconds=duration).to_msg()
            if should_publish:
                marker.action = marker.ADD
                marker.scale.x = 0.3
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 1.0
                marker.color.a = 1.0
                for p in self.points:
                    pt = Point()
                    pt.x = p[0]
                    pt.y = p[1]
                    pt.z = 0.0
                    marker.points.append(pt)
            else:
                # delete
                marker.action = marker.DELETE
            self.traj_pub.publish(marker)
            print("publishing traj")
        elif self.traj_pub.get_subscription_count() == 0:
            print("Not publishing trajectory, no subscribers")

    def publish_viz(self, duration=0):
        if not self.visualize:
            print("Cannot visualize path, not initialized with visualization enabled")
            return
        self.publish_start_point(duration=duration)
        self.publish_trajectory(duration=duration)
        self.publish_end_point(duration=duration)

    def make_header(self, frame_id, stamp=None):
        if stamp == None:
            stamp = self.node.get_clock().now().to_msg()
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        return header


class PathProcessor:
    """A class to handle post-processing and smoothing of paths."""

    def __init__(
        self, collision_checker, max_smoothing_iterations=100, max_attempts=10
    ):
        """
        Initialize the path processor.

        Args:
            collision_checker: Function that checks if a path between two points is collision-free
            max_smoothing_iterations (int): Maximum number of iterations for path smoothing
            max_attempts (int): Maximum number of attempts to find a valid shortcut
        """
        self.collision_checker = collision_checker
        self.max_smoothing_iterations = max_smoothing_iterations
        self.max_attempts = max_attempts

    def smooth_path(self, path):
        """
        Smooth the path using a combination of techniques.

        Args:
            path (list): List of (x,y) points representing the path

        Returns:
            list: Smoothed path
        """
        if len(path) < 3:
            return path

        # First, try to remove unnecessary waypoints
        path = self._remove_redundant_points(path)

        # Then try to find shortcuts
        path = self._find_shortcuts(path)

        return path

    def _remove_redundant_points(self, path):
        """
        Remove points that don't contribute to the path's shape.
        A point is redundant if the path from its previous to next point
        is collision-free.
        """
        if len(path) < 3:
            return path

        new_path = [path[0]]  # Always keep start point
        i = 1

        while i < len(path) - 1:
            # Check if we can skip this point
            if self.collision_checker(new_path[-1], path[i + 1]):
                # Skip this point
                i += 1
            else:
                # Keep this point
                new_path.append(path[i])
                i += 1

        new_path.append(path[-1])  # Always keep end point
        return new_path

    def _find_shortcuts(self, path):
        """
        Try to find shortcuts between non-consecutive points in the path.
        This helps straighten out unnecessary curves.
        """
        if len(path) < 3:
            return path

        new_path = [path[0]]  # Start with first point
        current_idx = 0

        while current_idx < len(path) - 1:
            # Try to find the furthest point we can connect to
            best_idx = current_idx + 1
            attempts = 0

            for i in range(len(path) - 1, current_idx + 1, -1):
                if self.collision_checker(path[current_idx], path[i]):
                    best_idx = i
                    break

                attempts += 1
                if attempts >= self.max_attempts:
                    break

            # Add the best point we found
            new_path.append(path[best_idx])
            current_idx = best_idx

        return new_path

    def _smooth_corners(self, path, max_deviation=0.1):
        """
        Smooth sharp corners in the path by adding intermediate points.
        This helps make turns more gradual.

        Args:
            path (list): List of (x,y) points
            max_deviation (float): Maximum allowed deviation from original path
        """
        if len(path) < 3:
            return path

        new_path = [path[0]]

        for i in range(1, len(path) - 1):
            prev = path[i - 1]
            curr = path[i]
            next_p = path[i + 1]

            # Calculate vectors
            v1 = (curr[0] - prev[0], curr[1] - prev[1])
            v2 = (next_p[0] - curr[0], next_p[1] - curr[1])

            # Calculate angle between vectors
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            det = v1[0] * v2[1] - v1[1] * v2[0]
            angle = math.atan2(det, dot)

            # If the turn is too sharp, add intermediate points
            if abs(angle) > math.pi / 4:  # 45 degrees
                # Add points along a circular arc
                num_points = int(
                    abs(angle) / (math.pi / 8)
                )  # Add points every 22.5 degrees
                for j in range(1, num_points):
                    t = j / num_points
                    # Interpolate position
                    x = curr[0] + max_deviation * math.cos(t * angle)
                    y = curr[1] + max_deviation * math.sin(t * angle)
                    new_path.append((x, y))

            new_path.append(curr)

        new_path.append(path[-1])
        return new_path
