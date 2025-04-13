import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from nav_msgs.msg import Odometry
from .utils import LineTrajectory
from tf_transformations import euler_from_quaternion
import numpy as np


class PurePursuit(Node):
    """Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed."""

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter("odom_topic", "default")
        self.declare_parameter("drive_topic", "default")

        self.odom_topic = (
            self.get_parameter("odom_topic").get_parameter_value().string_value
        )
        self.drive_topic = (
            self.get_parameter("drive_topic").get_parameter_value().string_value
        )

        self.lookahead = 1.0  # FILL IN #
        self.speed = 1.0  # FILL IN #
        self.wheelbase_length = 0.35  # FILL IN #

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(
            PoseArray, "/trajectory/current", self.trajectory_callback, 1
        )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 1
        )
        self.pose_sub = self.create_subscription(
            Odometry, "/pf/pose/odom", self.pose_callback, 1
        )
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.current_pos = np.array([0.0, 0.0])
        self.trajectory_array = None
        self.end_goal = None

    def minimum_distance_vectorized(self):
        # self.get_logger().info(f'The trajectory array {self.trajectory_array}')
        starts = self.trajectory_array[:-1]
        ends = self.trajectory_array[1:]
        segment_vectors = ends - starts
        segment_lengths = np.diff(
            self.trajectory.distances
        )  # Gives length of each segment
        segment_lengths2 = segment_lengths**2
        # Handle zero-length segments to avoid divide-by-zero
        segment_lengths2 = np.where(segment_lengths2 == 0, 1e-10, segment_lengths2)

        start_to_current_pos = self.current_pos - starts

        # Projection scalar t for each segment
        t = np.sum(start_to_current_pos * segment_vectors, axis=1) / segment_lengths2
        # Clip the value of t to ensure it falls on the finite line segment
        t = np.clip(t, 0.0, 1.0)  # Clamp to [0, 1]

        # Projection point on each segment
        projections = starts + (t[:, np.newaxis] * segment_vectors)  # Shape: (N-1, 2)

        # Distance from current_pos to each projection
        dists = np.linalg.norm(projections - self.current_pos, axis=1)

        # Find minimum distance and segment index
        min_index = np.argmin(dists)
        min_point = projections[min_index]
        return min_point, min_index

    def find_lookahead_point(self, segment_index):
        circle_radius = self.lookahead
        circle_center = self.current_pos
        segments_to_check = self.trajectory_array[segment_index:]

        valid_points = []

        for i in range(len(segments_to_check) - 1):
            segment_start = segments_to_check[i]
            segment_end = segments_to_check[i + 1]
            segment_vector = segment_end - segment_start

            a = np.dot(segment_vector, segment_vector)
            b = 2 * np.dot(segment_vector, segment_start - circle_center)
            c = (
                np.dot(segment_start - circle_center, segment_start - circle_center)
                - circle_radius**2
            )

            discriminant = b**2 - 4 * a * c

            if discriminant < 0:
                continue  # No intersection

            sqrt_discriminant = np.sqrt(discriminant)
            t1 = (-b + sqrt_discriminant) / (2 * a)
            t2 = (-b - sqrt_discriminant) / (2 * a)

            for t in [t1, t2]:
                if 0 <= t <= 1:
                    lookahead_point = segment_start + t * segment_vector
                    valid_points.append((segment_index + i, lookahead_point))

        if valid_points:
            # Pick the one from the furthest segment, or furthest along segment
            best_point = max(valid_points, key=lambda x: x[0])
            return best_point[1]

        self.get_logger().info(
            "No valid lookahead point found: circle does not intersect any segment."
        )
        return None

    def control(self, lookahead_point):
        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = self.get_clock().now().to_msg()

        # Transform lookahead point to robot frame
        dx = lookahead_point[0] - self.current_pos[0]
        dy = lookahead_point[1] - self.current_pos[1]

        # Rotate into robot frame using negative yaw
        local_x = np.cos(-self.current_theta) * dx - np.sin(-self.current_theta) * dy
        local_y = np.sin(-self.current_theta) * dx + np.cos(-self.current_theta) * dy

        # Avoid divide-by-zero
        if local_x <= 0.001:
            local_x = 0.001

        angle_to_goal = np.arctan2(local_y, local_x)
        angle = np.arctan(
            2 * self.wheelbase_length * np.sin(angle_to_goal) / (self.lookahead + 1e-6)
        )

        drive_cmd.drive.speed = self.speed
        drive_cmd.drive.steering_angle = angle
        self.drive_pub.publish(drive_cmd)

    def send_stop_cmd(self):
        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = self.get_clock().now().to_msg()
        drive_cmd.drive.speed = 0.0
        drive_cmd.drive.steering_angle = 0.0
        self.drive_pub.publish(drive_cmd)

    def pose_callback(self, odometry_msg):

        # Access the orientation quaternion
        orientation = odometry_msg.pose.pose.orientation
        # Convert quaternion to Euler angles
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.current_x = odometry_msg.pose.pose.position.x
        self.current_y = odometry_msg.pose.pose.position.y
        self.current_theta = yaw
        self.current_pos[0] = self.current_x
        self.current_pos[1] = self.current_y

        if self.trajectory_array is not None:
            # Check if we already reached the end of the trajectory
            if np.linalg.norm(self.current_pos - self.trajectory_array[-1]) < 0.5:
                self.send_stop_cmd()
            else:
                min_point, segment_idx = self.minimum_distance_vectorized()
                lookahead_point = self.find_lookahead_point(segment_idx)
                if lookahead_point is not None:
                    self.control(lookahead_point)
        # self.get_logger().info(f'Receiving a pose from localization: {self.current_x, self.current_y, self.current_theta}')

    # find the closest point from the robot to a trajectory segment (assuming piecewise linear segments)

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
        self.initialized_traj = True
        self.trajectory_array = np.array(self.trajectory.points)
        self.end_goal = self.trajectory_array[-1]
        # self.get_logger().info(f'type of traj {type(self.trajectory_array), type(self.trajectory_array[0]), self.trajectory_array[0]}')


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
