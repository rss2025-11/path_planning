import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float32MultiArray, Bool
from rclpy.node import Node
from nav_msgs.msg import Odometry
from .utils import LineTrajectory
from tf_transformations import euler_from_quaternion
import numpy as np
from visualization_msgs.msg import Marker


class PurePursuit(Node):
    """Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed."""

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter("odom_topic", "default")
        self.declare_parameter("drive_topic", "default")
        self.declare_parameter("path_topic", "/planned_path")

        self.odom_topic = (
            self.get_parameter("odom_topic").get_parameter_value().string_value
        )
        self.drive_topic = (
            self.get_parameter("drive_topic").get_parameter_value().string_value
        )
        self.path_topic = (
            self.get_parameter("path_topic").get_parameter_value().string_value
        )

        self.lookahead_baseline = 1.5 # 0.5  # FILL IN #
        self.lookahead = self.lookahead_baseline
        self.speed_baseline = 1.0  # FILL IN #
        self.wheelbase_length = 0.35  # FILL IN #

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(
            PoseArray, self.path_topic, self.trajectory_callback, 1
        )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 1
        )
        self.pose_sub = self.create_subscription(
            Odometry, self.odom_topic, self.pose_callback, 1
        )
        self.exit_sub = self.create_subscription(
            Bool, "/exit_follow", self.exit_callback, 1
        )

        self.viz_pt = self.create_publisher(
            Marker, "/lookahead_viz", 1
        )
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.current_pos = np.array([0.0, 0.0])
        self.trajectory_array = None
        # self.end_goal = None
        self.max_speed = 2.0  # CHANGE HERE
        self.min_speed = 0.0  # CHANGE HERE
        self.min_lookahead = .25 #0.1  # CHANGE HERE
        self.max_lookahead = 2.0  # CHANGE HERE
        self.goal_threshold = 0.5  # CHANGE HERE
        self.reached_end = False

        self.reached_end_pub = self.create_publisher(
            Float32MultiArray, "/reached_end", 1
        )
    
    def exit_callback(self, msg):
        self.trajectory_array = None

    def minimum_distance_vectorized(self):
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
        # Calculate lookahead using distance to the next waypoint
        new_lookahead_val = self._calculate_waypoint_distance_based_lookahead(
            min_index,  # min_index is the index for the start of the segment
        )
        self.lookahead = new_lookahead_val

        return min_point, min_index

    def _calculate_waypoint_distance_based_lookahead(self, min_segment_idx):
        """
        Calculates lookahead distance based on the proximity to the next waypoint.
        Shorter lookahead for closer waypoints (e.g., turns), longer for distant ones.
        """
        # The "upcoming goal point" is the end of the segment identified by min_segment_idx.
        # min_segment_idx is an index for the 'starts' array, so it's the index of the start point of the segment.
        upcoming_waypoint_idx = min_segment_idx + 1

        if upcoming_waypoint_idx < len(self.trajectory_array):
            upcoming_waypoint = self.trajectory_array[upcoming_waypoint_idx]
            dist_to_upcoming_waypoint = np.linalg.norm(
                self.current_pos - upcoming_waypoint
            )
            scaled_dist_to_upcoming_waypoint = dist_to_upcoming_waypoint / 2
            return np.clip(dist_to_upcoming_waypoint, self.min_lookahead, self.max_lookahead)
        else:
            # This case should ideally not be reached if the trajectory has at least 2 points.
            # Fallback to a default lookahead (e.g., baseline or min_lookahead).
            self.get_logger().warn(
                f"Waypoint lookahead: upcoming_waypoint_idx {upcoming_waypoint_idx} "
                f"out of bounds for trajectory length {len(self.trajectory_array)}. Using baseline lookahead."
            )
            return self.lookahead_baseline  # Fallback

    def find_lookahead_point(self, segment_index):
        circle_radius = self.lookahead_baseline
        circle_center = self.current_pos
        segments = self.trajectory_array[segment_index:]

        if len(segments) < 2:
            return None  # Not enough points to define a segment

        starts = segments[:-1]  # (N, 2)
        ends = segments[1:]  # (N, 2)
        vectors = ends - starts  # (N, 2)

        # Coefficients for quadratic intersection equation
        a = np.sum(vectors * vectors, axis=1)  # (N,)
        start_to_center = starts - circle_center  # (N, 2)
        b = 2 * np.sum(vectors * start_to_center, axis=1)  # (N,)
        c = np.sum(start_to_center * start_to_center, axis=1) - circle_radius**2  # (N,)

        discriminant = b**2 - 4 * a * c  # (N,)
        valid = discriminant >= 0

        if not np.any(valid):
            self.get_logger().info(
                "No valid lookahead point found: no segment intersects."
            )
            return None

        # Only compute intersections for valid segments
        a = a[valid]
        b = b[valid]
        c = c[valid]
        vectors = vectors[valid]
        starts = starts[valid]
        segment_idxs = np.arange(segment_index, segment_index + len(valid))[valid]

        sqrt_discriminant = np.sqrt(discriminant[valid])
        t1 = (-b + sqrt_discriminant) / (2 * a)
        t2 = (-b - sqrt_discriminant) / (2 * a)

        # Combine both t1 and t2 into a single array for filtering
        t_all = np.stack([t1, t2], axis=1)  # (N, 2)
        t_mask = (t_all >= 0) & (t_all <= 1)  # Valid t values

        valid_points = []
        for i in range(t_all.shape[0]):
            for j in range(2):
                if t_mask[i, j]:
                    t = t_all[i, j]
                    point = starts[i] + t * vectors[i]
                    valid_points.append((segment_idxs[i], point))

        if valid_points:
            best_point = max(valid_points, key=lambda x: x[0])
            return best_point[1]

        self.get_logger().info(
            "No valid lookahead point found: all intersections out of bounds."
        )
        return None

    def inch_towards_start(self, min_point):
        """
        Move forward slowly until within range of a trajectory.
        """
        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = self.get_clock().now().to_msg()
        dx = min_point[0] - self.current_x
        dy = min_point[1] - self.current_y

        # Rotate into robot frame
        local_x = np.cos(-self.current_theta) * dx - np.sin(-self.current_theta) * dy
        local_y = np.sin(-self.current_theta) * dx + np.cos(-self.current_theta) * dy

        # Compute angle to target in robot frame
        angle_to_goal = np.arctan2(local_y, local_x)

        # Use pure pursuit logic
        angle = np.arctan(
            2 * self.wheelbase_length * np.sin(angle_to_goal) / (self.lookahead + 1e-6)
        )
        drive_cmd.drive.speed = 0.5
        drive_cmd.drive.steering_angle = angle
        self.drive_pub.publish(drive_cmd)

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
            2 * self.wheelbase_length * np.sin(angle_to_goal) / (self.lookahead_baseline + 1e-6) # ALWAYS USE BASELINE LOOKAHEAD
        )
        drive_cmd.drive.speed = 1.0
        drive_cmd.drive.steering_angle = angle
        # self.get_logger().info(f"speed of robot: {scaled_speed}")
        self.drive_pub.publish(drive_cmd)

    def send_stop_cmd(self, trajectory):
        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = self.get_clock().now().to_msg()
        drive_cmd.drive.speed = 0.0
        drive_cmd.drive.steering_angle = 0.0
        self.drive_pub.publish(drive_cmd)

        # reached_end = Float32MultiArray()
        # reached_end.data = [trajectory[0], trajectory[1]]
        # self.reached_end_pub.publish(reached_end)

    def viz_lookahead(self, lookahead_pt):
        marker = Marker()
        marker.header.frame_id = "map" # Set the frame ID
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "my_namespace"
        marker.type = Marker.SPHERE  # Example: SPHERE marker
        marker.action = Marker.ADD

        # Define pose
        marker.pose.position.x = lookahead_pt[0]
        marker.pose.position.y = lookahead_pt[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Define scale
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        # Define color
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.viz_pt.publish(marker)
    # If particle filter is not running, the robot will remain stationary
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
            # self.get_logger().info(f"current pos {self.current_pos}, reached end {self.reached_end}")
            # if (not self.reached_end and 
            #     np.linalg.norm(self.current_pos - self.trajectory_array[-1])
            #     < self.goal_threshold
            # ):
            dist_to_goal = np.linalg.norm(self.current_pos - self.trajectory_array[-1])
            if (
                dist_to_goal < self.goal_threshold
            ):
                self.send_stop_cmd(self.trajectory_array[-1])
            else:
                min_point, segment_idx = self.minimum_distance_vectorized()
                # if dist_to_goal < self.lookahead_baseline:
                #     self.lookahead_baseline = self.lookahead_baseline/2
                lookahead_point = self.find_lookahead_point(segment_idx)
                
                if lookahead_point is not None:
                    # if lookahead_point[0] < 0:
                    #     self.lookahead_baseline = np.linalg.norm(self.current_pos - self.trajectory_array[-1])/2
                        
                    # else:
                    self.viz_lookahead(lookahead_point)
                    self.control(lookahead_point)
            

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
        self.initialized_traj = True
        self.trajectory_array = np.array(self.trajectory.points)
        # self.end_goal = self.trajectory_array[-1]


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()